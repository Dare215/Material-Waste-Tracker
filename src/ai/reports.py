# src/ai/reports.py
import os
from typing import Dict, Tuple, Optional
import pandas as pd

# ---------------------------
# Helpers
# ---------------------------

_PERIODS = {
    "Weekly":   "W-MON",   # week ending Monday (adjust if you prefer SUN)
    "Monthly":  "MS",      # month start
    "Quarterly":"QS",      # quarter start
    "Annual":   "YS",      # year start
}

def _safe_sum(x):
    try:
        return float(pd.to_numeric(x, errors="coerce").fillna(0).sum())
    except Exception:
        return 0.0

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the frame is time-indexed and has a cost column."""
    d = df.copy()
    if "date" not in d.columns:
        raise ValueError("Expected a 'date' column in dataframe.")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")
    # normalize cost/qty
    if "cost" not in d.columns:
        d["cost"] = 0.0
    d["cost"] = pd.to_numeric(d["cost"], errors="coerce").fillna(0.0)

    # boolean columns might not exist; guard for anomaly/subscription flags
    if "ai_anomaly" not in d.columns:
        d["ai_anomaly"] = False
    if "subscription_like" not in d.columns:
        d["subscription_like"] = False
    return d

def _aggregate(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate by the given resample rule ('W-MON','MS','QS','YS')."""
    ts = d.set_index("date").sort_index()
    grp = ts.resample(rule).agg(
        total_cost=("cost", "sum"),
        events=("cost", "count"),
        anomalies=("ai_anomaly", "sum"),
        subs=("subscription_like", "sum"),
    )
    # top drivers (cost by material/vendor if available)
    if "material" in ts.columns:
        top_mat = ts.groupby([pd.Grouper(freq=rule), "material"])["cost"].sum()
        top_mat = top_mat.groupby(level=0).nlargest(5).reset_index(level=0, drop=True)
        grp["top_materials"] = (
            top_mat.groupby(level=0)
            .apply(lambda s: ", ".join([str(k) for k in s.sort_values(ascending=False).index.get_level_values(0)[:5]]))
            .reindex(grp.index)
        )
    else:
        grp["top_materials"] = ""

    if "vendor" in ts.columns:
        top_vendor = ts.groupby([pd.Grouper(freq=rule), "vendor"])["cost"].sum()
        top_vendor = top_vendor.groupby(level=0).nlargest(5).reset_index(level=0, drop=True)
        grp["top_vendors"] = (
            top_vendor.groupby(level=0)
            .apply(lambda s: ", ".join([str(k) for k in s.sort_values(ascending=False).index.get_level_values(0)[:5]]))
            .reindex(grp.index)
        )
    else:
        grp["top_vendors"] = ""

    grp = grp.fillna({"top_materials": "", "top_vendors": ""})
    return grp

def _format_period_label(name: str, idx: pd.Timestamp) -> str:
    if name == "Weekly":
        # show week range (Mon–Sun style label)
        start = idx.normalize()
        end = (start + pd.Timedelta(days=6))
        return f"{start.date()} – {end.date()}"
    if name == "Monthly":
        return idx.strftime("%Y-%m")
    if name == "Quarterly":
        q = ((idx.month - 1) // 3) + 1
        return f"{idx.year} Q{q}"
    if name == "Annual":
        return str(idx.year)
    return str(idx.date())

# ---------------------------
# Template + optional LLM
# ---------------------------

def _template_block(period_name: str, period_label: str, row: pd.Series) -> str:
    total = float(row.get("total_cost", 0.0))
    events = int(row.get("events", 0))
    anomalies = int(row.get("anomalies", 0))
    subs = int(row.get("subs", 0))
    top_m = row.get("top_materials", "") or "n/a"
    top_v = row.get("top_vendors", "") or "n/a"

    return (
        f"{period_name} Report — {period_label}\n"
        f"- Total recorded cost: ${total:,.0f}\n"
        f"- Events recorded: {events}\n"
        f"- Anomalies flagged: {anomalies}\n"
        f"- Potential subscriptions: {subs}\n"
        f"- Top materials (by cost): {top_m}\n"
        f"- Top vendors (by cost): {top_v}\n"
        "Recommendations: Review high-cost anomalies, confirm subscription-like merchants, "
        "and enforce standardized reason codes for better root-cause analytics.\n"
    )

def _try_llm(summary_prompt: str) -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise operations analyst."},
                {"role": "user", "content": summary_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception:
        return None

# ---------------------------
# Public APIs
# ---------------------------

def summarize_period(df: pd.DataFrame, period_name: str) -> Tuple[pd.DataFrame, str]:
    """
    Returns (aggregated_dataframe, concatenated_text_report) for one period.
    period_name in {'Weekly','Monthly','Quarterly','Annual'}.
    """
    if period_name not in _PERIODS:
        raise ValueError(f"Unknown period '{period_name}'. Valid: {list(_PERIODS)}")

    d = _prep(df)
    rule = _PERIODS[period_name]
    agg = _aggregate(d, rule)

    # Build a concise text report for each row and concatenate.
    blocks = []
    for idx, row in agg.iterrows():
        label = _format_period_label(period_name, idx)
        blocks.append(_template_block(period_name, label, row))

    report_text = "\n".join(blocks)

    # Optional LLM enhancement (single overall summary at top)
    # Uses the most recent 6 rows to limit token usage.
    recent = agg.tail(6).copy()
    if not recent.empty:
        md_table = recent[["total_cost", "events", "anomalies", "subs"]].to_markdown()
        prompt = (
            f"Create a succinct executive summary for the following {period_name.lower()} data. "
            "Call out anomalies, subscription-like patterns, and top drivers in <=120 words.\n\n"
            f"{md_table}"
        )
        llm_txt = _try_llm(prompt)
        if llm_txt:
            report_text = f"=== {period_name} Executive Summary ===\n{llm_txt}\n\n{report_text}"

    return agg, report_text

def generate_all_reports(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """
    Generate Weekly, Monthly, Quarterly, Annual reports at once.
    Returns a dict:
    {
      'Weekly':   {'table': DataFrame, 'text': str},
      'Monthly':  {'table': DataFrame, 'text': str},
      'Quarterly':{'table': DataFrame, 'text': str},
      'Annual':   {'table': DataFrame, 'text': str},
    }
    """
    outputs = {}
    for name in ["Weekly", "Monthly", "Quarterly", "Annual"]:
        table, text = summarize_period(df, name)
        outputs[name] = {"table": table, "text": text}
    return outputs
