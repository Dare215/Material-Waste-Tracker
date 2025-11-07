# src/ai/reports.py
import os
from typing import Dict, Tuple, Optional
import pandas as pd

# ---------------------------
# Helpers
# ---------------------------

_PERIODS = {
    "Weekly":    "W-MON",   # week starting Monday
    "Monthly":   "MS",      # month start
    "Quarterly": "QS",      # quarter start
    "Annual":    "YS",      # year start
}

def _safe_sum(x):
    try:
        return float(pd.to_numeric(x, errors="coerce").fillna(0).sum())
    except Exception:
        return 0.0

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the frame is time-indexed and has normalized columns we expect."""
    d = df.copy()
    if "date" not in d.columns:
        raise ValueError("Expected a 'date' column in dataframe.")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")

    # normalize cost
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

    # Top drivers (cost by material/vendor if available)
    if "material" in ts.columns:
        top_mat = ts.groupby([pd.Grouper(freq=rule), "material"])["cost"].sum()
        # For each period, take top 5 materials by cost and join as a label
        grp["top_materials"] = (
            top_mat.groupby(level=0)
            .apply(lambda s: ", ".join(
                [str(idx) for idx, _ in s.sort_values(ascending=False).head(5).items()]
            ))
            .reindex(grp.index)
        )
    else:
        grp["top_materials"] = ""

    if "vendor" in ts.columns:
        top_vendor = ts.groupby([pd.Grouper(freq=rule), "vendor"])["cost"].sum()
        grp["top_vendors"] = (
            top_vendor.groupby(level=0)
            .apply(lambda s: ", ".join(
                [str(idx) for idx, _ in s.sort_values(ascending=False).head(5).items()]
            ))
            .reindex(grp.index)
        )
    else:
        grp["top_vendors"] = ""

    grp = grp.fillna({"top_materials": "", "top_vendors": ""})
    return grp

def _format_period_label(name: str, idx: pd.Timestamp) -> str:
    """Human-friendly label for each period index."""
    if name == "Weekly":
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
# Fixed summary template
# ---------------------------

def _template_block(period_name: str, period_label: str, row: pd.Series) -> str:
    """
    Fixed (hard-coded) narrative; only metrics/labels change.
    This text is intended to be rendered with st.code()/st.markdown in the UI
    so users cannot edit it.
    """
    total = float(row.get("total_cost", 0.0))
    events = int(row.get("events", 0))
    anomalies = int(row.get("anomalies", 0))
    subs = int(row.get("subs", 0))
    top_m = (row.get("top_materials", "") or "N/A")
    top_v = (row.get("top_vendors", "") or "N/A")

    return (
f"""{period_name} Waste Report — {period_label}
------------------------------------------------
Summary
• Total recorded cost: ${total:,.2f}
• Events recorded: {events}
• Anomalies flagged: {anomalies}
• Potential subscriptions: {subs}

Operational Notes
• Investigate highest-cost days/materials first and document root causes in the SOP.
• Validate anomaly flags before escalation; attach evidence to the ticket.
• Confirm inventory/ordering adjustments prior to the next purchase cycle.

Top Signals
• Top materials (by cost): {top_m}
• Top vendors (by cost):   {top_v}

Governance
• This is a fixed report format; narrative text is not user-editable.
• Use for high-level review; final decisions must reference raw event logs."""
    )

# ---------------------------
# Public APIs
# ---------------------------

def summarize_period(df: pd.DataFrame, period_name: str) -> Tuple[pd.DataFrame, str]:
    """
    Returns (aggregated_dataframe, concatenated_text_report) for one period.
    period_name in {'Weekly','Monthly','Quarterly','Annual'}.

    The text report uses a fixed template to prevent user-edited narratives.
    """
    if period_name not in _PERIODS:
        raise ValueError(f"Unknown period '{period_name}'. Valid: {list(_PERIODS)}")

    d = _prep(df)
    rule = _PERIODS[period_name]
    agg = _aggregate(d, rule)

    # Build a fixed text report for each row and concatenate.
    blocks = []
    for idx, row in agg.iterrows():
        label = _format_period_label(period_name, idx)
        blocks.append(_template_block(period_name, label, row))

    report_text = "\n".join(blocks)
    return agg, report_text

def generate_all_reports(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """
    Generate Weekly, Monthly, Quarterly, Annual reports at once.
    Returns a dict:
    {
      'Weekly':    {'table': DataFrame, 'text': str},
      'Monthly':   {'table': DataFrame, 'text': str},
      'Quarterly': {'table': DataFrame, 'text': str},
      'Annual':    {'table': DataFrame, 'text': str},
    }
    """
    outputs = {}
    for name in ["Weekly", "Monthly", "Quarterly", "Annual"]:
        table, text = summarize_period(df, name)
        outputs[name] = {"table": table, "text": text}
    return outputs
