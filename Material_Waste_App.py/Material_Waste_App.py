# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# material_waste_app.py
# Material Waste Tracker — Scan-Free Edition
# Features kept: roles (Manager-only cost), mock hard-coded prices, suites (Commercial/Clinical/C & D Spaces),
# per-entry CSV + master CSV, SOP links, audit hash chain, dashboards, heatmap, and
# exports (daily/weekly/monthly/quarterly/annual) + optional monthly email.

# Material_Waste_App.py
# Material Waste Tracker — Scan-Free Edition
# Adds: per-suite material grids, per-material suite grids, overall material line chart,
# SOP URL nan fix, export download buttons, and previous export bugfixes.

# Material_Waste_App.py
# Material Waste Tracker — Scan-Free Edition
# Updates:
# - Removed Line/Room/Site inputs & filters; kept suite
# - Quantity is integer-only
# - "Employee/Operator ID" -> "Operator Initial/date"
# - Shift options updated to requested set
# - Replaced old shift×line heatmap with simple per-shift bar chart
# - Exports bug fix + download buttons retained

# Material_Waste_App.py
# Material Waste Tracker — Scan-Free Edition
# Includes:
# - Per-suite & per-material faceted charts + overall line chart (Manager toggle: Qty/Cost)
# - Exports: fixed st.success() Path->str + download buttons
# - Removed Room/Line/Site inputs & filters; keep columns blank for schema compatibility
# - Quantity integer only
# - "Employee/Operator ID" -> "Operator Initial/date"
# - Shift list updated per request

# Material_Waste_App.py
# Material Waste Tracker — Scan-Free Edition
# Includes:
# - Per-suite & per-material faceted charts + overall line chart (Manager toggle: Qty/Cost)
# - Exports: fixed st.success() Path->str + download buttons
# - Removed Room/Line/Site inputs & filters; keep columns blank for schema compatibility
# - Quantity integer only
# - "Employee/Operator ID" -> "Operator Initial/date"
# - Shift list updated per request

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, date, timedelta
from pathlib import Path
from uuid import uuid4
import hashlib
import re
import smtplib
from email.message import EmailMessage
import sys

# --- Robust path setup so Streamlit Cloud finds src/ modules ---
_THIS_FILE = Path(__file__).resolve()
for base in {_THIS_FILE.parent, _THIS_FILE.parent.parent, Path.cwd()}:
    cand = base / "src"
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
# ----------------------------------------------------------------

# === AI imports (after path fix) ===
from src.ai.anomaly import flag_anomalies
from src.ai.duplicates import detect_duplicates_and_subs
from src.ai.categorize import categorize_basic
from src.ai.trends import aggregate_timeseries, naive_forecast
from src.ai.reports import generate_all_reports
# ===================================

st.set_page_config(page_title="Material Waste Tracker", page_icon="♻️", layout="wide")

# ------------------------- Paths -------------------------
DATA_DIR = Path("waste_logs"); DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR = DATA_DIR / "images"; IMAGES_DIR.mkdir(exist_ok=True)
EXPORTS_DIR = DATA_DIR / "exports"; EXPORTS_DIR.mkdir(exist_ok=True)

MASTER_CSV = DATA_DIR / "waste_master.csv"
MATERIALS_CSV = DATA_DIR / "materials_catalog.csv"
SOP_CSV = DATA_DIR / "sop_links.csv"
MOCK_PRICE_CSV = DATA_DIR / "mock_price_table.csv"
LAST_EMAIL_FLAG = DATA_DIR / "last_monthly_email.txt"

# ------------------------- Suites -------------------------
COMMERCIAL_SUITES = [f"Commercial-{i}" for i in range(1601, 1613)]  # 1601-1612
CLINICAL_SUITES = [f"Clinical-{i}" for i in range(1501, 1505)]      # 1501-1504
OTHER_SPACES = ["C Space", "D Space"]
ALL_SUITES = COMMERCIAL_SUITES + CLINICAL_SUITES + OTHER_SPACES

# ------------------------- Defaults -------------------------
DEFAULT_MATERIALS = pd.DataFrame([
    {"name": "Brauns Red Caps",           "unit": "ea", "cost_per_unit": 0.35},
    {"name": "50 mL Conical Tubes",       "unit": "ea", "cost_per_unit": 0.20},
    {"name": "EV3000 Bag",                "unit": "ea", "cost_per_unit": 65.00},
    {"name": "5L Labtainer Bags",         "unit": "ea", "cost_per_unit": 120.00},
    {"name": "EV1000 Bag",                "unit": "ea", "cost_per_unit": 45.00},
    {"name": "MS-450 Manifold",           "unit": "ea", "cost_per_unit": 85.00},
    {"name": "1000mL Vacuum Filter Unit", "unit": "ea", "cost_per_unit": 12.00},
    {"name": "Lovo Kit",                  "unit": "ea", "cost_per_unit": 450.00},
    {"name": "Lovo Ancillary Bag",        "unit": "ea", "cost_per_unit": 80.00},
    {"name": "Manifold",                  "unit": "ea", "cost_per_unit": 50.00},
])

WASTE_REASONS = [
    "Damage during handling", "Contamination", "Expired",
    "Process loss", "Calibration/test loss", "Operator error", "Other"
]

# ✅ Updated shifts
SHIFTS = [
    "MFG-A Shift Day", "MFG-A Shift Night",
    "MFG-B Shift Day", "MFG-B Shift Night",
    "MSAT", "QC", "QO"
]

# ------------------------- Helpers -------------------------
def sanitize_filename(text: str) -> str:
    text = text.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-\.]", "", text)

def _ensure_cols(df: pd.DataFrame, cols):
    for c, default in cols.items():
        if c not in df.columns:
            df[c] = default
    return df

def load_materials() -> pd.DataFrame:
    if MATERIALS_CSV.exists():
        df = pd.read_csv(MATERIALS_CSV)
    else:
        df = DEFAULT_MATERIALS.copy()
        df.to_csv(MATERIALS_CSV, index=False)
    df = _ensure_cols(df, {"name":"", "unit":"ea", "cost_per_unit":0.0})
    df["cost_per_unit"] = pd.to_numeric(df["cost_per_unit"], errors="coerce").fillna(0.0)
    return df

def save_materials(df: pd.DataFrame):
    df = df[["name","unit","cost_per_unit"]].copy()
    df.to_csv(MATERIALS_CSV, index=False)

def load_mock_prices(materials: pd.DataFrame) -> pd.DataFrame:
    if MOCK_PRICE_CSV.exists():
        mp = pd.read_csv(MOCK_PRICE_CSV)
    else:
        mp = materials[["name","cost_per_unit"]].rename(columns={"cost_per_unit":"mock_price"}).copy()
        mp.to_csv(MOCK_PRICE_CSV, index=False)
    mp = _ensure_cols(mp, {"name":"", "mock_price":0.0})
    mp["mock_price"] = pd.to_numeric(mp["mock_price"], errors="coerce").fillna(0.0)
    return mp

def save_mock_prices(df: pd.DataFrame):
    df = df[["name","mock_price"]].copy()
    df.to_csv(MOCK_PRICE_CSV, index=False)

def load_sop() -> pd.DataFrame:
    if SOP_CSV.exists():
        df = pd.read_csv(SOP_CSV)
    else:
        df = pd.DataFrame({"reason": WASTE_REASONS, "sop_url": [""]*len(WASTE_REASONS)})
        df.to_csv(SOP_CSV, index=False)
    df = _ensure_cols(df, {"reason":"", "sop_url":""})
    df["sop_url"] = df["sop_url"].fillna("")
    return df

def save_sop(df: pd.DataFrame):
    df = df[["reason","sop_url"]].copy()
    df.to_csv(SOP_CSV, index=False)

def load_master() -> pd.DataFrame:
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
        for c in ["quantity","cost_per_unit","total_cost"]:
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
        if "timestamp_local" in df:
            df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
        return df
    return pd.DataFrame()

def append_to_master(row: dict):
    header = not MASTER_CSV.exists()
    pd.DataFrame([row]).to_csv(MASTER_CSV, mode="a", header=header, index=False)

def save_individual_log(row: dict, uid: str) -> Path:
    when = datetime.fromisoformat(row["timestamp_local"])
    ts = when.strftime("%Y%m%d_%H%M%S")
    mat = sanitize_filename(row["material"])
    fn = DATA_DIR / f"waste_{ts}_{mat}_{uid}.csv"
    pd.DataFrame([row]).to_csv(fn, index=False)
    return fn

def compute_hash(record: dict) -> str:
    # Keep legacy keys for schema compatibility (site/line/room blank)
    fields = [
        "id","timestamp_local","date","time","site","suite","line","room","shift",
        "employee_id","material","unit","quantity","reason","batch_lot",
        "cost_per_unit","total_cost","notes","photo_filename","sop_url","prev_hash"
    ]
    payload = "|".join(str(record.get(k,"")) for k in fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def ensure_nonneg_int(x) -> int:
    try:
        return max(int(x), 0)
    except:
        return 0

# Download helper (uses string path to avoid _repr_html_ issues)
def show_saved_file(path_str: str, key: str):
    st.success(f"Saved: {path_str}")
    p = Path(path_str)
    try:
        with open(p, "rb") as fh:
            st.download_button("⬇️ Download export", data=fh, file_name=p.name,
                               mime="text/csv", key=key)
    except Exception:
        pass

# ------------------------- Sidebar: Role & Admin -------------------------
st.sidebar.title("👤 Role & Admin")
role = st.sidebar.selectbox("Select your role", ["Operator", "Supervisor", "Manager"])

st.sidebar.markdown("---")
st.sidebar.subheader("📘 SOP Links (Reason → URL)")
sop_df = load_sop()
edit_reason = st.sidebar.selectbox("Reason", sop_df["reason"].tolist(), key="sop_reason")
new_url = st.sidebar.text_input("SOP URL", value=sop_df.loc[sop_df["reason"]==edit_reason, "sop_url"].iloc[0] if not sop_df.empty else "")
if st.sidebar.button("💾 Save SOP link"):
    sop_df.loc[sop_df["reason"]==edit_reason, "sop_url"] = new_url.strip()
    save_sop(sop_df)
    st.sidebar.success("SOP link saved.")

# Materials & Mock Prices
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Materials")
materials = load_materials()
with st.sidebar.expander("Add new material"):
    mn = st.text_input("Name", key="new_mat_name")
    mu = st.text_input("Unit (kg/L/ea...)", value="ea", key="new_mat_unit")
    mc = st.number_input("Mock/catalog cost per unit", min_value=0.0, value=0.0, step=0.01, key="new_mat_cpu")
    if st.button("➕ Add material"):
        if mn.strip():
            new_row = {"name": mn.strip(), "unit": mu.strip() or "ea", "cost_per_unit": float(mc)}
            materials = pd.concat([materials, pd.DataFrame([new_row])], ignore_index=True)
            save_materials(materials)
            st.success(f"Added '{mn.strip()}'")

with st.sidebar.expander("Edit existing material"):
    if not materials.empty:
        selm = st.selectbox("Select", materials["name"].tolist(), key="edit_sel")
        row = materials[materials["name"]==selm].iloc[0]
        e_name = st.text_input("Name", value=row["name"], key="e_name")
        e_unit = st.text_input("Unit", value=row["unit"], key="e_unit")
        e_cost = st.number_input("Cost per unit", min_value=0.0, value=float(row["cost_per_unit"]), step=0.01, key="e_cost")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save"):
                idx = materials[materials["name"]==selm].index[0]
                materials.loc[idx, "name"] = e_name.strip()
                materials.loc[idx, "unit"] = e_unit.strip()
                materials.loc[idx, "cost_per_unit"] = float(e_cost)
                save_materials(materials)
                st.success("Material updated.")
        with c2:
            if st.button("🗑️ Delete"):
                materials = materials[materials["name"]!=selm].reset_index(drop=True)
                save_materials(materials)
                st.warning(f"Deleted '{selm}'.")

# Mock price override
st.sidebar.markdown("---")
st.sidebar.subheader("💲 Mock Pricing (Overrides)")
mock_prices = load_mock_prices(materials)
use_mock = st.sidebar.checkbox("Use mock price overrides (recommended for now)", value=True)
with st.sidebar.expander("Edit mock prices"):
    mp_sel = st.selectbox("Material", materials["name"].tolist(), key="mp_sel")
    cur_mp = mock_prices.loc[mock_prices["name"]==mp_sel, "mock_price"]
    mp_val = float(cur_mp.iloc[0]) if not cur_mp.empty else 0.0
    new_mp = st.number_input("Mock price", min_value=0.0, value=mp_val, step=0.01, key="mp_val")
    if st.button("💾 Save mock price"):
        if (mock_prices["name"] == mp_sel).any():
            mock_prices.loc[mock_prices["name"] == mp_sel, "mock_price"] = new_mp
        else:
            mock_prices = pd.concat([mock_prices, pd.DataFrame([{"name": mp_sel, "mock_price": new_mp}])], ignore_index=True)
        save_mock_prices(mock_prices)
        st.success("Mock price saved.")

# Alerts
st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Alerts")
per_entry_cost_threshold = st.sidebar.number_input("High-cost alert per entry ($)", min_value=0.0, value=500.0, step=10.0)
daily_cost_threshold = st.sidebar.number_input("Daily total alert ($)", min_value=0.0, value=2000.0, step=50.0)

# Email config
st.sidebar.markdown("---")
st.sidebar.subheader("📧 Monthly Email Report (SMTP)")
manager_email = st.sidebar.text_input("Manager email", value="")
smtp_host = st.sidebar.text_input("SMTP host", value="")
smtp_port = st.sidebar.number_input("SMTP port", min_value=1, value=587, step=1)
smtp_user = st.sidebar.text_input("SMTP username", value="")
smtp_pass = st.sidebar.text_input("SMTP password", type="password", value="")
auto_send_monthly = st.sidebar.checkbox("Auto-send on 1st of month (while app is running)", value=False)

def send_email_with_attachment(to_addr: str, subject: str, body: str, attach_path: Path):
    if not (smtp_host and smtp_port and smtp_user and smtp_pass and to_addr):
        st.error("SMTP settings incomplete — cannot send email.")
        return False
    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    data = attach_path.read_bytes()
    msg.add_attachment(data, maintype="text", subtype="csv", filename=attach_path.name)
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email send failed: {e}")
        return False

# ------------------------- Main UI -------------------------
st.title("♻️ Material Waste Tracker — Scan-Free")

master_df = load_master()

# Filters (no Sites/Lines)
with st.expander("🔎 Filters", expanded=True):
    colf = st.columns(5)
    start_default = master_df["timestamp_local"].min().date() if not master_df.empty and master_df["timestamp_local"].notna().any() else date.today()
    end_default = master_df["timestamp_local"].max().date() if not master_df.empty and master_df["timestamp_local"].notna().any() else date.today()
    with colf[0]:
        start_date = st.date_input("Start date", value=start_default)
    with colf[1]:
        end_date = st.date_input("End date", value=end_default)
    with colf[2]:
        filt_materials = st.multiselect("Materials", options=materials["name"].tolist())
    with colf[3]:
        filt_reasons = st.multiselect("Reasons", options=WASTE_REASONS)
    with colf[4]:
        filt_suites = st.multiselect("Suites", options=ALL_SUITES)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df = df[df["timestamp_local"].notna()]
    df["date_only"] = df["timestamp_local"].dt.date
    mask = (df["date_only"] >= start_date) & (df["date_only"] <= end_date)
    if filt_materials: mask &= df["material"].isin(filt_materials)
    if filt_reasons: mask &= df["reason"].isin(filt_reasons)
    if filt_suites and "suite" in df: mask &= df["suite"].isin(filt_suites)
    return df.loc[mask].copy()

filtered = apply_filters(master_df)

# Tabs (AI tab inserted)
tab_log, tab_trends, tab_ai, tab_materials, tab_reasons, tab_cost, tab_suites, tab_compliance, tab_exports = st.tabs(
    ["📝 Log", "📈 Trends", "🤖 AI (Starter)", "📦 Materials", "🧭 Reasons", "💲 Cost (Managers)", "🏢 Suites", "🔒 Compliance", "📤 Exports"]
)

# LOG
with tab_log:
    st.subheader("Submit a Waste Log")
    with st.form("waste_form", clear_on_submit=True):
        c = st.columns(4)
        with c[0]:
            log_date = st.date_input("Date", value=date.today())
            log_time = st.time_input("Time", value=datetime.now().time().replace(microsecond=0))
            shift = st.selectbox("Shift", options=SHIFTS, index=0)
        with c[1]:
            suite = st.selectbox("Suite", options=ALL_SUITES)
            mat_name = st.selectbox("Material", options=materials["name"].tolist())
        with c[2]:
            qty = st.number_input("Quantity", min_value=0, value=0, step=1, format="%d")  # integer only
            reason = st.selectbox("Reason", options=WASTE_REASONS, index=0)
        with c[3]:
            batch = st.text_input("Batch/Lot (optional)", placeholder="e.g., B-2025-09-02-A")
            notes = st.text_area("Notes (optional)")
            photo = st.file_uploader("Photo evidence (JPG/PNG)", type=["png","jpg","jpeg"])

        sop_url_cur = sop_df.loc[sop_df["reason"]==reason, "sop_url"].iloc[0] if reason in sop_df["reason"].values else ""
        if sop_url_cur:
            st.markdown(f"📘 **SOP for {reason}:** [{sop_url_cur}]({sop_url_cur})")

        operator_id = st.text_input("Operator Initial/date", placeholder="e.g., DB/2025-09-02")

        submitted = st.form_submit_button("✅ Save waste log")

    if submitted:
        sel = materials[materials["name"] == mat_name].iloc[0]
        unit = sel["unit"]

        price_lookup = load_mock_prices(materials).set_index("name")["mock_price"].to_dict()
        cost_per_unit = price_lookup.get(mat_name, float(sel["cost_per_unit"])) if use_mock else float(sel["cost_per_unit"])

        dt_local = datetime.combine(log_date, log_time)
        uid = uuid4().hex[:8]

        record = {
            "id": uid,
            "timestamp_local": dt_local.isoformat(),
            "date": log_date.isoformat(),
            "time": log_time.isoformat(),
            "site": "",        # kept for schema compatibility
            "suite": suite.strip(),
            "line": "",
            "room": "",
            "shift": shift,
            "employee_id": operator_id.strip(),
            "material": mat_name,
            "unit": unit,
            "quantity": int(qty),
            "reason": reason,
            "batch_lot": batch.strip(),
            "cost_per_unit": cost_per_unit,
            "total_cost": round(int(qty) * cost_per_unit, 4),
            "notes": notes.strip(),
            "photo_filename": "",
            "sop_url": sop_url_cur,
        }

        if photo is not None:
            ext = Path(photo.name).suffix.lower()
            img_name = f"evidence_{sanitize_filename(mat_name)}_{uid}{ext}"
            img_path = IMAGES_DIR / img_name
            with open(img_path, "wb") as f:
                f.write(photo.getbuffer())
            record["photo_filename"] = str(img_path)

        prev_hash = ""
        if MASTER_CSV.exists():
            try:
                prev_df = pd.read_csv(MASTER_CSV, usecols=lambda c: c in ["hash"])
                if "hash" in prev_df and not prev_df.empty:
                    prev_hash = str(prev_df["hash"].iloc[-1])
            except Exception:
                prev_hash = ""
        record["prev_hash"] = prev_hash
        record["hash"] = compute_hash(record)

        entry_path = save_individual_log(record, uid)
        append_to_master(record)

        if record["total_cost"] >= per_entry_cost_threshold > 0:
            st.warning(f"🚨 High-cost entry: ${record['total_cost']:,.2f} ≥ ${per_entry_cost_threshold:,.2f}")
        df_today = load_master()
        if not df_today.empty and "timestamp_local" in df_today:
            dft = df_today[df_today["timestamp_local"].dt.date == date.today()]
            today_cost = pd.to_numeric(dft["total_cost"], errors="coerce").fillna(0).sum()
            if today_cost >= daily_cost_threshold > 0:
                st.error(f"🚨 Daily cost threshold exceeded: ${today_cost:,.2f} ≥ ${daily_cost_threshold:,.2f}")

        st.success(f"Saved waste log → {entry_path.name}")
        st.toast("Entry saved ✔", icon="✅")

# TRENDS
with tab_trends:
    st.subheader("Trends Over Time")
    if filtered.empty:
        st.info("No data to display. Add logs or widen filters.")
    else:
        daily = filtered.copy()
        daily["date"] = pd.to_datetime(daily["timestamp_local"]).dt.date
        daily_sum = daily.groupby("date", as_index=False)[["quantity","total_cost"]].sum()

        c1, c2 = st.columns(2)
        qty_line = alt.Chart(daily_sum).mark_line(point=True).encode(
            x="date:T", y=alt.Y("quantity:Q", title="Quantity"), tooltip=["date:T","quantity:Q"]
        ).properties(height=250)
        c1.altair_chart(qty_line, use_container_width=True)

        if role == "Manager":
            cost_line = alt.Chart(daily_sum).mark_line(point=True).encode(
                x="date:T", y=alt.Y("total_cost:Q", title="Total Cost ($)"), tooltip=["date:T","total_cost:Q"]
            ).properties(height=250)
            c2.altair_chart(cost_line, use_container_width=True)
        else:
            c2.info("Cost trend visible to Managers only.")

        st.markdown("### Quantity by Shift")
        by_shift = filtered.groupby("shift", as_index=False)["quantity"].sum().sort_values("quantity", ascending=False)
        shift_bar = alt.Chart(by_shift).mark_bar().encode(
            x=alt.X("shift:N", title="Shift"),
            y=alt.Y("quantity:Q", title="Quantity"),
            tooltip=["shift","quantity"]
        ).properties(height=260)
        st.altair_chart(shift_bar, use_container_width=True)

# AI (Starter)
with tab_ai:
    st.subheader("🤖 AI (Starter)")
    if filtered.empty:
        st.info("No data within current filters. Add logs or widen the date range.")
    else:
        # --- Build canonical AI frame from your schema ---
        df_ai = filtered.copy()
        df_ai["date"] = pd.to_datetime(df_ai["timestamp_local"], errors="coerce")
        df_ai["qty"] = pd.to_numeric(df_ai["quantity"], errors="coerce").fillna(0)
        df_ai["cost"] = pd.to_numeric(df_ai["total_cost"], errors="coerce").fillna(0)
        # No explicit vendor field available; use material as a proxy for duplicate/subscription heuristics
        df_ai["vendor"] = df_ai["material"].astype(str)
        df_ai["description"] = df_ai["reason"].fillna("")
        base_cols = ["date", "suite", "material", "vendor", "qty", "cost", "description"]
        df_ai = df_ai[base_cols].dropna(subset=["date"]).reset_index(drop=True)

        # 1) Anomaly / waste detection
        st.markdown("### AI-Powered Waste Detection")
        df_ano, _meta = flag_anomalies(df_ai)
        st.dataframe(
            df_ano[["date","suite","material","qty","cost","ai_anomaly","ai_score","ai_waste_flag"]],
            use_container_width=True
        )
        st.download_button("⬇️ Download with AI flags (CSV)", df_ano.to_csv(index=False), "ai_flags.csv")

        # 2) Duplicate & subscription-like patterns
        st.markdown("### Duplicate Expenses & Subscription-like Patterns")
        dups = detect_duplicates_and_subs(df_ano)
        st.dataframe(dups, use_container_width=True)

        # merge subscription flag back (for reports)
        merge_cols = ["date","vendor","cost"]
        dups_merge = dups[merge_cols + ["subscription_like"]].copy()
        df_for_reports = df_ano.merge(dups_merge, on=merge_cols, how="left")
        df_for_reports["subscription_like"] = df_for_reports["subscription_like"].fillna(False)

        # 3) Basic categorization
        st.markdown("### AI Category Analysis")
        cat_df, _ = categorize_basic(df_ano)
        st.dataframe(
            cat_df[["date","vendor","description","cost","ai_category","ai_category_source"]],
            use_container_width=True
        )

        # 4) Trends & simple forecast
        st.markdown("### Spending Trends & Basic Forecast")
        monthly = aggregate_timeseries(df_ano)
        st.line_chart(monthly[["cost","cost_ma3"]])
        fcst = naive_forecast(monthly, periods=3)
        st.write("**Next 3 months forecast (naive moving-average):**")
        st.dataframe(fcst, use_container_width=True)

        # 5) Reports: Weekly / Monthly / Quarterly / Annual
        st.markdown("### AI Reports & Insights (Weekly • Monthly • Quarterly • Annual)")
        rep_input = df_for_reports[["date","cost","ai_anomaly","subscription_like","material","vendor"]]
        all_reports = generate_all_reports(rep_input)

        for period_name in ["Weekly", "Monthly", "Quarterly", "Annual"]:
            st.markdown(f"#### {period_name} Report")
            tbl = all_reports[period_name]["table"]
            txt = all_reports[period_name]["text"]

            if not tbl.empty:
                st.dataframe(tbl.tail(8), use_container_width=True)
                st.text_area(f"{period_name} Summary", txt, height=180)

                st.download_button(
                    f"⬇️ Download {period_name} Table (CSV)",
                    tbl.to_csv().encode("utf-8"),
                    file_name=f"{period_name.lower()}_report.csv"
                )
                st.download_button(
                    f"⬇️ Download {period_name} Summary (TXT)",
                    txt.encode("utf-8"),
                    file_name=f"{period_name.lower()}_summary.txt"
                )
            else:
                st.info(f"No data available yet for {period_name}.")

# MATERIALS
with tab_materials:
    st.subheader("Waste by Material")
    if filtered.empty:
        st.info("No data.")
    else:
        by_mat = filtered.groupby("material", as_index=False)[["quantity","total_cost"]].sum().sort_values("quantity", ascending=False)
        chart = alt.Chart(by_mat).mark_bar().encode(
            x=alt.X("material:N", sort='-y', title="Material"),
            y=alt.Y("quantity:Q", title="Quantity"),
            tooltip=["material","quantity","total_cost"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(filtered.drop(columns=["date_only"]), use_container_width=True)

# REASONS
with tab_reasons:
    st.subheader("Waste by Reason")
    if filtered.empty:
        st.info("No data.")
    else:
        by_reason = filtered.groupby("reason", as_index=False)[["quantity","total_cost"]].sum().sort_values("quantity", ascending=False)
        chart = alt.Chart(by_reason).mark_bar().encode(
            x=alt.X("reason:N", sort='-y', title="Reason"),
            y=alt.Y("quantity:Q", title="Quantity"),
            tooltip=["reason","quantity","total_cost"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        st.markdown("#### SOP Coverage")
        sop_merge = by_reason.merge(sop_df, on="reason", how="left")
        st.dataframe(sop_merge.rename(columns={"sop_url":"SOP URL"}), use_container_width=True)

# COST (Managers)
with tab_cost:
    st.subheader("Cost KPIs & Charts (Managers Only)")
    if role != "Manager":
        st.warning("Only Managers can view cost KPIs and charts.")
    else:
        if filtered.empty:
            st.info("No data.")
        else:
            m = st.columns(4)
            m[0].metric("Entries", f"{len(filtered):,}")
            m[1].metric("Total Qty", f"{filtered['quantity'].sum():,.0f}")
            m[2].metric("Avg Qty/Entry", f"{filtered['quantity'].mean():.2f}")
            m[3].metric("Total Cost", f"${filtered['total_cost'].sum():,.2f}")

            by_suite = filtered.groupby("suite", as_index=False)["total_cost"].sum().sort_values("total_cost", ascending=False)
            chart_suite = alt.Chart(by_suite).mark_bar().encode(
                x=alt.X("suite:N", sort='-y', title="Suite"),
                y=alt.Y("total_cost:Q", title="Total Cost ($)"),
                tooltip=["suite","total_cost"]
            ).properties(height=320)
            st.altair_chart(chart_suite, use_container_width=True)

# SUITES
with tab_suites:
    st.subheader("Per-Suite Tracking")
    if filtered.empty:
        st.info("No data.")
    else:
        top_n = st.slider("Top materials to show in grids", min_value=3, max_value=20, value=8, step=1)

        # Overview by suite (quantity)
        by_suite_qty = filtered.groupby("suite", as_index=False)["quantity"].sum().sort_values("quantity", ascending=False)
        chart_qty = alt.Chart(by_suite_qty).mark_bar().encode(
            x=alt.X("suite:N", sort='-y', title="Suite"),
            y=alt.Y("quantity:Q", title="Quantity"),
            tooltip=["suite","quantity"]
        ).properties(height=320)
        st.altair_chart(chart_qty, use_container_width=True)

        # 1) Facet: material breakdown per suite
        suite_mat = filtered.copy()
        suite_mat["quantity"] = pd.to_numeric(suite_mat["quantity"], errors="coerce").fillna(0)
        agg_sm = suite_mat.groupby(["suite","material"], as_index=False)["quantity"].sum()
        agg_sm["rank"] = agg_sm.groupby("suite")["quantity"].rank(method="first", ascending=False)
        agg_sm_top = agg_sm[agg_sm["rank"] <= top_n]
        base = alt.Chart(agg_sm_top).mark_bar().encode(
            x=alt.X("material:N", sort='-y', title="Material"),
            y=alt.Y("quantity:Q", title="Qty"),
            tooltip=["suite","material","quantity"]
        ).properties(height=220)
        faceted_suite = base.facet(
            column=alt.Column("suite:N", title="Suite"),
            columns=3
        ).resolve_scale(y="independent")
        st.markdown("#### Material breakdown per suite")
        st.altair_chart(faceted_suite, use_container_width=True)

        # 2) Facet: per-material charts (bars by suite)
        mat_suite = filtered.copy()
        mat_suite["quantity"] = pd.to_numeric(mat_suite["quantity"], errors="coerce").fillna(0)
        agg_ms = mat_suite.groupby(["material","suite"], as_index=False)["quantity"].sum()
        top_mats = (agg_ms.groupby("material")["quantity"].sum()
                    .sort_values(ascending=False).head(top_n).index.tolist())
        agg_ms_top = agg_ms[agg_ms["material"].isin(top_mats)]
        base2 = alt.Chart(agg_ms_top).mark_bar().encode(
            x=alt.X("suite:N", title="Suite"),
            y=alt.Y("quantity:Q", title="Qty"),
            color="suite:N",
            tooltip=["material","suite","quantity"]
        ).properties(height=220)
        faceted_mat = base2.facet(
            column=alt.Column("material:N", title="Material", sort=top_mats),
            columns=3
        ).resolve_scale(y="independent")
        st.markdown("#### Per-material charts (suite comparison)")
        st.altair_chart(faceted_mat, use_container_width=True)

        # 3) Overall (all suites combined) — line chart
        metric_label = "Quantity"
        if role == "Manager":
            metric_label = st.radio("Overall line chart metric", ["Quantity", "Total Cost ($)"], horizontal=True, index=0)
        if metric_label.startswith("Total"):
            tot = (filtered.groupby("material", as_index=False)["total_cost"]
                   .sum().rename(columns={"total_cost":"value"})
                   .sort_values("value", ascending=False))
            y_enc = alt.Y("value:Q", title="Total Cost ($)")
        else:
            tot = (filtered.groupby("material", as_index=False)["quantity"]
                   .sum().rename(columns={"quantity":"value"})
                   .sort_values("value", ascending=False))
            y_enc = alt.Y("value:Q", title="Total Quantity")

        st.markdown("#### Overall waste per material (all suites combined)")
        line = alt.Chart(tot).mark_line(point=True).encode(
            x=alt.X("material:N", sort=tot["material"].tolist(), title="Material"),
            y=y_enc,
            tooltip=["material","value"]
        ).properties(height=260)
        st.altair_chart(line, use_container_width=True)

# COMPLIANCE
with tab_compliance:
    st.subheader("Audit Trail & Verification")
    if master_df.empty:
        st.info("No logs yet.")
    else:
        st.write("The master log uses a chained SHA-256 hash for tamper-evidence.")
        if st.button("🔍 Verify hash chain"):
            dfc = load_master()
            ok = True
            for _, r in dfc.iterrows():
                rec = r.to_dict()
                expected = compute_hash(rec)
                if str(expected) != str(rec.get("hash","")):
                    ok = False
                    break
            if ok:
                st.success("Hash chain verified ✅ No integrity issues detected.")
            else:
                st.error("Hash chain verification FAILED ❌. The log may have been altered.")

        st.download_button(
            "⬇️ Download full master log (CSV)",
            data=master_df.to_csv(index=False),
            file_name="waste_master.csv",
            mime="text/csv",
        )

# EXPORTS
with tab_exports:
    st.subheader("On-Demand Exports (Daily / Weekly / Monthly / Quarterly / Annual)")
    st.caption("Files are saved to waste_logs/exports/ and include all fields.")

    def export_range(df, start_d: date, end_d: date, label: str) -> str | None:
        if df.empty:
            st.warning("No data to export.")
            return None
        dfe = df.copy()
        dfe["d"] = pd.to_datetime(dfe["timestamp_local"]).dt.date
        dfe = dfe[(dfe["d"]>=start_d) & (dfe["d"]<=end_d)].drop(columns=["d"])
        if dfe.empty:
            st.warning(f"No entries in {label}.")
            return None
        fname = EXPORTS_DIR / f"waste_export_{label.replace(' ','_')}.csv"
        dfe.to_csv(fname, index=False)
        return str(fname)

    today = date.today()
    day_start = today; day_end = today
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    month_start = today.replace(day=1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)
    month_end = next_month - timedelta(days=1)
    q = (today.month - 1)//3 + 1
    q_start_month = 3*(q-1) + 1
    quarter_start = date(today.year, q_start_month, 1)
    next_q_month = q_start_month + 3
    quarter_end = date(today.year, next_q_month, 1) - timedelta(days=1) if next_q_month <= 12 else date(today.year, 12, 31)
    year_start = date(today.year, 1, 1)
    year_end = date(today.year, 12, 31)

    cols = st.columns(5)
    with cols[0]:
        if st.button("📅 Export Today"):
            f = export_range(load_master(), day_start, day_end, f"day_{day_start}")
            if f: show_saved_file(f, key="exp_today")
    with cols[1]:
        if st.button("🗓️ Export This Week"):
            f = export_range(load_master(), week_start, week_end, f"week_{week_start}_{week_end}")
            if f: show_saved_file(f, key="exp_week")
    with cols[2]:
        if st.button("🗓️ Export This Month"):
            f = export_range(load_master(), month_start, month_end, f"month_{month_start}_{month_end}")
            if f: show_saved_file(f, key="exp_month")
    with cols[3]:
        if st.button("📈 Export This Quarter"):
            f = export_range(load_master(), quarter_start, quarter_end, f"quarter_{quarter_start}_{quarter_end}")
            if f: show_saved_file(f, key="exp_quarter")
    with cols[4]:
        if st.button("📦 Export This Year"):
            f = export_range(load_master(), year_start, year_end, f"year_{year_start}_{year_end}")
            if f: show_saved_file(f, key="exp_year")

    st.markdown("**Custom range**")
    c1, c2 = st.columns(2)
    with c1:
        cs = st.date_input("Start", value=today, key="custom_start")
    with c2:
        ce = st.date_input("End", value=today, key="custom_end")
    if st.button("📁 Export Custom Range"):
        if cs > ce:
            st.error("Start date must be ≤ End date.")
        else:
            f = export_range(load_master(), cs, ce, f"custom_{cs}_{ce}")
            if f: show_saved_file(f, key="exp_custom")

    st.markdown("---")
    st.subheader("📧 Monthly Manager Report")
    st.caption("Exports the current month and emails it to the manager. Requires SMTP settings.")
    if st.button("📤 Send This Month Now"):
        f = export_range(load_master(), month_start, month_end, f"month_{month_start}_{month_end}")
        if f:
            ok = send_email_with_attachment(
                to_addr=manager_email,
                subject=f"[Waste Report] {month_start} to {month_end}",
                body=f"Attached is the monthly waste report for {month_start} to {month_end}.",
                attach_path=Path(f)
            )
            if ok:
                LAST_EMAIL_FLAG.write_text(f"{today.year}-{today.month}")
                st.success("Monthly report emailed to manager ✅")

# Auto-send monthly (while app is running)
if auto_send_monthly and manager_email and smtp_host and smtp_user and smtp_pass:
    today = date.today()
    if today.day == 1:
        last_sent = LAST_EMAIL_FLAG.read_text().strip() if LAST_EMAIL_FLAG.exists() else ""
        tag = f"{today.year}-{today.month}"
        if last_sent != tag:
            month_start = today.replace(day=1)
            next_month = (month_start + timedelta(days=32)).replace(day=1)
            month_end = next_month - timedelta(days=1)
            df = load_master()
            if not df.empty:
                df["d"] = pd.to_datetime(df["timestamp_local"]).dt.date
                dfe = df[(df["d"]>=month_start) & (df["d"]<=month_end)].drop(columns=["d"])
                if not dfe.empty:
                    p = EXPORTS_DIR / f"waste_export_month_{month_start}_{month_end}.csv"
                    dfe.to_csv(p, index=False)
                    if send_email_with_attachment(
                        to_addr=manager_email,
                        subject=f"[Waste Report] {month_start} to {month_end}",
                        body=f"Attached is the monthly waste report for {month_start} to {month_end}.",
                        attach_path=p
                    ):
                        LAST_EMAIL_FLAG.write_text(tag)
