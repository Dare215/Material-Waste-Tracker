import pandas as pd

# Canonical column mapping — ensures all datasets align to the same structure.
CANON = {
    "date": ["date", "event_ts", "timestamp"],
    "suite": ["suite", "area", "area_code"],
    "material": ["material", "item", "sku", "name"],
    "vendor": ["vendor", "merchant", "supplier", "payee"],
    "qty": ["qty", "quantity", "amount_units"],
    "cost": ["cost", "amount", "price", "value", "usd"],
    "category": ["category", "cat", "type"],
    "description": ["description", "memo", "note", "reason"],
    "lot": ["lot", "batch", "batch_id"]
}

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names across uploaded datasets.
    This helps the AI modules use consistent field names like
    'date', 'suite', 'material', 'vendor', 'qty', and 'cost'.
    """
    cols = {c.lower(): c for c in df.columns}
    out = {}

    # Map all expected columns from any variant found in the data.
    for tgt, alts in CANON.items():
        for a in alts:
            if a in cols or a in df.columns:
                out[tgt] = df[cols.get(a, a)]
                break
        # If the column is missing, create an empty placeholder.
        if tgt not in out:
            out[tgt] = pd.Series([None] * len(df))

    out = pd.DataFrame(out)

    # Type conversions
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce")
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")

    # Drop rows missing a valid date
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out
