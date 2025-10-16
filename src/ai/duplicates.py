import pandas as pd

def detect_duplicates_and_subs(df: pd.DataFrame, window_days: int = 90):
    d = df.copy()
    d["vendor_norm"] = d["vendor"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()
    d["amount_bucket"] = (d["cost"].round(2)).astype("float")

    # Duplicate expenses
    d = d.sort_values("date")
    d["dup_candidate"] = (
        (d["vendor_norm"].eq(d["vendor_norm"].shift())) &
        (d["amount_bucket"].eq(d["amount_bucket"].shift())) &
        ((d["date"] - d["date"].shift()).dt.days.abs() <= 3)
    )

    # Subscription-like patterns
    d["gap"] = (d["date"] - d.groupby(["vendor_norm","amount_bucket"])["date"].shift()).dt.days
    sub_mask = d.groupby(["vendor_norm","amount_bucket"])["gap"]\
                .transform(lambda s: (s.between(27,33)).rolling(3,min_periods=1).sum()>=1)
    d["subscription_like"] = sub_mask.fillna(False)

    return d[["date","vendor","cost","description","dup_candidate","subscription_like"]]
