import pandas as pd
from sklearn.ensemble import IsolationForest

def flag_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Detect waste anomalies using IsolationForest.
    Adds ai_score, ai_anomaly (bool), and ai_waste_flag columns.
    Returns (dataframe_with_flags, metadata)
    """
    df = df.copy()
    if df.empty:
        return df, {"message": "No data to analyze"}

    # --- numeric columns ---
    num_cols = [c for c in ["qty", "cost"] if c in df.columns]
    if not num_cols:
        return df, {"message": "No numeric columns found"}

    # fill missing numeric
    df[num_cols] = df[num_cols].fillna(0.0)
    X = df[num_cols].to_numpy()

    # --- model ---
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    scores = model.decision_function(X)
    preds = model.predict(X)

    df["ai_score"] = scores
    df["ai_anomaly"] = preds == -1
    df["ai_waste_flag"] = df["ai_anomaly"].apply(lambda x: "🚨 Waste Risk" if x else "✅ Normal")

    metadata = {
        "n_records": len(df),
        "n_anomalies": int(df["ai_anomaly"].sum()),
        "contamination": 0.05,
    }
    return df, metadata
