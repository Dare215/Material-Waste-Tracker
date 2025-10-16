import pandas as pd
from sklearn.ensemble import IsolationForest

def flag_anomalies(df: pd.DataFrame, contamination: float = 0.03, random_state: int = 42):
    use = df.copy()
    use["qty_f"]  = use["qty"].fillna(0).clip(lower=0)
    use["cost_f"] = use["cost"].fillna(0).clip(lower=0)
    X = use[["qty_f","cost_f"]].values
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    labels = iso.fit_predict(X)
    scores = iso.decision_function(X)
    df["ai_anomaly"] = (labels == -1)
    df["ai_score"] = scores
    df["ai_waste_flag"] = df["ai_anomaly"] & (df["cost_f"] > df["cost_f"].median())
    return df, {"model":"IsolationForest","contamination":contamination}
