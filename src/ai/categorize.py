import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RULES = {
    "utilities": ["electric","water","utility","sewer","waste mgmt"],
    "software": ["license","saas","subscription","cloud","azure","aws","gcp"],
    "supplies": ["gloves","tubes","pipette","label","sticker","cleanroom","wipe"],
    "logistics": ["shipping","freight","ups","fedex","dhl","courier"]
}

def rule_category(vendor: str, desc: str) -> str|None:
    text = f"{vendor} {desc}".lower()
    for cat, keys in RULES.items():
        if any(k in text for k in keys):
            return cat
    return None

def categorize_basic(df: pd.DataFrame, model=None, vectorizer=None):
    d = df.copy()
    d["rule_cat"] = [rule_category(v, x) for v,x in zip(d["vendor"].fillna(""), d["description"].fillna(""))]
    if model is None or vectorizer is None:
        d["ai_category"] = d["rule_cat"]
        d["ai_category_source"] = d["rule_cat"].notna().map({True:"rule", False:"unknown"})
        return d, None
    text = (d["vendor"].fillna("") + " " + d["description"].fillna("")).tolist()
    X = vectorizer.transform(text)
    d["ai_category"] = model.predict(X)
    d["ai_category"] = d["ai_category"].fillna(d["rule_cat"])
    d["ai_category_source"] = d["rule_cat"].notna().map({True:"rule+ml", False:"ml"})
    return d, {"model":"LogReg","features":"tfidf"}
