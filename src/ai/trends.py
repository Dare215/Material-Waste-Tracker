import pandas as pd
from dateutil.relativedelta import relativedelta

def aggregate_timeseries(df: pd.DataFrame, freq="MS"):
    s = df.copy().set_index("date").sort_index()
    ts = s["cost"].resample(freq).sum().to_frame("cost")
    ts["cost_ma3"] = ts["cost"].rolling(3, min_periods=1).mean()
    return ts

def naive_forecast(ts: pd.DataFrame, periods=3, freq="MS"):
    last_date = ts.index.max()
    future_idx = [last_date + relativedelta(months=i) for i in range(1, periods+1)]
    base = ts["cost_ma3"].iloc[-1] if "cost_ma3" in ts else ts["cost"].iloc[-1]
    out = pd.DataFrame({"cost_fcst":[base]*periods},
                       index=pd.DatetimeIndex(future_idx, name=ts.index.name))
    return out
