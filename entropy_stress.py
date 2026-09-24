"""
engine for the website: data, entropy calculation and metrics
same method as the research notebooks 02 (mse) and 03 (mmse), one value per trading day

results are saved in data/ and only new days are calculated on each update
the full history is built with research/05_build_data.ipynb (or: python entropy_stress.py)
the website never rebuilds the full history itself
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---------- settings (same as the research notebooks) ----------

start="1991-01-01"
tau=5              # ma detrend scale (1 week)
window=1044        # 261 x 4 years
m=2                # embedding dimension
r_factor=0.15      # tolerance per channel, in units of std
step=1             # one value per trading day
min_history=261    # 1 year of stress before a baseline is shown
recompute_last=5   # always redo the last few days: yahoo can revise the latest price

indices={"snp":"^GSPC","dji":"^DJI","nas":"^IXIC","rus":"^RUT"}
names={"snp":"S&P 500","dji":"Dow Jones","nas":"NASDAQ","rus":"Russell 2000"}
data_dir=Path(__file__).parent/"data"


# ---------- data ----------

def clean_prices(prices):
    s=prices.sort_index()
    s=s[~s.index.duplicated(keep="last")]
    s=s.dropna()
    return s[s>0]


def load_prices(ticker):
    import yfinance as yf
    data=yf.Ticker(ticker).history(start=start,auto_adjust=True)
    prices=data["Close"]
    prices.index=prices.index.tz_localize(None).normalize()
    return clean_prices(prices)


def load_all():
    return {key:load_prices(t).rename(key) for key,t in indices.items()}


# ---------- preprocessing ----------

def standardize(x):
    return (x-x.mean())/x.std()


def ma_detrend(x,tau=tau):
    # trailing moving average: the trend on day t uses days t-tau+1 ... t only,
    # so no future data is needed and the latest day can be calculated
    trend=x.rolling(tau).mean()
    return (x-trend).dropna()


# ---------- entropy ----------

def count_pairs(emb,r):
    # pairs (i < j) with chebyshev distance <= r
    tree=cKDTree(emb)
    return (tree.count_neighbors(tree,r,p=np.inf)-len(emb))/2


def sample_entropy(x,m,r):
    n_templates=len(x)-m
    b=count_pairs(np.lib.stride_tricks.sliding_window_view(x,m)[:n_templates],r)
    a=count_pairs(np.lib.stride_tricks.sliding_window_view(x,m+1)[:n_templates],r)
    return -np.log(a/b) if a>0 and b>0 else np.nan


def composite_vectors(data,dims,n_templates):
    parts=[np.lib.stride_tricks.sliding_window_view(data[:,k],d)[:n_templates] for k,d in enumerate(dims)]
    return np.hstack(parts)


def multivariate_sample_entropy(data,m,r):
    n,p=data.shape
    n_templates=n-m
    b=count_pairs(composite_vectors(data,[m]*p,n_templates),r)
    a=0.0
    for k in range(p):
        dims=[m]*p
        dims[k]=m+1
        a+=count_pairs(composite_vectors(data,dims,n_templates),r)
    a=a/p
    return -np.log(a/b) if a>0 and b>0 else np.nan


def mse_window(seg):
    return sample_entropy(seg/seg.std(),m,r_factor)


def mmse_window(seg):
    seg=seg/seg.std(axis=0)
    return multivariate_sample_entropy(seg,m,r_factor*seg.shape[1])


# ---------- saved results and incremental update ----------

def current_settings():
    return {"tau":tau,"window":window,"m":m,"r_factor":r_factor,"step":step,"detrend":"trailing"}


def settings_match():
    # saved results are only reused if they were made with the same settings
    path=data_dir/"settings.json"
    if not path.exists():
        return False
    return json.loads(path.read_text())==current_settings()


def load_saved(key):
    path=data_dir/f"{key}.csv"
    if not path.exists() or not settings_match():
        return None
    return pd.read_csv(path,index_col="date",parse_dates=True)["value"].rename(key)


def save(stress,key):
    try:
        data_dir.mkdir(exist_ok=True)
        stress.to_frame("value").to_csv(data_dir/f"{key}.csv",index_label="date")
        (data_dir/"settings.json").write_text(json.dumps(current_settings(),indent=1))
    except OSError:
        pass  # read-only disk: keep the result in memory only


class HistoryMissing(RuntimeError):
    pass


def update_series(key,y,entropy_fn,allow_full=True):
    # stress = 1 / entropy, calculating only the windows after the last saved date
    # allow_full=False (the website): refuse to rebuild the whole history, that is done in research/05
    saved=load_saved(key)
    if saved is None and not allow_full:
        raise HistoryMissing(f"no saved history for {key} with the current settings")
    after=None
    if saved is not None and len(saved)>recompute_last:
        after=saved.index[-recompute_last-1]
        saved=saved[saved.index<=after]
    else:
        saved=None
    ends=[e for e in range(window,len(y)+1,step) if after is None or y.index[e-1]>after]
    new={}
    for end in tqdm(ends,desc=key,disable=len(ends)<50):
        new[y.index[end-1]]=1/entropy_fn(y.values[end-window:end])
    new=pd.Series(new,dtype=float)
    stress=new if saved is None else pd.concat([saved,new])
    stress=stress[~stress.index.duplicated(keep="last")].sort_index().rename(key)
    save(stress,key)
    return stress


def update_all(prices,allow_full=True):
    # prices: dict key -> price series. returns dict of stress series
    stress={}
    for key,p in prices.items():
        y=ma_detrend(standardize(p))
        stress[f"mse_{key}"]=update_series(f"mse_{key}",y,mse_window,allow_full)
    frame=pd.concat(prices.values(),axis=1).dropna()
    y=frame.apply(standardize).apply(ma_detrend).dropna()
    stress["mmse_4idx"]=update_series("mmse_4idx",y,mmse_window,allow_full)
    return stress


# ---------- metrics (same as research notebook 04) ----------

def status(pct):
    return pd.cut(pct,bins=[-1,25,75,95,101],labels=["calm","normal","elevated","high"])


def change(s,days):
    past=s.reindex(s.index-pd.Timedelta(days=days),method="ffill")
    past.index=s.index
    return s-past


def daily_metrics(stress):
    s=stress.dropna()
    past=s.expanding(min_periods=min_history)
    out=pd.DataFrame({"stress":s,"normal":past.median()})
    out["vs_normal_pct"]=(s/out["normal"]-1)*100
    out["percentile"]=past.rank(pct=True)*100
    out["status"]=status(out["percentile"])
    for label,days in [("1d",1),("1w",7),("1m",30)]:
        out[f"change_{label}"]=change(s,days)
    out["avg_ytd"]=s.groupby(s.index.year).expanding().mean().droplevel(0)
    yearly=s.groupby(s.index.year).mean()
    out["avg_last_year"]=pd.Series(s.index.year-1,index=s.index).map(yearly)
    above=(s>out["normal"]).astype(int)
    out["streak_above_normal"]=above.groupby((above!=above.shift()).cumsum()).cumsum()
    return out


def status_runs(df):
    # consecutive days with the same status -> (start, end, status)
    st=df["status"].dropna()
    if st.empty:
        return []
    groups=(st!=st.shift()).cumsum()
    runs=[]
    for _,part in st.groupby(groups):
        nxt=st.index.searchsorted(part.index[-1])+1
        end=st.index[nxt] if nxt<len(st) else part.index[-1]
        runs.append((part.index[0],end,part.iloc[0]))
    return runs


def returns_summary(prices):
    p=prices.dropna()
    last=p.iloc[-1]
    d=p.index[-1]
    since=lambda days: (last/p.asof(d-pd.Timedelta(days=days))-1)*100
    prev=p[p.index.year<d.year]
    return {
        "date":d,
        "last_close":last,
        "r_1d":(last/p.iloc[-2]-1)*100,
        "r_1w":since(7),
        "r_1m":since(30),
        "r_ytd":(last/prev.iloc[-1]-1)*100 if len(prev) else np.nan,
        "r_1y":since(365),
        "drawdown":(last/p.max()-1)*100,
    }


if __name__=="__main__":
    prices=load_all()
    for key,p in prices.items():
        print(f"{key}: {len(p)} days, {p.index[0].date()} to {p.index[-1].date()}")
    stress=update_all(prices)
    for key,s in stress.items():
        last=daily_metrics(s).iloc[-1]
        print(f"{key}: {len(s)} values, last {s.index[-1].date()}, stress {last['stress']:.3f} ({last['status']})")
