"""
step 1: financial stress via modified multiscale sample entropy (mod-mse)
based on xiao et al. (2026), financial innovation 12:30

install once:  pip install yfinance numpy pandas scipy matplotlib tqdm
run:           python entropy_stress.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

# settings from the paper
ticker="^GSPC"      # s&p 500. try "^RUT" (russell 2000), "^DJI", "^IXIC"
start="1991-01-01"
tau=5               # ma filter scale (1 week)
window=1044         # 261 points x 4 years
m=2                 # embedding dimension
r_factor=0.15       # tolerance = r_factor * std of window (not stated in paper, check this)
step=5              # compute every 5 days to keep it fast (1 = every day, like the paper)

crises=[
    ("1997-01-01","1999-12-31","economic boom","#dddddd"),
    ("2000-01-01","2003-12-31","internet bubble burst","#f4b6b6"),
    ("2008-01-01","2011-12-31","subprime crisis","#f4b6b6"),
    ("2020-01-01","2021-12-31","covid","#f4b6b6"),
]


def load_prices(ticker,start):
    import yfinance as yf
    data=yf.Ticker(ticker).history(start=start,auto_adjust=True)
    prices=data["Close"].dropna()
    prices.index=prices.index.tz_localize(None)
    return prices


def ma_detrend(x,tau):
    # eq (1)-(2): remove the local trend with a centred moving average
    x=pd.Series(x)
    trend=x.rolling(tau,center=True).mean()
    return (x-trend).dropna()


def count_matches(x,dim,n_templates,r):
    # number of template pairs (i != j) with chebyshev distance <= r
    emb=np.lib.stride_tricks.sliding_window_view(x,dim)[:n_templates]
    tree=cKDTree(emb)
    total=tree.count_neighbors(tree,r,p=np.inf)
    return (total-n_templates)/2  # remove self matches, count each pair once


def sample_entropy(x,m=2,r=0.2):
    x=np.asarray(x,dtype=float)
    n_templates=len(x)-m  # same number of templates for m and m+1
    b=count_matches(x,m,n_templates,r)
    a=count_matches(x,m+1,n_templates,r)
    if a==0 or b==0:
        return np.nan
    return -np.log(a/b)


def rolling_mod_mse(prices,tau,window,m,r_factor,step):
    # standardize, detrend, then sampen on a sliding 4-year window
    z=(prices-prices.mean())/prices.std()
    y=ma_detrend(z.values,tau)
    y.index=prices.index[y.index]
    dates=[]
    values=[]
    ends=range(window,len(y)+1,step)
    for end in tqdm(ends,desc="computing entropy",unit="window"):
        seg=y.values[end-window:end]
        r=r_factor*np.std(seg)
        values.append(sample_entropy(seg,m,r))
        dates.append(y.index[end-1])
    return pd.Series(values,index=dates)


def plot_stress(entropy,ticker):
    stress=1/entropy
    fig,ax=plt.subplots(figsize=(12,4))
    for s,e,label,color in crises:
        ax.axvspan(pd.Timestamp(s),pd.Timestamp(e),color=color,alpha=0.4,label=label)
    ax.plot(stress.index,stress.values,color="black",lw=1.2)
    ax.set_title(f"stress (1 / mod-mse) - {ticker}")
    ax.set_ylabel("1 / sample entropy  (higher = more stress)")
    ax.legend(loc="upper left",fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    name=f"stress_{ticker.replace('^','')}.png"
    fig.savefig(name,dpi=150)
    print(f"saved {name}")
    plt.show()


if __name__=="__main__":
    prices=load_prices(ticker,start)
    print(f"downloaded {len(prices)} days of {ticker}")
    entropy=rolling_mod_mse(prices,tau,window,m,r_factor,step)
    entropy.to_csv(f"entropy_{ticker.replace('^','')}.csv")
    plot_stress(entropy,ticker)
