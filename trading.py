"""
trading engine for the website: the trend + entropy strategy on my stocks
same rules as research/08_trade_simulation.ipynb

nothing is saved about positions: every run replays the strategy from start_close with the prices,
so what is held today always follows from the prices (as long as every action was done)

run once locally, then push data/:  python trading.py
(builds the stress history of the stocks in data/ and prints today's actions)
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar,Holiday,sunday_to_monday,nearest_workday,
    USMartinLutherKingJr,USPresidentsDay,GoodFriday,USMemorialDay,USLaborDay,USThanksgivingDay)
from pandas.tseries.offsets import CustomBusinessDay,BDay
import entropy_stress as es

# ---------- settings ----------

stocks=["SNDK","CORZ","NBIS","MU","NVDA","META","SMSN.IL","AAPL","TSLA","HOOD"]
start_close="2026-09-29"    # first close the strategy decides on, nothing held before; its actions are done at the next open
check_every=5               # decide every 5 trading days, counted from start_close (as tested); 1 = every day
history_years=12            # indicators use prices from 12 years before start_close (10 years + 2 warm-up, as in notebook 08)

fast_ema=20
slow_ema=60
entropy_window=126          # 6 months, on log prices
entropy_r=0.20
min_history=252             # days of past entropy before its percentile is used
high_cut=75                 # entropy percentile above this -> hold half
back_cut=60                 # back to full only below this
high_size=0.5


# ---------- trading days ----------

class NyseHolidays(AbstractHolidayCalendar):
    rules=[
        Holiday("new year",month=1,day=1,observance=sunday_to_monday),
        USMartinLutherKingJr,USPresidentsDay,GoodFriday,USMemorialDay,
        Holiday("juneteenth",month=6,day=19,start_date="2022-01-01",observance=nearest_workday),
        Holiday("independence day",month=7,day=4,observance=nearest_workday),
        USLaborDay,USThanksgivingDay,
        Holiday("christmas",month=12,day=25,observance=nearest_workday),
    ]


us_day=CustomBusinessDay(calendar=NyseHolidays())


def exchange(t):
    # time zone and closing time (hour, minute) of the stock's exchange
    if t.endswith((".IL",".L")):
        return "Europe/London",(16,30),BDay()
    return "America/New_York",(16,0),us_day


def next_trading_day(day,t):
    return (pd.Timestamp(day)+exchange(t)[2]).normalize()


def expected_last_close(t):
    # the last close that should be in the data by now
    tz,(hh,mm),cal=exchange(t)
    now=datetime.now(ZoneInfo(tz))
    today=pd.Timestamp(now.date())
    closed=now.hour*60+now.minute>=hh*60+mm+30
    if cal.is_on_offset(today) and closed:
        return today
    return (today-cal).normalize()


# ---------- prices (finished days only) ----------

def load_prices(t):
    import yfinance as yf
    tz,(hh,mm),cal=exchange(t)
    data=yf.Ticker(t).history(start=es.start,auto_adjust=True)[["Open","Close"]]
    data.index=data.index.tz_localize(None).normalize()
    data=data[~data.index.duplicated(keep="last")].dropna()
    data=data[(data>0).all(axis=1)].sort_index()
    now=datetime.now(ZoneInfo(tz))
    if now.hour*60+now.minute<hh*60+mm+15:
        data=data[data.index<pd.Timestamp(now.date())]     # today's row is not a finished day yet
    return data


# ---------- indicators and strategy ----------

def rolling_entropy(logp):
    # same as research helpers.rolling_entropy: trailing detrend, sample entropy of the last 126 days
    y=es.ma_detrend(es.standardize(logp))
    out={}
    for end in range(entropy_window,len(y)+1):
        seg=y.values[end-entropy_window:end]
        out[y.index[end-1]]=es.sample_entropy(seg/seg.std(),es.m,entropy_r)
    return pd.Series(out,dtype=float).replace([np.inf,-np.inf],np.nan).reindex(logp.index)


def indicators(close):
    p=close[close.index>=pd.Timestamp(start_close)-pd.DateOffset(years=history_years)]
    df=pd.DataFrame({"price":p})
    df["ema_fast"]=p.ewm(span=fast_ema,adjust=False).mean()
    df["ema_slow"]=p.ewm(span=slow_ema,adjust=False).mean()
    df.loc[df.index[:slow_ema],["ema_fast","ema_slow"]]=np.nan
    df["ent"]=rolling_entropy(np.log(p))
    df["ent_pct"]=df["ent"].expanding(min_periods=min_history).rank(pct=True)*100
    return df


def strategy(df):
    up=df["ema_fast"]>df["ema_slow"]
    high=pd.Series(np.nan,index=df.index)
    high[df["ent_pct"]>=high_cut]=1
    high[df["ent_pct"]<=back_cut]=0
    high=high.ffill().fillna(0)
    target=up.astype(float)
    target[up&(high==1)]=high_size
    return target.where(df["ema_slow"].notna()),up,high


def reason(old,new):
    if new==0:
        return "downtrend started"
    if old==0:
        return "uptrend started" if new==1 else "uptrend started, entropy high: half"
    if new>old:
        return "entropy back to normal: half → full"
    return "entropy high: full → half"


# ---------- replay from start_close ----------

def replay(t,data):
    df=indicators(data["Close"])
    target,up,high=strategy(df)
    days=df.index[df.index>=pd.Timestamp(start_close)]
    held=0.0
    log=[]
    for i,day in enumerate(days):
        if i%check_every:
            continue
        want=target[day]
        if np.isnan(want) or want==held:
            continue
        act=next_trading_day(day,t)
        price=data["Open"].get(act,np.nan)              # nan while the open has not happened yet
        log.append({"decided":day,"act":act,"from":held,"to":want,"price":price,"reason":reason(held,want)})
        held=want
    last=df.index[-1]
    today=log[-1] if log and log[-1]["decided"]==last else None
    done=[e for e in log if e is not today]

    # position, cost and shares of what is held now (one unit = a full buy)
    shares=0.0
    cost=0.0
    since=None
    position=pd.Series(0.0,index=data.index)
    for e in done:
        if np.isnan(e["price"]):
            continue
        if e["to"]>e["from"]:
            if e["from"]==0:
                since=e["act"]
            shares+=(e["to"]-e["from"])/e["price"]
            cost+=e["to"]-e["from"]
        else:
            part=(e["from"]-e["to"])/e["from"]
            shares-=shares*part
            cost-=cost*part
            if e["to"]==0:
                since=None
        position[position.index>=e["act"]]=e["to"]
    held_now=done[-1]["to"] if done else 0.0
    close=data["Close"].iloc[-1]
    open_gain=(shares*close/cost-1)*100 if cost>0 else np.nan

    n=len(days)
    if n:
        next_check=(last+(check_every-(n-1)%check_every)*exchange(t)[2]).normalize()
    else:
        next_check=pd.Timestamp(start_close)

    return {
        "ticker":t,
        "last_date":last,
        "close":close,
        "held":held_now,
        "action":(today["from"],today["to"]) if today else None,
        "reason":today["reason"] if today else "",
        "trend":"up" if up.iloc[-1] else "down",
        "ent_pct":df["ent_pct"].iloc[-1],
        "since":since,
        "open_gain":open_gain,
        "last_action":done[-1] if done else None,
        "trades":pd.DataFrame(done,columns=["decided","act","from","to","price","reason"]),
        "position":position,
        "prices":data,
        "next_check":next_check,
        "check_day":today is not None or (n>0 and (n-1)%check_every==0),
    }


def stock_stress(t,close,allow_full):
    # same stress as the indices (entropy_stress.py), saved in data/ like them
    key="mse_"+t.lower().replace(".","_")
    return es.update_series(key,es.ma_detrend(es.standardize(close)),es.mse_window,allow_full)


def run(allow_full=False):
    out={}
    for t in stocks:
        try:
            data=load_prices(t)
            res=replay(t,data)
        except Exception as err:
            out[t]={"ticker":t,"error":str(err)}
            continue
        try:
            s=stock_stress(t,data["Close"],allow_full).dropna()
            res["stress"]=es.daily_metrics(s) if len(s) else None
        except es.HistoryMissing:
            res["stress"]=None
        out[t]=res
    return out


if __name__=="__main__":
    for t,r in run(allow_full=True).items():
        if "error" in r:
            print(f"{t}: error {r['error']}")
            continue
        a=f"{r['action'][0]:.0%} -> {r['action'][1]:.0%} ({r['reason']})" if r["action"] else "no change"
        print(f"{t}: close {r['last_date'].date()}, held {r['held']:.0%}, today: {a}, "
              f"stress {'not enough history' if r['stress'] is None else len(r['stress'])}")
