"""
market stress monitor website
run:  streamlit run app.py
all calculations live in entropy_stress.py, this file only builds the page
"""
import re
import smtplib
from email.message import EmailMessage
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import entropy_stress as es
import trading as tr

st.set_page_config(page_title="Market stress monitor",page_icon="📉",layout="wide")

paper_url="https://link.springer.com/article/10.1186/s40854-025-00836-2"
contact="hongjian.xiao2016@gmail.com"
status_colors={"calm":"#6aa6c8","normal":"#e4e6e9","elevated":"#eeb05a","high":"#a8322a"}
pill_colors={"calm":("#d6e8f2","#1f4f6b"),"normal":("#e4e6e9","#3d424b"),
    "elevated":("#f7dfb8","#7a4a0c"),"high":("#f1d2cf","#7d1f18")}
ranges={"1M":31,"3M":92,"6M":183,"1Y":365,"5Y":365*5,"10Y":3650,"All":None}
ink="#16181d"
number_colors={"calm":"#2d6f94","normal":"#16181d","elevated":"#b8741a","high":"#a8322a"}
good_color="#2e7d4f"
bad_color="#b3412e"
num_style="font-variant-numeric:tabular-nums"
price_color="#2f5d8a"


@st.cache_data(ttl=60*60*6,show_spinner=False)
def get_data():
    # refreshed at most every 6 hours; only new days are calculated
    prices=es.load_all()
    stress=es.update_all(prices,allow_full=False)
    metrics={key:es.daily_metrics(s) for key,s in stress.items()}
    return prices,metrics


# ---------- small page helpers ----------

def pill(label):
    bg,fg=pill_colors.get(label,pill_colors["normal"])
    return (f'<span style="display:inline-block;padding:4px 12px;border-radius:999px;background:{bg};'
        f'color:{fg};font-size:13px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase">{label}</span>')


def rows(items):
    html="".join(f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #e6e3dc">'
        f'<span style="color:#3d424b">{label}</span><span style="{num_style}">{value}</span></div>'
        for label,value in items)
    return f'<div style="font-size:14px">{html}</div>'


def signed(x,digits=3,unit=""):
    return "–" if pd.isna(x) else f"{x:+.{digits}f}{unit}"


def colored(text,value,up_is_bad):
    # muted green for good news, muted red for bad news
    if pd.isna(value) or value==0:
        return text
    bad=(value>0)==up_is_bad
    return f'<span style="color:{bad_color if bad else good_color}">{text}</span>'


def pct_change(row,label):
    # stress change as % of the earlier value, easier to read than 0.002
    ch=row[f"change_{label}"]
    return ch/(row["stress"]-ch)*100


def stress_change(row,label):
    v=pct_change(row,label)
    return colored(signed(v,1,"%"),v,up_is_bad=True)


def return_text(v):
    return colored(signed(v,1,"%"),v,up_is_bad=False)


def above_below(value,reference):
    if pd.isna(reference):
        return "–"
    word="above" if value>reference else "below"
    return colored(word,value-reference,up_is_bad=True)


def cut(df,choice):
    days=ranges[choice]
    if days is None:
        return df
    return df[df.index>=df.index[-1]-pd.Timedelta(days=days)]


def chart_controls(key):
    # period buttons and a switch for the normal line, side by side
    a,b=st.columns([4,1.3])
    with a:
        choice=st.segmented_control("Period",list(ranges.keys()),default="1Y",key=f"range_{key}",
            label_visibility="collapsed") or "1Y"
    with b:
        show_normal=st.toggle("Show normal level",value=True,key=f"normal_{key}")
    return choice,show_normal


def add_status_bands(fig,df,**where):
    for start,end,label in es.status_runs(df):
        if label!="normal":
            fig.add_vrect(x0=start,x1=end,fillcolor=status_colors[label],opacity=0.45,line_width=0,layer="below",
                exclude_empty_subplots=False,**where)  # bands are drawn before the lines exist


def legend():
    items="".join(f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:18px">'
        f'<span style="width:13px;height:13px;border-radius:3px;background:{status_colors[k]};border:1px solid #c9c5bb"></span>'
        f'{k} <span style="color:#6b7079">{r}</span></span>'
        for k,r in [("calm","below 25%"),("normal","25–75%"),("elevated","75–95%"),("high","above 95%")])
    return f'<div style="font-size:12px">{items}<span style="color:#6b7079">· dashed line = normal level</span></div>'


def summary_sentence(df):
    last=df.iloc[-1]
    since=df["percentile"].first_valid_index()
    return (f"{abs(last['vs_normal_pct']):.0f}% {'above' if last['vs_normal_pct']>0 else 'below'} its normal level. "
        f"Higher than {last['percentile']:.0f}% of all days since {since.year}.")


def stress_rows(df):
    last=df.iloc[-1]
    return [
        ("vs yesterday",stress_change(last,"1d")),
        ("vs last week",stress_change(last,"1w")),
        ("vs last month",stress_change(last,"1m")),
        ("vs this year's average",above_below(last["stress"],last["avg_ytd"])),
        ("vs last year's average",above_below(last["stress"],last["avg_last_year"])),
        ("days above normal",f"{int(last['streak_above_normal'])} in a row"),
    ]


def base_layout(fig,height):
    fig.update_layout(height=height,margin=dict(l=10,r=10,t=10,b=10),hovermode="x unified",
        plot_bgcolor="white",showlegend=False)
    fig.update_xaxes(showgrid=True,gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True,gridcolor="#eeeeee")


def normal_note(fig,df):
    fig.add_annotation(text=f"normal level {df['normal'].iloc[-1]:.3f} (hidden)",xref="paper",yref="paper",
        x=0.01,y=0.98,showarrow=False,font=dict(size=11,color="#6b7079"),bgcolor="rgba(255,255,255,0.8)")


def stress_chart(df,show_normal):
    fig=go.Figure()
    add_status_bands(fig,df)
    if show_normal:
        fig.add_trace(go.Scatter(x=df.index,y=df["normal"],mode="lines",line=dict(color=ink,width=1,dash="dash"),
            name="normal",hovertemplate="%{y:.3f}"))
    else:
        normal_note(fig,df)
    fig.add_trace(go.Scatter(x=df.index,y=df["stress"],mode="lines",line=dict(color=ink,width=1.6),
        name="stress",hovertemplate="%{y:.3f}"))
    base_layout(fig,360)
    fig.update_yaxes(title_text="stress")
    return fig


def detail_chart(df,prices,show_normal):
    p=prices[prices.index>=df.index[0]]
    returns=p.pct_change()*100
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.72,0.28],vertical_spacing=0.04,
        specs=[[{"secondary_y":True}],[{}]])
    add_status_bands(fig,df,row=1,col=1)
    if show_normal:
        fig.add_trace(go.Scatter(x=df.index,y=df["normal"],mode="lines",line=dict(color=ink,width=1,dash="dash"),
            name="normal",hovertemplate="%{y:.3f}"),row=1,col=1,secondary_y=False)
    else:
        normal_note(fig,df)
    fig.add_trace(go.Scatter(x=df.index,y=df["stress"],mode="lines",line=dict(color=ink,width=1.6),
        name="stress",hovertemplate="%{y:.3f}"),row=1,col=1,secondary_y=False)
    fig.add_trace(go.Scatter(x=p.index,y=p.values,mode="lines",line=dict(color=price_color,width=1.3),
        name="price",hovertemplate="%{y:,.0f}"),row=1,col=1,secondary_y=True)
    fig.add_trace(go.Bar(x=returns.index,y=returns.values,name="daily return",hovertemplate="%{y:+.2f}%",
        marker_color=np.where(returns.values>=0,price_color,"#b5542c"),marker_line_width=0),row=2,col=1)
    base_layout(fig,560)
    fig.update_layout(bargap=0)
    fig.update_yaxes(title_text="stress",row=1,col=1,secondary_y=False)
    fig.update_yaxes(title_text="price",row=1,col=1,secondary_y=True,showgrid=False,
        title_font_color=price_color,tickfont_color=price_color)
    fig.update_yaxes(title_text="return %",row=2,col=1)
    return fig


# ---------- my stocks: trend + entropy strategy (trading.py) ----------

buy_color="#1f6f4a"
sell_color="#8a2b4f"
held_color="#eb6834"


@st.cache_data(ttl=60*60,show_spinner=False)
def get_trading():
    # refreshed at most every hour; positions are replayed from tr.start_close, nothing is stored
    return tr.run(allow_full=False)


def trade_word(old,new):
    if new>old:
        return "buy full" if new-old>0.9 else "buy half"
    return "sell all" if new==0 else "sell half"


def action_of(r):
    # (kind, label) of today's action
    if r["action"] is None:
        if r["held"]==0:
            return "out","Stay out"
        return "hold","Hold full" if r["held"]==1 else "Hold half"
    old,new=r["action"]
    word=trade_word(old,new)
    return ("buy" if new>old else "sell"),("▲ " if new>old else "▼ ")+word.capitalize()


def action_pill(r):
    kind,label=action_of(r)
    bg,fg,border={"buy":(buy_color,"#ffffff","none"),"sell":(sell_color,"#ffffff","none"),
        "hold":("#ffffff",ink,f"1px solid {ink}"),"out":("#ebe8e1","#3d424b","none")}[kind]
    return (f'<span style="display:inline-block;padding:4px 12px;border-radius:999px;background:{bg};color:{fg};border:{border};'
        f'font-size:13px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap">{label}</span>')


def held_text(x):
    return {0:"—",0.5:"half",1:"full"}.get(x,f"{x:.0%}")


def date_box(label,value,dark=False):
    bg,fg,sub=("#16181d","#ffffff","#c9c5bb") if dark else ("#ffffff",ink,"#6b7079")
    return (f'<div style="padding:14px 18px;margin-bottom:8px;background:{bg};border:1px solid #d8d4ca;border-radius:10px">'
        f'<div style="font-size:12px;color:{sub};text-transform:uppercase;letter-spacing:0.06em">{label}</div>'
        f'<div style="{num_style};font-size:20px;color:{fg}">{value}</div></div>')


def count_tiles(res):
    kinds=[action_of(r)[0] for r in res.values()]
    tiles=[("buy",buy_color,"#ffffff"),("sell",sell_color,"#ffffff"),("hold","#ffffff",ink),("out","#ebe8e1","#3d424b")]
    html="".join(f'<div style="flex:1;padding:12px 14px;border-radius:10px;background:{bg};color:{fg};border:1px solid #d8d4ca">'
        f'<div style="font-size:12px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase">{"stay out" if k=="out" else k}</div>'
        f'<div style="{num_style};font-size:28px">{kinds.count(k)}</div></div>' for k,bg,fg in tiles)
    return f'<div style="display:flex;gap:14px">{html}</div>'


def action_rows(res):
    # each stock has its own dates: a london stock can be a day ahead of the us ones
    grid="display:grid;grid-template-columns:90px 170px 1fr 200px;gap:16px"
    head=(f'<div style="{grid};padding-bottom:6px;border-bottom:1px solid #d8d4ca;font-size:12px;color:#6b7079;'
        f'text-transform:uppercase;letter-spacing:0.06em"><span>stock</span><span>action</span><span>why</span><span>based on → act on</span></div>')
    body="".join(f'<div style="{grid};align-items:center;padding:10px 0;border-bottom:1px solid #e6e3dc">'
        f'<span style="font-weight:600">{r["ticker"]}</span><span>{action_pill(r)}</span><span style="font-size:14px">{r["reason"]}</span>'
        f'<span style="{num_style};font-size:13px">{r["last_date"]:%d %b} close → {tr.next_trading_day(r["last_date"],r["ticker"]):%a %d %b}</span></div>'
        for r in res.values() if r["action"] is not None)
    return head+body


def trade_chart(r,choice):
    p=cut(r["prices"][["Close"]],choice)["Close"]
    pos=r["position"].reindex(p.index).fillna(0)
    fig=go.Figure()
    for _,part in pos.groupby((pos!=pos.shift()).cumsum()):
        size=part.iloc[0]
        if size>0:
            nxt=p.index.searchsorted(part.index[-1])+1
            end=p.index[nxt] if nxt<len(p) else part.index[-1]
            fig.add_vrect(x0=part.index[0],x1=end,fillcolor=held_color,opacity=0.28 if size==1 else 0.1,line_width=0,layer="below")
    fig.add_trace(go.Scatter(x=p.index,y=p.values,mode="lines",line=dict(color=ink,width=1.4),name="price",hovertemplate="%{y:,.2f}"))
    trades=r["trades"]
    trades=trades[(trades["act"]>=p.index[0])&trades["price"].notna()]
    for kind,symbol,color in [("buy","triangle-up",buy_color),("sell","triangle-down",sell_color)]:
        sel=trades[trades["to"]>trades["from"]] if kind=="buy" else trades[trades["to"]<trades["from"]]
        if len(sel):
            big=(sel["to"]-sel["from"]>0.9) if kind=="buy" else (sel["to"]==0)
            fig.add_trace(go.Scatter(x=sel["act"],y=sel["price"],mode="markers",name=kind,
                marker=dict(symbol=symbol,color=color,size=np.where(big,14,9),line=dict(color="white",width=1)),
                hovertemplate=f"{kind} at %{{y:,.2f}}<extra></extra>"))
    base_layout(fig,420)
    fig.update_yaxes(type="log",title_text="price")
    return fig


# ---------- page ----------

st.title("Market stress monitor")
st.markdown(f"Stress measured as loss of complexity in daily prices. Based on Xiao et al. (2026), "
    f"[*Financial stress evaluation: a complexity science approach*, Financial Innovation 12:30]({paper_url}).  \n"
    f"Questions or feedback: [{contact}](mailto:{contact})")

with st.spinner("Loading prices and updating stress..."):
    try:
        prices,metrics=get_data()
    except es.HistoryMissing as err:
        st.error(f"The stress history is not built yet ({err}). Run research/05_build_data.ipynb, then push the data folder.")
        st.stop()
    except Exception as err:
        st.error(f"Could not load data from Yahoo Finance: {err}. Please try again in a minute.")
        st.stop()

overall=metrics["mmse_4idx"]

# overall market
st.header("Overall US market")
st.caption("Multivariate entropy of 4 indices (MMSE)")
left,right=st.columns([1,2.3],gap="large")
with left:
    last=overall.iloc[-1]
    st.markdown(pill(last["status"]),unsafe_allow_html=True)
    st.markdown(f'<div style="{num_style};font-size:52px;font-weight:600;line-height:1.2;color:{number_colors[last["status"]]}">'
        f'{last["stress"]:.3f}<span style="font-size:14px;font-weight:400;color:#3d424b"> stress</span></div>'
        f'<div style="font-size:13px;color:#6b7079">as of {overall.index[-1].strftime("%d %b %Y")}</div>',unsafe_allow_html=True)
    st.write(summary_sentence(overall))
    st.markdown(rows(stress_rows(overall)),unsafe_allow_html=True)
with right:
    choice,show_normal=chart_controls("overall")
    st.plotly_chart(stress_chart(cut(overall,choice),show_normal),width="stretch")
    st.markdown(legend(),unsafe_allow_html=True)

# individual indices
st.header("Individual indices")
st.caption("Univariate entropy of each index (MSE)")
if "picked" not in st.session_state:
    st.session_state["picked"]="snp"


def pick(key):
    st.session_state["picked"]=key


cards=st.columns(4)
for col,(key,name) in zip(cards,es.names.items()):
    last=metrics[f"mse_{key}"].iloc[-1]
    selected=st.session_state["picked"]==key
    with col:
        with st.container(border=True):
            st.button(name,key=f"card_{key}",type="primary" if selected else "secondary",width="stretch",
                on_click=pick,args=(key,),help="Show details below")
            st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="{num_style};font-size:28px;font-weight:600;color:{number_colors[last["status"]]}">{last["stress"]:.3f}</span>'
                f'{pill(last["status"])}</div>'
                f'<div style="font-size:13px;color:#3d424b">{colored(signed(last["vs_normal_pct"],0,"%"),last["vs_normal_pct"],True)} vs normal · '
                f'{stress_change(last,"1m")} vs last month</div>',unsafe_allow_html=True)

key=st.session_state["picked"]
picked=es.names[key]
df=metrics[f"mse_{key}"]

# detail
st.subheader(f"{picked} in detail")
choice,show_normal=chart_controls("detail")
st.plotly_chart(detail_chart(cut(df,choice),prices[key],show_normal),width="stretch")
st.markdown(legend()+'<div style="font-size:12px;color:#6b7079">black = stress (left axis) · blue = price (right axis) · bottom = daily return</div>',
    unsafe_allow_html=True)

ret=es.returns_summary(prices[key])
last=df.iloc[-1]
a,b=st.columns(2,gap="large")
with a:
    st.markdown("**Stress**")
    st.markdown(rows([
        ("current",f'<span style="color:{number_colors[last["status"]]};font-weight:600">{last["stress"]:.3f} · {last["status"]}</span>'
            f' ({df.index[-1].strftime("%d %b %Y")})'),
        ("vs normal level",colored(signed(last["vs_normal_pct"],0,"%"),last["vs_normal_pct"],True)),
        ("percentile in history",f"{last['percentile']:.0f}%"),
        ("vs yesterday / last week / last month",f"{stress_change(last,'1d')} / {stress_change(last,'1w')} / {stress_change(last,'1m')}"),
        ("vs this year / last year average",f"{above_below(last['stress'],last['avg_ytd'])} / {above_below(last['stress'],last['avg_last_year'])}"),
    ]),unsafe_allow_html=True)
with b:
    st.markdown("**Returns**")
    st.markdown(rows([
        ("last close",f"{ret['last_close']:,.2f} ({ret['date'].strftime('%d %b %Y')})"),
        ("1 day / 1 week",f"{return_text(ret['r_1d'])} / {return_text(ret['r_1w'])}"),
        ("1 month / year to date",f"{return_text(ret['r_1m'])} / {return_text(ret['r_ytd'])}"),
        ("1 year",return_text(ret["r_1y"])),
        ("below all-time high",return_text(ret["drawdown"])),
    ]),unsafe_allow_html=True)

# today's actions (trend + entropy strategy on my stocks)
with st.spinner("Updating my stocks..."):
    trading=get_trading()
res={t:r for t,r in trading.items() if "error" not in r}
failed=[t for t,r in trading.items() if "error" in r]

st.divider()
st.header("Today's actions")
if res:
    us=[r for r in res.values() if tr.exchange(r["ticker"])[0]=="America/New_York"] or list(res.values())
    based=max(r["last_date"] for r in us)
    act_on=tr.next_trading_day(based,us[0]["ticker"])
    a,b,c=st.columns([1,1,2],gap="medium")
    with a:
        st.markdown(date_box("based on the US close of",f"{based:%a %d %b %Y}"),unsafe_allow_html=True)
    with b:
        st.markdown(date_box("act on (US stocks)",f"{act_on:%a %d %b %Y}, at the open",dark=True),unsafe_allow_html=True)
    late=[t for t,r in res.items() if r["last_date"]<tr.expected_last_close(t)]
    with c:
        if late:
            since=min(res[t]["last_date"] for t in late)
            st.warning(f"No new close since {since:%a %d %b %Y} for {', '.join(late)}. Do not trade on this list for these stocks.")
    st.markdown('<div style="height:28px"></div>',unsafe_allow_html=True)
    left,right=st.columns([1,2.3],gap="large")
    with left:
        st.markdown(count_tiles(res),unsafe_allow_html=True)
    with right:
        if any(r["action"] is not None for r in res.values()):
            st.markdown(action_rows(res),unsafe_allow_html=True)
            st.caption("Other stocks: no change.")
        else:
            nxt=min(r["next_check"] for r in res.values())
            st.write(f"No actions today. The next decision is on the close of {nxt:%a %d %b %Y}.")
if failed:
    st.caption(f"Could not load: {', '.join(failed)}")

# my stocks
if res:
    st.markdown('<div style="height:32px"></div>',unsafe_allow_html=True)
    st.header("My stocks")
    st.caption("Today's action, position, trend and stress of each stock · select one for detail")
    if st.session_state.get("stock") not in res:
        st.session_state["stock"]=next(iter(res))


    def pick_stock(t):
        st.session_state["stock"]=t


    def line(label,value):
        return f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px"><span>{label}</span><span>{value}</span></div>'


    tickers=list(res)
    for first in range(0,len(tickers),5):
        for col,t in zip(st.columns(5),tickers[first:first+5]):
            r=res[t]
            s=r["stress"]
            stress_html=pill(s["status"].iloc[-1]) if s is not None and not pd.isna(s["status"].iloc[-1]) else '<span style="color:#6b7079">not enough history</span>'
            with col:
                with st.container(border=True):
                    st.button(t,key=f"stock_{t}",type="primary" if st.session_state["stock"]==t else "secondary",width="stretch",
                        on_click=pick_stock,args=(t,),help="Show details below")
                    st.markdown(action_pill(r)+'<div style="font-size:13px;color:#3d424b;margin-top:8px">'
                        +line("close",f'<span style="{num_style}">{r["close"]:,.2f} ({r["last_date"]:%d %b})</span>')
                        +line("held",held_text(r["held"]))
                        +line("trend",r["trend"])
                        +line("stress",stress_html)+'</div>',unsafe_allow_html=True)

    t=st.session_state["stock"]
    r=res[t]
    st.subheader(f"{t} in detail")
    st.markdown(action_pill(r),unsafe_allow_html=True)
    last=r["last_action"]
    since=f" since {r['since']:%d %b %Y}" if r["since"] is not None and r["held"]>0 else ""
    st.markdown(rows([
        ("held now",held_text(r["held"])+since),
        ("last action","—" if last is None else f"{trade_word(last['from'],last['to'])} on {last['act']:%d %b %Y}"),
        ("open gain",return_text(r["open_gain"]) if r["held"]>0 else "—"),
    ]),unsafe_allow_html=True)
    choice=st.segmented_control("Period",list(ranges.keys()),default="1Y",key="range_stock",label_visibility="collapsed") or "1Y"
    st.plotly_chart(trade_chart(r,choice),width="stretch")
    st.markdown('<div style="font-size:12px;color:#6b7079">black = price (log scale) · shading = held (dark = full, light = half) · '
        '▲ buy · ▼ sell (big = full / all, small = half)</div>',unsafe_allow_html=True)
    if r["stress"] is not None:
        show_normal=st.toggle("Show normal level",value=True,key="normal_stock")
        st.plotly_chart(stress_chart(cut(r["stress"],choice),show_normal),width="stretch")
        st.markdown(legend(),unsafe_allow_html=True)
    else:
        st.caption(f"Stress needs 4 years of prices: not available for {t} yet.")

st.divider()


# ---------- how it works ----------

st.subheader("How it works")
st.markdown(
    "**The idea.** In calm markets daily prices move randomly, which gives high sample entropy. "
    "In a crisis they become more predictable and entropy falls, so stress = 1 / entropy rises.\n\n"
    "**The calculation.** The trend is removed with a 5-day trailing moving average, so each value only uses prices "
    "available on that day. Each value then looks back over 4 years of prices, so a crisis stays in the measure for "
    "4 years. The overall market combines all 4 indices (multivariate entropy) and needs all of them on the same day.\n\n"
    "**The status.** Today's stress is compared with all earlier days: calm below the 25th percentile, normal 25–75%, "
    "elevated 75–95%, high above 95%. Changes in stress are shown as % of the earlier value. "
    "Green means good news (lower stress, positive return), red means bad news.")
st.caption("Data source: Yahoo Finance, daily closing prices since 1991, updated daily.")


# ---------- request form ----------

def send_request(kind,what,email,message):
    # sends the request to the owner's inbox; login details come from streamlit secrets, never from the code
    conf=st.secrets["email"]
    msg=EmailMessage()
    msg["Subject"]=f"Market stress monitor request: {kind} – {what or 'no name given'}"
    msg["From"]=conf["user"]
    msg["To"]=conf["to"]
    msg["Reply-To"]=email
    msg.set_content(f"Request type: {kind}\nTicker / name: {what}\nFrom: {email}\n\n{message}")
    with smtplib.SMTP_SSL("smtp.gmail.com",465,timeout=20) as server:
        server.login(conf["user"],conf["app_password"])
        server.send_message(msg)


def email_configured():
    # true when the gmail login is set in streamlit secrets; no secrets file at all is fine too
    try:
        return "email" in st.secrets
    except Exception:
        return False


def email_ok(text):
    return re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",text.strip()) is not None


st.divider()
st.subheader("Request a ticker or feature")
st.write("Would you like to see another index, stock or market here, or have an idea for the monitor? Send a request.")

if st.session_state.get("request_sent"):
    st.success("Thank you, your request has been sent. I'll get back to you by email.")
elif not email_configured():
    st.info(f"Please send requests by email to [{contact}](mailto:{contact}).")
else:
    with st.form("request",clear_on_submit=False):
        a,b=st.columns(2)
        with a:
            kind=st.selectbox("Request type",["New ticker or index","New feature","Question","Other"])
            what=st.text_input("Ticker or name (optional)",placeholder="e.g. FTSE 100, AAPL, gold",max_chars=100)
        with b:
            email=st.text_input("Your email",placeholder="so I can reply",max_chars=200)
            message=st.text_area("Message",max_chars=2000,height=108)
        st.caption("Your email is only used to reply to your request.")
        submitted=st.form_submit_button("Send request")
    if submitted:
        if not email_ok(email):
            st.warning("Please enter a valid email address.")
        elif not (what.strip() or message.strip()):
            st.warning("Please add a ticker name or a short message.")
        else:
            try:
                send_request(kind,what.strip(),email.strip(),message.strip())
                st.session_state["request_sent"]=True
                st.rerun()
            except Exception:
                st.error(f"Sorry, the request could not be sent. Please email [{contact}](mailto:{contact}) instead.")
