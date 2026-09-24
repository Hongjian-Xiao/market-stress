"""
market stress monitor website
run:  streamlit run app.py
all calculations live in entropy_stress.py, this file only builds the page
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import entropy_stress as es

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
    stress=es.update_all(prices)
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


# ---------- page ----------

st.title("Market stress monitor")
st.markdown(f"Stress measured as loss of complexity in daily prices. Based on Xiao et al. (2026), "
    f"[*Financial stress evaluation: a complexity science approach*, Financial Innovation 12:30]({paper_url}).  \n"
    f"Questions or feedback: [{contact}](mailto:{contact})")

with st.spinner("Loading prices and updating stress (the first run after a restart can take a minute)..."):
    try:
        prices,metrics=get_data()
    except Exception as err:
        st.error(f"Could not load data from Yahoo Finance: {err}. Please try again in a minute.")
        st.stop()

overall=metrics["mmse_4idx"]
price_date=max(p.index[-1] for p in prices.values())
st.caption(f"Stress up to {overall.index[-1].strftime('%d %b %Y')} · prices up to {price_date.strftime('%d %b %Y')} "
    "(stress lags prices by a few days, see How it works)")

# overall market
st.header("Overall US market")
st.caption("Multivariate entropy of 4 indices (MMSE)")
left,right=st.columns([1,2.3],gap="large")
with left:
    last=overall.iloc[-1]
    st.markdown(pill(last["status"]),unsafe_allow_html=True)
    st.markdown(f'<div style="{num_style};font-size:52px;font-weight:600;line-height:1.2;color:{number_colors[last["status"]]}">'
        f'{last["stress"]:.3f}<span style="font-size:14px;font-weight:400;color:#3d424b"> stress</span></div>',unsafe_allow_html=True)
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
        ("current",f'<span style="color:{number_colors[last["status"]]};font-weight:600">{last["stress"]:.3f} · {last["status"]}</span>'),
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

st.divider()
st.subheader("How it works")
st.write("In calm markets daily prices move randomly (high sample entropy). In crises they become more predictable "
    "(low entropy), so stress = 1 / entropy rises. Each value uses the previous 4 years of prices, so a crisis stays "
    "in the measure for 4 years. The moving-average filter needs 2 days on each side, and the market-wide measure "
    "needs all 4 indices on the same day, so stress lags the latest prices by a few trading days. Changes in stress are "
    "shown as % of the earlier value. Green means good news (lower stress, positive return), red bad news. The status compares today with all earlier days: calm below the 25th percentile, "
    "normal 25–75%, elevated 75–95%, high above 95%. Data: Yahoo Finance, updated daily.")
