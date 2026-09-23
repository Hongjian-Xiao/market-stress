"""
market stress website (step 3)
run:  streamlit run app.py
"""
import streamlit as st
import plotly.graph_objects as go
from entropy_stress import load_prices,rolling_mod_mse,crises

st.set_page_config(page_title="Market stress monitor",page_icon="📉",layout="wide")

tickers={
    "S&P 500":"^GSPC",
    "Russell 2000":"^RUT",
    "Dow Jones":"^DJI",
    "NASDAQ Composite":"^IXIC",
    "Gold":"GC=F",
}


@st.cache_data(ttl=60*60*24,show_spinner=False)
def get_stress(ticker):
    # cached for 24 hours: the first visitor each day triggers a fresh download
    prices=load_prices(ticker,"1991-01-01")
    entropy=rolling_mod_mse(prices,tau=5,window=1044,m=2,r_factor=0.15,step=5)
    return prices,1/entropy


def make_chart(stress,name):
    fig=go.Figure()
    for s,e,label,color in crises:
        fig.add_vrect(x0=s,x1=e,fillcolor=color,opacity=0.35,line_width=0,
            annotation_text=label,annotation_position="top left",annotation_font_size=10)
    fig.add_trace(go.Scatter(x=stress.index,y=stress.values,mode="lines",
        line=dict(color="#1f3b57",width=1.6),name=name,
        hovertemplate="%{x|%d %b %Y}<br>stress %{y:.3f}<extra></extra>"))
    fig.update_layout(height=480,margin=dict(l=10,r=10,t=30,b=10),
        yaxis_title="Stress (1 / sample entropy)",hovermode="x unified",showlegend=False,
        xaxis=dict(rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1,label="1y",step="year",stepmode="backward"),
                dict(count=5,label="5y",step="year",stepmode="backward"),
                dict(count=10,label="10y",step="year",stepmode="backward"),
                dict(step="all",label="All"),
            ])))
    return fig


st.title("Market stress monitor")
st.write("Stress measured as loss of complexity in daily prices, "
    "based on Xiao et al. (2026), *Financial Innovation* 12:30. Higher means more stress.")

choice=st.selectbox("Index",list(tickers.keys()))
ticker=tickers[choice]

with st.spinner(f"Downloading {choice} and computing entropy (about 20 seconds the first time)..."):
    try:
        prices,stress=get_stress(ticker)
    except Exception as err:
        st.error(f"Could not load {choice} from Yahoo Finance: {err}. Try again in a minute.")
        st.stop()

latest=stress.iloc[-1]
median=stress.median()
col1,col2,col3=st.columns(3)
col1.metric("Current stress",f"{latest:.3f}",f"{latest-stress.iloc[-6]:+.3f} vs 1 month ago",delta_color="inverse")
col2.metric("Long-run median",f"{median:.3f}")
col3.metric("Last data point",prices.index[-1].strftime("%d %b %Y"))

st.plotly_chart(make_chart(stress,choice),width="stretch")

st.caption("Each point uses the previous 4 years of prices, so a crisis stays in the "
    "measure for 4 years and drops out suddenly afterwards. Data: Yahoo Finance, refreshed daily.")
