# Market Stress Monitor

A live monitor of US financial market stress, based on the entropy method from my peer-reviewed paper.

**Live app:** https://market-stress-complexity-science.streamlit.app

![Market Stress Monitor](docs/screenshot.png)

## The idea

In calm markets, daily prices move close to randomly, so their sample entropy is high. In a crisis, prices become more predictable and entropy falls. The monitor measures stress as **1 / entropy**, for the overall US market (four indices together) and for each index on its own.

> Xiao H., Xu Y. L., Cukic A., Constantinides A. G., Mandic D. P. (2026). *Financial stress evaluation: a complexity science approach.* Financial Innovation 12:30. https://doi.org/10.1186/s40854-025-00836-2

## Method

Each day, the trend is removed with a 5-day trailing moving average, and sample entropy is computed over the previous 4 years (m = 2, r = 0.15). The status (calm, normal, elevated, high) compares today's stress with all earlier days. Unlike the paper's centred filter, the trailing filter uses no future prices, as needed for real-time monitoring.

## Run locally

```
pip install -r requirements.txt
python entropy_stress.py      # build the full history, only needed once
streamlit run app.py
```

## Data

Daily closing prices from Yahoo Finance, from 1991 onwards. For research and education, not investment advice.
