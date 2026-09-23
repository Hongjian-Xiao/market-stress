# Market stress monitor

A website that tracks financial market stress using entropy-based complexity analysis.
When markets are calm, daily prices behave randomly (high sample entropy). During crises,
prices become more predictable (low sample entropy), so 1 / entropy rises as stress rises.

Based on: Xiao H., Xu Y. L., Cukic A., Constantinides A. G., Mandic D. P. (2026).
Financial stress evaluation: a complexity science approach. *Financial Innovation* 12:30.
https://doi.org/10.1186/s40854-025-00836-2

## Files

- `entropy_stress.py`: data download and entropy calculation
- `app.py`: the website (Streamlit)

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Data: Yahoo Finance, daily closing prices from 1991.
