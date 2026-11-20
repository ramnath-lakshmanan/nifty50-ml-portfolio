import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import joblib

from pypfopt import EfficientFrontier, risk_models, expected_returns

from pypfopt.objective_functions import L2_reg

import plotly.express as px

import plotly.graph_objects as go

import warnings

warnings.filterwarnings("ignore")



st.set_page_config(page_title="Nifty 50 ML Portfolio Optimizer", layout="wide")

st.title("Nifty 50 ML-Enhanced Portfolio Optimizer")

st.caption("Using trained XGBoost models + Equity Curve vs Nifty 50")



@st.cache_resource

def load_models():

    try:

        models = joblib.load("models/xgb_models.joblib")

        feature_cols = joblib.load("models/feature_cols.joblib")

        return models, feature_cols

    except Exception as e:

        st.error(f"Could not load models: {e}")

        return None, None



models, feature_cols_dict = load_models()



st.sidebar.header("Settings")

blend = st.sidebar.slider("ML Blend Weight", 0.0, 1.0, 0.4, 0.05)

lookback_years = st.sidebar.slider("Lookback Years", 1, 5, 2)



BENCHMARK = "^NSEI"



@st.cache_data(ttl=3600)

def load_prices(tickers):

    data = yf.download(tickers, start="2019-01-01", auto_adjust=True, progress=False)["Close"]

    return data.dropna(how="all", axis=1)



if models:

    NIFTY50 = list(models.keys())

    prices = load_prices(NIFTY50 + [BENCHMARK])

    

    if st.sidebar.button("Run Optimization", type="primary"):

        with st.spinner("Running ML optimization..."):

            end_date = prices.index[-1]

            start_date = end_date - pd.DateOffset(years=lookback_years)

            window = prices.loc[start_date:end_date]

            stock_window = window.drop(columns=[BENCHMARK], errors="ignore")

            

            # Historical mean

            hist_mu = expected_returns.mean_historical_return(stock_window, frequency=252)

            

            # ML predictions

            ml_mu = hist_mu.copy()

            for ticker in hist_mu.index:

                if ticker in models and ticker in stock_window.columns:

                    price = stock_window[ticker].dropna()

                    if len(price) > 63:

                        ret = price.pct_change()

                        features = pd.DataFrame({

                            "ret_lag_1": [ret.iloc[-1]],

                            "ret_lag_5": [ret.iloc[-5] if len(ret) > 5 else 0],

                            "mom_21": [price.iloc[-1] / price.iloc[-21] - 1],

                            "vol_21": [ret.tail(21).std()]

                        })

                        feat_cols = feature_cols_dict.get(ticker, features.columns)

                        features = features.reindex(columns=feat_cols, fill_value=0)

                        pred_daily = models[ticker].predict(features)[0]

                        ml_mu[ticker] = pred_daily * 252

            

            final_mu = (1 - blend) * hist_mu + blend * ml_mu

            final_mu = final_mu.clip(-0.4, 0.8)

            

            # Optimize

            S = risk_models.sample_cov(stock_window, frequency=252)

            ef = EfficientFrontier(final_mu, S)

            ef.add_objective(L2_reg, gamma=0.1)

            try:

                weights = ef.max_sharpe()

            except:

                weights = ef.min_volatility()

            

            cleaned = ef.clean_weights()

            weights_df = pd.Series(cleaned)

            weights_df = weights_df[weights_df > 0.01].sort_values(ascending=False)

            

            # === Equity Curve ===

            returns = stock_window.pct_change().dropna()

            port_weights = weights_df.reindex(returns.columns).fillna(0)

            port_weights = port_weights / port_weights.sum()

            port_ret = (returns * port_weights).sum(axis=1)

            port_equity = (1 + port_ret).cumprod()

            

            nifty_ret = window[BENCHMARK].pct_change().dropna()

            nifty_equity = (1 + nifty_ret).cumprod()

            

            # Align dates

            common_idx = port_equity.index.intersection(nifty_equity.index)

            port_equity = port_equity.loc[common_idx]

            nifty_equity = nifty_equity.loc[common_idx]

            nifty_equity = nifty_equity / nifty_equity.iloc[0] * port_equity.iloc[0]

            

            # Display

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Portfolio Weights")

                fig_pie = px.pie(values=weights_df.values, names=weights_df.index)

                st.plotly_chart(fig_pie, use_container_width=True)

            

            with col2:

                st.subheader("Top Holdings")

                st.dataframe(weights_df.head(12).to_frame("Weight").style.format("{:.1%}"))

            

            st.subheader("Equity Curve Comparison")

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=port_equity.index, y=port_equity, name="ML Portfolio", line=dict(width=2.5)))

            fig.add_trace(go.Scatter(x=nifty_equity.index, y=nifty_equity, name="Nifty 50", line=dict(width=2)))

            fig.update_layout(

                title="Growth of ₹1",

                yaxis_title="Portfolio Value",

                hovermode="x unified",

                height=500

            )

            st.plotly_chart(fig, use_container_width=True)

            

            # Simple metrics

            port_total = port_equity.iloc[-1] / port_equity.iloc[0] - 1

            nifty_total = nifty_equity.iloc[-1] / nifty_equity.iloc[0] - 1

            col_a, col_b = st.columns(2)

            col_a.metric("ML Portfolio Return", f"{port_total:.1%}")

            col_b.metric("Nifty 50 Return", f"{nifty_total:.1%}")

            

            st.success(f"Optimization complete | Blend: {blend} | Lookback: {lookback_years} years")

else:

    st.warning("Models not found. Please save them from the notebook first.")