"""
AI Stock Analyzer v2
====================
Verschil met v1: het signaal wordt alleen als betrouwbaar getoond wanneer de
modellen in een walk-forward test (out-of-sample, met purge) aantoonbaar beter
zijn dan een naieve basislijn. Anders staat er "ONVOLDOENDE BEWIJS".

Geen financieel advies.
"""
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

warnings.filterwarnings("ignore")

SEED = 42
H_SHORT = 5                                # 1 week = 5 handelsdagen
H_LONG = {"1 jaar": 252, "2 jaar": 504}
BLOCK = 21                                 # lengte validatieblok (handelsdagen)
MIN_TRAIN = 250                            # minimaal aantal trainingsrijen
MIN_OOS = 100                              # minimaal aantal out-of-sample dagen
MIN_EFF_N = 20                             # minimaal aantal "onafhankelijke" metingen


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("Geen koersdata gevonden voor deze ticker")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Kolommen ontbreken: {missing}")
    df = df[needed].dropna(subset=["Close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if len(df) < 500:
        raise ValueError(f"Te weinig historie ({len(df)} dagen, minimaal 500)")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alleen stationaire features (geen ruwe prijsniveaus)."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)

    for n in (1, 5, 10, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)

    sma10, sma20, sma50 = (c.rolling(n).mean() for n in (10, 20, 50))
    f["dist_sma10"] = c / sma10 - 1
    f["dist_sma50"] = c / sma50 - 1
    f["sma10_vs_sma50"] = sma10 / sma50 - 1

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_pct"] = macd / c
    f["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    # RSI met Wilder-smoothing
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = (100 - 100 / (1 + gain / loss)) / 100

    r1 = c.pct_change()
    vol10, vol20 = r1.rolling(10).std(), r1.rolling(20).std()
    f["vol_20"] = vol20
    f["vol_ratio"] = vol10 / vol20

    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c

    f["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    f["dist_high_252"] = c / c.rolling(252, min_periods=100).max() - 1
    f["boll_pos"] = (c - sma20) / (2 * c.rolling(20).std())

    return f.replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.Series, horizon: int) -> pd.Series:
    """1 als de koers over `horizon` dagen hoger staat, 0 als niet, NaN als nog onbekend."""
    fwd = close.shift(-horizon) / close - 1
    y = (fwd > 0).astype(float)
    y[fwd.isna()] = np.nan
    return y


# ----------------------------------------------------------------------------
# Modellen
# ----------------------------------------------------------------------------
def short_models() -> dict:
    models = {
        "Logistic Regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=150, max_depth=4, min_samples_leaf=30,
            max_features="sqrt", random_state=SEED, n_jobs=1),
    }
    if xgb is not None:
        models["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
            eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0)
    if lgb is not None:
        models["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.03, num_leaves=4,
            min_child_samples=30, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED,
            n_jobs=1, verbosity=-1)
    return models


def long_models() -> dict:
    return {
        "Logistic Regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.05, max_iter=1000)),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=150, max_depth=3, min_samples_leaf=50,
            max_features="sqrt", random_state=SEED, n_jobs=1),
    }


def proba_up(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return p[:, classes.index(1)]


# ----------------------------------------------------------------------------
# Validatie
# ----------------------------------------------------------------------------
def walk_forward(X, y, horizon, factories, n_blocks, block=BLOCK, min_train=MIN_TRAIN):
    """
    Expanding-window walk-forward met purge.
    Training eindigt `horizon` rijen voor het begin van het testblok, zodat de
    labels van de training nooit in de testperiode kijken.
    """
    n = len(X)
    parts = []
    for b in range(n_blocks):
        end = n - b * block
        start = end - block
        train_end = start - horizon
        if train_end < min_train:
            break
        ytr = y.iloc[:train_end]
        if ytr.nunique() < 2:
            continue
        part = pd.DataFrame(index=X.index[start:end])
        for name, make in factories.items():
            m = make()
            m.fit(X.iloc[:train_end], ytr)
            part[name] = proba_up(m, X.iloc[start:end])
        part["y"] = y.iloc[start:end].values
        parts.append(part)
    if not parts:
        return None
    return pd.concat(parts).sort_index()


def oos_metrics(oos: pd.DataFrame, names: list, horizon: int) -> dict:
    y = oos["y"].astype(int).values
    ens = oos[names].mean(axis=1).values
    out = {
        "n_oos": int(len(y)),
        "eff_n": len(y) / horizon,
        "base_up": float(y.mean()),
        "auc": np.nan, "brier": np.nan, "brier_base": np.nan,
        "acc": float(((ens > 0.5) == y).mean()),
        "hit_bull": np.nan, "n_bull": 0, "per_model_auc": {},
    }
    if len(np.unique(y)) < 2:
        return out
    out["auc"] = float(roc_auc_score(y, ens))
    out["brier"] = float(brier_score_loss(y, ens))
    out["brier_base"] = float(brier_score_loss(y, np.full(len(y), y.mean())))
    bull = ens >= 0.55
    out["n_bull"] = int(bull.sum())
    if bull.sum() >= 10:
        out["hit_bull"] = float(y[bull].mean())
    for n in names:
        out["per_model_auc"][n] = float(roc_auc_score(y, oos[n].values))
    return out


# ----------------------------------------------------------------------------
# Analyse (zwaar -> gecached) en classificatie (licht)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def analyze_ticker(ticker: str, period: str, n_blocks: int) -> dict:
    df = load_prices(ticker, period)
    feats = make_features(df).dropna()
    if len(feats) < 400:
        raise ValueError("Te weinig bruikbare rijen na berekenen van features")
    close = df["Close"]
    x_latest = feats.iloc[[-1]]

    rows = []

    # --- Korte termijn (1 week) met walk-forward validatie ---
    y = make_target(close, H_SHORT).reindex(feats.index)
    labeled = y.notna()
    Xl, yl = feats[labeled], y[labeled].astype(int)
    if yl.nunique() < 2:
        raise ValueError("Doelvariabele heeft maar 1 klasse")

    factories = short_models()
    oos = walk_forward(Xl, yl, H_SHORT, factories, n_blocks)
    metrics = oos_metrics(oos, list(factories), H_SHORT) if oos is not None else None

    probs = []
    for name, make in factories.items():
        m = make()
        m.fit(Xl, yl)
        p = float(proba_up(m, x_latest)[0])
        probs.append(p)
        auc = metrics["per_model_auc"].get(name, np.nan) if metrics else np.nan
        rows.append({"Model": name, "Horizon": "1 week", "p": p,
                     "OOS AUC": auc, "Gevalideerd": metrics is not None})
    p_ens = float(np.mean(probs))
    agree = float(np.mean([p > 0.5 for p in probs]))

    # --- Lange termijn: alleen indicatief (niet statistisch valideerbaar) ---
    long_p = {}
    for label, hz in H_LONG.items():
        yy = make_target(close, hz).reindex(feats.index)
        lab = yy.notna()
        if lab.sum() < 300 or yy[lab].nunique() < 2:
            continue
        ps = []
        for name, make in long_models().items():
            m = make()
            m.fit(feats[lab], yy[lab].astype(int))
            p = float(proba_up(m, x_latest)[0])
            ps.append(p)
            rows.append({"Model": name, "Horizon": f"{label} (indicatief)", "p": p,
                         "OOS AUC": np.nan, "Gevalideerd": False})
        long_p[label] = float(np.mean(ps))

    return {
        "ticker": ticker,
        "last_date": str(feats.index[-1].date()),
        "table": pd.DataFrame(rows),
        "p": p_ens,
        "agree": agree,
        "metrics": metrics,
        "long_p": long_p,
        "n_train": int(len(Xl)),
    }


def classify(res: dict, p_buy: float, p_strong: float, min_agree: float, min_auc: float) -> dict:
    m, p, agree = res["metrics"], res["p"], res["agree"]
    p_strong = max(p_strong, p_buy)

    reasons = []
    if m is None:
        reasons.append("te weinig data voor walk-forward validatie")
    else:
        if m["n_oos"] < MIN_OOS:
            reasons.append(f"slechts {m['n_oos']} out-of-sample dagen (min. {MIN_OOS})")
        if m["eff_n"] < MIN_EFF_N:
            reasons.append(f"effectief ~{m['eff_n']:.0f} onafhankelijke metingen (min. {MIN_EFF_N})")
        if not (m["auc"] >= min_auc):
            reasons.append(f"OOS AUC {m['auc']:.2f} onder drempel {min_auc:.2f}")
        if not (m["brier"] < m["brier_base"]):
            reasons.append("Brier-score niet beter dan de naieve basislijn")
    validated = not reasons

    if p >= p_strong and agree >= 1.0:
        raw = "STRONG BUY"
    elif p >= p_buy and agree >= min_agree:
        raw = "BUY"
    elif p <= 0.45 and agree <= 0.25:
        raw = "BEARISH"
    else:
        raw = "NEUTRAAL"

    return {"signal": raw if validated else "ONVOLDOENDE BEWIJS",
            "raw": raw, "validated": validated, "reasons": reasons}


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def style_cell(val):
    colors = {
        "STRONG BUY": "background-color:#155724;color:white;font-weight:bold;",
        "BUY": "background-color:#28a745;color:white;font-weight:bold;",
        "NEUTRAAL": "background-color:#e2e3e5;color:#383d41;font-weight:bold;",
        "BEARISH": "background-color:#f8d7da;color:#721c24;font-weight:bold;",
        "ONVOLDOENDE BEWIJS": "background-color:#fff3cd;color:#856404;font-weight:bold;",
    }
    return colors.get(val, "")


GREEN = "background-color:#28a745;color:white;font-weight:bold;"


def style_green_buy(val):
    """Kolom 'Ruwe neiging': groen bij BUY of STRONG BUY."""
    return GREEN if val in ("BUY", "STRONG BUY") else ""


def style_green_bullish(val):
    """Kolom 'Modellen bullish (%)': groen bij alles boven 70."""
    return GREEN if pd.notna(val) and val > 70 else ""


def style_map(styler, func, subset):
    """Styler.map (pandas >= 2.1) of Styler.applymap (oudere pandas)."""
    return (styler.map if hasattr(styler, "map") else styler.applymap)(func, subset=subset)


def model_table(res: dict) -> pd.DataFrame:
    t = res["table"].copy()
    t["Stijgingskans"] = (t["p"] * 100).map("{:.1f}%".format)
    t["Richting"] = np.where(t["p"] > 0.5, "BULLISH", "BEARISH")
    t["OOS AUC"] = t["OOS AUC"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    return t[["Model", "Horizon", "Stijgingskans", "Richting", "OOS AUC"]]


def style_direction(row):
    if row["Richting"] == "BULLISH":
        return ["background-color:#d4edda;color:#155724;"] * len(row)
    return ["background-color:#f8d7da;color:#721c24;"] * len(row)


def show_single(res: dict, params: dict):
    cls = classify(res, **params)
    m = res["metrics"]
    st.subheader(f"Analyse voor {res['ticker']}  (laatste koers: {res['last_date']})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kans stijging (1 week)", f"{res['p'] * 100:.1f}%")
    c2.metric("Modellen bullish", f"{res['agree'] * 100:.0f}%")
    c3.metric("OOS AUC (0.50 = toeval)", f"{m['auc']:.2f}" if m else "n.v.t.")
    c4.metric("Basiskans omhoog", f"{m['base_up'] * 100:.0f}%" if m else "n.v.t.")

    sig = cls["signal"]
    if sig in ("STRONG BUY", "BUY"):
        st.success(f"**Signaal: {sig}** - gevalideerd: de modellen waren in de walk-forward test beter dan de basislijn.")
    elif sig == "BEARISH":
        st.error("**Signaal: BEARISH** - lage stijgingskans en gevalideerd model.")
    elif sig == "NEUTRAAL":
        st.info("**Signaal: NEUTRAAL** - geen duidelijke richting (gevalideerd model).")
    else:
        st.warning(
            f"**Geen betrouwbaar signaal.** Ruwe modelneiging: {cls['raw']}, maar het model is niet bewezen beter dan toeval.\n\n"
            + "Reden: " + "; ".join(cls["reasons"]) + ".")

    if m:
        hit = f"{m['hit_bull'] * 100:.0f}% (n={m['n_bull']})" if not pd.isna(m["hit_bull"]) else "te weinig signalen"
        st.caption(
            f"Validatie: {m['n_oos']} out-of-sample dagen, accuracy {m['acc'] * 100:.0f}%, "
            f"Brier {m['brier']:.3f} vs basislijn {m['brier_base']:.3f}. "
            f"Hitrate bij kans >= 55%: {hit} vs basiskans {m['base_up'] * 100:.0f}%.")

    lp = res["long_p"]
    if lp:
        txt = ", ".join(f"{k}: {v * 100:.0f}%" for k, v in lp.items())
        st.caption(f"Lange termijn (indicatief, niet valideerbaar): {txt}.")
        if all(v < 0.5 for v in lp.values()) and sig in ("STRONG BUY", "BUY"):
            st.warning("Let op: de lange-termijnmodellen wijzen omlaag.")

    st.write("### Modeloverzicht")
    st.dataframe(model_table(res).style.apply(style_direction, axis=1), use_container_width=True)


def main():
    st.set_page_config(page_title="AI Stock Analyzer v2", layout="wide")
    st.title("AI & ML Trading Analyzer v2")
    st.caption("Signalen worden alleen als betrouwbaar getoond na een out-of-sample validatie. Geen financieel advies.")

    st.sidebar.header("Instellingen")
    period = st.sidebar.selectbox("Koershistorie", ["5y", "10y", "max"], index=1)
    n_blocks = st.sidebar.slider("Validatieblokken (21 dagen elk)", 5, 12, 8)
    with st.sidebar.expander("Signaaldrempels"):
        p_buy = st.slider("BUY vanaf kans", 0.50, 0.70, 0.55, 0.01)
        p_strong = st.slider("STRONG BUY vanaf kans", 0.55, 0.80, 0.62, 0.01)
        min_agree = st.slider("Min. aandeel modellen bullish (BUY)", 0.50, 1.00, 0.75, 0.25)
        min_auc = st.slider("Min. OOS AUC voor 'gevalideerd'", 0.50, 0.65, 0.55, 0.01)
    params = dict(p_buy=p_buy, p_strong=p_strong, min_agree=min_agree, min_auc=min_auc)

    tab1, tab2 = st.tabs(["Enkele analyse", "Batch"])

    with tab1:
        ticker = st.text_input("Ticker:", value="NVDA").strip().upper()
        if st.button("Start analyse"):
            with st.spinner(f"Data ophalen en modellen valideren voor {ticker}..."):
                try:
                    st.session_state["single"] = analyze_ticker(ticker, period, n_blocks)
                except Exception as e:
                    st.session_state["single"] = {"error": str(e), "ticker": ticker}
        res = st.session_state.get("single")
        if res:
            if "error" in res:
                st.error(f"Fout bij {res['ticker']}: {res['error']}")
            else:
                show_single(res, params)

    with tab2:
        st.caption("Tickers gescheiden door komma, spatie of nieuwe regel (max. 510). "
                   "Elke ticker duurt ca. 5-20 seconden; resultaten worden een uur gecached.")
        batch_input = st.text_area("Tickers:", value="NVDA, AAPL, MSFT, AMZN, GOOGL, TSLA, META", height=100)
        if st.button("Start batch analyse"):
            raw = batch_input.replace(",", " ").split()
            tickers = list(dict.fromkeys(t.strip().upper() for t in raw if t.strip()))
            if len(tickers) > 510:
                st.warning(f"{len(tickers)} tickers ingevoerd; ingekort tot 510.")
                tickers = tickers[:510]
            if not tickers:
                st.error("Geen geldige tickers ingevoerd.")
            else:
                results, bar, status = {}, st.progress(0.0), st.empty()
                for i, t in enumerate(tickers):
                    status.text(f"Verwerken: {t} ({i + 1}/{len(tickers)})")
                    try:
                        results[t] = analyze_ticker(t, period, n_blocks)
                    except Exception as e:
                        results[t] = {"error": str(e)[:200], "ticker": t}
                    bar.progress((i + 1) / len(tickers))
                    time.sleep(0.2)  # vriendelijk voor Yahoo
                status.text("Batch afgerond.")
                st.session_state["batch"] = results

        batch = st.session_state.get("batch")
        if batch:
            rows = []
            for t, r in batch.items():
                if "error" in r:
                    rows.append({"Ticker": t, "Signaal": "FOUT", "Ruwe neiging": "-",
                                 "Kans (%)": np.nan, "Modellen bullish (%)": np.nan,
                                 "OOS AUC": np.nan, "Gevalideerd": False, "Opmerking": r["error"]})
                    continue
                c = classify(r, **params)
                m = r["metrics"]
                rows.append({"Ticker": t, "Signaal": c["signal"], "Ruwe neiging": c["raw"],
                             "Kans (%)": r["p"] * 100, "Modellen bullish (%)": r["agree"] * 100,
                             "OOS AUC": m["auc"] if m else np.nan,
                             "Gevalideerd": c["validated"], "Opmerking": "; ".join(c["reasons"])})
            sdf = pd.DataFrame(rows).sort_values(["Gevalideerd", "Kans (%)"], ascending=[False, False])
            st.subheader("Batchoverzicht")
            styler = sdf.style
            styler = style_map(styler, style_cell, ["Signaal"])
            styler = style_map(styler, style_green_buy, ["Ruwe neiging"])
            styler = style_map(styler, style_green_bullish, ["Modellen bullish (%)"])
            styler = styler.format({"Kans (%)": "{:.1f}", "Modellen bullish (%)": "{:.0f}", "OOS AUC": "{:.2f}"}, na_rep="-")
            st.dataframe(styler, use_container_width=True)
            st.download_button("Download als CSV", sdf.to_csv(index=False).encode("utf-8"),
                               file_name="batch_resultaten.csv", mime="text/csv")

            st.subheader("Details per aandeel (eerste 100)")
            shown = [t for t in sdf["Ticker"] if "error" not in batch[t]][:100]
            for t in shown:
                with st.expander(t):
                    st.dataframe(model_table(batch[t]).style.apply(style_direction, axis=1),
                                 use_container_width=True)


if __name__ == "__main__":
    main()
