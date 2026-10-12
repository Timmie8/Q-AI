import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Probeer Plotly te importeren
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Probeer PyTorch te importeren
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Stock ML Predictor - Fast", layout="wide")
st.title("📈 Stock ML Predictor - Fast Mode")
st.markdown("**Snelle analyse met 7 modellen** | **LSTM optioneel**")

if not PLOTLY_AVAILABLE:
    st.warning("⚠️ Plotly niet beschikbaar. Voeg 'plotly' toe aan requirements.txt.")

if not PYTORCH_AVAILABLE:
    st.warning("⚠️ PyTorch niet beschikbaar. LSTM wordt overgeslagen.")

# Sidebar configuratie
with st.sidebar:
    st.header("⚙️ Configuratie")
    
    # ✅ MINDER TICKERS VOOR SNELLERE START
    default_tickers = "AAPL,MSFT,GOOGL,NVDA,TSLA"
    ticker_input = st.text_area(
        "Voer tickers in (max 10 aanbevolen):",
        value=default_tickers,
        height=100
    )
    
    # ✅ KORTERE PERIODE
    period_options = {
        "6 maanden": "6mo",
        "1 jaar": "1y",
        "2 jaar": "2y"
    }
    selected_period = st.selectbox("Selecteer periode:", list(period_options.keys()), index=0)
    period = period_options[selected_period]
    
    # ✅ LSTM OPTIONEEL (standaard uit voor snelheid)
    use_lstm = st.checkbox("Gebruik LSTM (langzamer)", value=False)
    
    # ✅ MINDER COMPLEXE MODELLEN
    fast_mode = st.checkbox("Fast Mode (minder estimators)", value=True)
    
    start_button = st.button("🚀 Start Analyse", type="primary", use_container_width=True)

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def create_features(df_close, df_volume):
    df = pd.DataFrame({'Close': df_close, 'Volume': df_volume})
    
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_3d'] = df['Close'].pct_change(3)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    df['Volatility_10'] = df['Return_1d'].rolling(window=10).std()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df.dropna()

# ==========================================
# LSTM MODEL (optioneel)
# ==========================================
if PYTORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=16, num_layers=1):  # ✅ KLEINER MODEL
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return self.sigmoid(out)

    def create_sequences(X, y, seq_length=5):  # ✅ KORTERE SEQUENTIE
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X.iloc[i:i+seq_length].values)
            y_seq.append(y.iloc[i+seq_length])
        return np.array(X_seq), np.array(y_seq)

# ==========================================
# MODEL TRAINING (GEOPIMALISEERD)
# ==========================================
def train_all_models(df, ticker, use_lstm=False, fast_mode=True):
    feature_cols = ['Return_1d', 'Return_3d', 'Return_5d', 'SMA_5', 'SMA_10', 
                    'SMA_20', 'Volatility_10', 'RSI_14']
    
    if len(df) < 100:
        return None, None, f"Niet genoeg data ({len(df)} rijen)"
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df['Target']
    X_test, y_test = test_df[feature_cols], test_df['Target']
    
    # ✅ MINDER ESTIMATORS VOOR SNELLERE TRAINING
    n_estimators = 50 if fast_mode else 100
    
    model_weights = {
        "Logistic Regression": 1,
        "SVM": 1,
        "Random Forest": 1,
        "Gradient Boosting": 1,
        "XGBoost": 3,
        "LightGBM": 3,
        "MLP": 1,
        "LSTM": 2
    }
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),  # ✅ MINDER ITERATIES
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=0.1, random_state=42, 
                                     eval_metric='logloss', verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.1, random_state=42, 
                                       verbose=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)  # ✅ KLEINER NETWERK
    }
    
    results = []
    model_predictions = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            results.append({
                "Ticker": ticker,
                "Model": name,
                "Weight": model_weights[name],
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4)
            })
            
            model_predictions[name] = {
                'pred': y_pred,
                'prob': y_prob,
                'weight': model_weights[name]
            }
            
        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Model": name,
                "Weight": model_weights[name],
                "Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0
            })
    
    # LSTM Model (optioneel)
    if use_lstm and PYTORCH_AVAILABLE:
        try:
            seq_length = 5
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
            
            if len(X_train_seq) >= 30:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(
                    X_train_seq.reshape(-1, X_train_seq.shape[-1])
                ).reshape(X_train_seq.shape)
                X_test_scaled = scaler.transform(
                    X_test_seq.reshape(-1, X_test_seq.shape[-1])
                ).reshape(X_test_seq.shape)
                
                X_train_t = torch.FloatTensor(X_train_scaled)
                y_train_t = torch.FloatTensor(y_train_seq)
                X_test_t = torch.FloatTensor(X_test_scaled)
                
                train_dataset = TensorDataset(X_train_t, y_train_t)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
                
                lstm_model = LSTMModel(input_size=len(feature_cols))
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
                
                lstm_model.train()
                for epoch in range(5):  # ✅ MINDER EPOCHS
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = lstm_model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                lstm_model.eval()
                with torch.no_grad():
                    lstm_probs = lstm_model(X_test_t).squeeze().numpy()
                    lstm_preds = (lstm_probs > 0.5).astype(int)
                
                results.append({
                    "Ticker": ticker,
                    "Model": "LSTM",
                    "Weight": model_weights["LSTM"],
                    "Accuracy": round(accuracy_score(y_test_seq, lstm_preds), 4),
                    "Precision": round(precision_score(y_test_seq, lstm_preds, zero_division=0), 4),
                    "Recall": round(recall_score(y_test_seq, lstm_preds, zero_division=0), 4),
                    "F1-Score": round(f1_score(y_test_seq, lstm_preds, zero_division=0), 4)
                })
                
                model_predictions["LSTM"] = {
                    'pred': lstm_preds,
                    'prob': lstm_probs,
                    'weight': model_weights["LSTM"]
                }
        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Model": "LSTM",
                "Weight": 2,
                "Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0
            })
    
    # Signal Logic
    signal = "HOLD / SELL"
    signal_color = ""
    signal_desc = "Geen sterke bullish signalen."
    
    xgb_latest = model_predictions.get("XGBoost", {}).get('pred', [0])[-1]
    lgb_latest = model_predictions.get("LightGBM", {}).get('pred', [0])[-1]
    lstm_latest = model_predictions.get("LSTM", {}).get('pred', [0])[-1] if "LSTM" in model_predictions else 0
    
    xgb_bullish = xgb_latest == 1
    lgb_bullish = lgb_latest == 1
    lstm_bullish = lstm_latest == 1
    
    if (xgb_bullish or lgb_bullish) and lstm_bullish:
        signal = "STRONG BUY"
        signal_color = ""
        signal_desc = "XGBoost/LightGBM én LSTM zijn bullish."
    elif xgb_bullish or lgb_bullish:
        signal = "BUY"
        signal_color = "🔵"
        signal_desc = "XGBoost of LightGBM is bullish."
    
    ensemble_prediction = {
        'signal': signal, 
        'signal_color': signal_color, 
        'signal_desc': signal_desc
    }
    
    return results, ensemble_prediction, None

# ==========================================
# KLEURCODERING
# ==========================================
def highlight_weights(row):
    if row['Model'] in ['XGBoost', 'LightGBM']:
        return ['background-color: #28a745; color: white; font-weight: bold'] * len(row)
    elif row['Model'] == 'LSTM':
        return ['background-color: #17a2b8; color: white; font-weight: bold'] * len(row)
    return [''] * len(row)

def highlight_signals(row):
    if "STRONG BUY" in row['Signaal']:
        return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row)
    elif "BUY" in row['Signaal']:
        return ['background-color: #cce5ff; color: #004085; font-weight: bold'] * len(row)
    return [''] * len(row)

# ==========================================
# HOOFDLOGICA MET BETERE FEEDBACK
# ==========================================
if start_button:
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    
    # ✅ WAARSCHUWING BIJ TE VEEL TICKERS
    if len(tickers) > 10:
        st.warning(f"⚠️ Je hebt {len(tickers)} tickers ingevoerd. Dit kan lang duren. Max 10 aanbevolen voor snelle resultaten.")
    
    if not tickers:
        st.error("❌ Geen tickers ingevoerd!")
        st.stop()
    
    # ✅ SPINNER VOOR FEEDBACK
    with st.spinner(f"📊 Analyseren van {len(tickers)} tickers... Dit kan 1-3 minuten duren."):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📥 Data ophalen...")
        try:
            all_data = yf.download(tickers, period=period, progress=False, threads=True)
            
            if isinstance(all_data.columns, pd.MultiIndex):
                close_data = all_data['Close']
                volume_data = all_data['Volume']
            else:
                close_data = all_data[['Close']]
                volume_data = all_data[['Volume']]
            
        except Exception as e:
            st.error(f"❌ Fout bij ophalen data: {e}")
            st.stop()
        
        ticker_data = {}
        
        for idx, ticker in enumerate(tickers, 1):
            status_text.text(f"[{idx}/{len(tickers)}] Verwerken {ticker}...")
            progress_bar.progress(idx / len(tickers))
            
            try:
                if isinstance(close_data, pd.DataFrame) and ticker in close_data.columns:
                    df_close = close_data[ticker].dropna()
                    df_volume = volume_data[ticker].dropna() if ticker in volume_data.columns else pd.Series(0, index=df_close.index)
                else:
                    continue
                
                df = create_features(df_close, df_volume)
                results, ensemble_pred, error = train_all_models(df, ticker, use_lstm=use_lstm, fast_mode=fast_mode)
                
                if not error and results:
                    ticker_data[ticker] = {
                        'results': results,
                        'ensemble': ensemble_pred,
                        'close_data': df_close,
                        'df': df
                    }
                    
            except Exception as e:
                continue
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analyse voltooid!")
    
    # ==========================================
    # OVERZICHTSPAGINA
    # ==========================================
    if ticker_data:
        st.success(f"✅ Succesvol verwerkt: {len(ticker_data)} tickers")
        st.markdown("---")
        
        overview_data = []
        for ticker, data in ticker_data.items():
            ens = data['ensemble']
            results_df = pd.DataFrame(data['results'])
            best_model_row = results_df.loc[results_df['F1-Score'].idxmax()]
            
            overview_data.append({
                'Ticker': ticker,
                'Signaal': f"{ens['signal_color']} {ens['signal']}",
                'Ensemble F1': f"{results_df[results_df['Model']!='LSTM']['F1-Score'].mean():.4f}",
                'Beste Model': best_model_row['Model'],
                'Beste F1': f"{best_model_row['F1-Score']:.4f}"
            })
        
        overview_df = pd.DataFrame(overview_data)
        
        def signal_sort_key(row):
            if "STRONG BUY" in row['Signaal']: return 0
            if "BUY" in row['Signaal']: return 1
            return 2
        
        overview_df['Sort_Key'] = overview_df.apply(signal_sort_key, axis=1)
        overview_df = overview_df.sort_values(['Sort_Key', 'Ticker']).drop('Sort_Key', axis=1)
        
        st.subheader("📋 Overzicht Alle Tickers & Signalen")
        
        st.markdown("""
        **Legenda:**
        - 🟢 **Groen** = XGBoost & LightGBM (hoofd BUY-signalen, wegen 3x)
        - 🔵 **Blauw** = LSTM (patroonherkenner, bevestigt Strong Buy)
        - ⚪ **Wit** = Overige ondersteunende modellen
        """)
        
        styled_overview = overview_df.style.apply(highlight_signals, axis=1)
        st.dataframe(styled_overview, use_container_width=True, height=400)
        
        # ==========================================
        # TICKER SELECTIE
        # ==========================================
        st.markdown("---")
        selected_ticker = st.selectbox(
            "🎯 Selecteer een ticker voor gedetailleerde analyse:",
            options=list(ticker_data.keys()),
            index=0,
            key="ticker_selector"
        )
        
        if selected_ticker in ticker_data:
            data = ticker_data[selected_ticker]
            results_df = pd.DataFrame(data['results'])
            ens = data['ensemble']
            
            st.markdown("---")
            if ens['signal'] == "STRONG BUY":
                st.markdown(f"<h1 style='color: green; text-align: center;'>{ens['signal_color']} {ens['signal']} ({selected_ticker})</h1>", unsafe_allow_html=True)
            elif ens['signal'] == "BUY":
                st.markdown(f"<h1 style='color: #004085; text-align: center;'>{ens['signal_color']} {ens['signal']} ({selected_ticker})</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='color: #721c24; text-align: center;'>{ens['signal_color']} {ens['signal']} ({selected_ticker})</h1>", unsafe_allow_html=True)
            
            st.info(f"**Waarom?** {ens['signal_desc']}")
            
            st.markdown("---")
            st.subheader(f"📊 Alle Modellen voor {selected_ticker}")
            
            results_df_sorted = results_df.sort_values('F1-Score', ascending=False)
            st.dataframe(results_df_sorted.style.apply(highlight_weights, axis=1), use_container_width=True, height=400)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 F1-Score per Model")
                if PLOTLY_AVAILABLE:
                    fig = px.bar(
                        results_df_sorted, 
                        x='F1-Score', 
                        y='Model', 
                        orientation='h',
                        color='Weight',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key=f"f1_chart_{selected_ticker}")
                else:
                    chart_data = results_df_sorted.set_index('Model')[['F1-Score']]
                    st.bar_chart(chart_data)
            
            with col2:
                st.subheader("⚖️ Model Weights")
                weights_df = results_df_sorted[['Model', 'Weight']].drop_duplicates()
                if PLOTLY_AVAILABLE:
                    fig = px.pie(
                        weights_df, 
                        values='Weight', 
                        names='Model',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key=f"weight_chart_{selected_ticker}")
                else:
                    st.dataframe(weights_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader(f"📈 {selected_ticker} Koersverloop")
            
            close_data = data['close_data']
            
            if isinstance(close_data, pd.Series):
                close_df = close_data.to_frame(name='Close')
            else:
                close_df = close_data
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=close_df.index,
                    y=close_df['Close'].values,
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#00d4aa', width=2)
                ))
                fig.update_layout(
                    title=f"{selected_ticker} - Laatste {len(close_df)} dagen",
                    xaxis_title="Datum",
                    yaxis_title="Prijs ($)",
                    height=400,
                    hovermode='x unified',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{selected_ticker}")
            else:
                st.line_chart(close_df, use_container_width=True)
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label=f"📥 Download resultaten voor {selected_ticker}",
                data=csv,
                file_name=f"{selected_ticker}_predictions_{timestamp}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("⚠️ Geen resultaten om weer te geven.")
