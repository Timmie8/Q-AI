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
import time

# Probeer PyTorch te importeren (optioneel)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch niet beschikbaar. LSTM model wordt overgeslagen. Voeg 'torch' toe aan requirements.txt voor alle 8 modellen.")

warnings.filterwarnings('ignore')

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Stock ML Predictor", layout="wide")
st.title("📈 Stock ML Predictor - Multiple Tickers")
st.markdown("Train 8 machine learning modellen op meerdere aandelen tegelijk")

# Sidebar configuratie
with st.sidebar:
    st.header("⚙️ Configuratie")
    
    # Ticker selectie
    default_tickers = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,AMD,INTC,JPM"
    ticker_input = st.text_area(
        "Voer tickers in (gescheiden door komma's):",
        value=default_tickers,
        height=150
    )
    
    # Periode selectie
    period_options = {
        "1 jaar": "1y",
        "2 jaar": "2y",
        "5 jaar": "5y",
        "10 jaar": "10y"
    }
    selected_period = st.selectbox("Selecteer periode:", list(period_options.keys()))
    period = period_options[selected_period]
    
    # Model selectie
    st.subheader("Modellen")
    use_lstm = st.checkbox("Gebruik LSTM (PyTorch)", value=PYTORCH_AVAILABLE, disabled=not PYTORCH_AVAILABLE)
    
    # Start knop
    start_button = st.button("🚀 Start Analyse", type="primary", use_container_width=True)

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def create_features(df_close, df_volume):
    """Maak technische features voor een enkel aandeel"""
    df = pd.DataFrame({
        'Close': df_close,
        'Volume': df_volume
    })
    
    # Returns
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_3d'] = df['Close'].pct_change(3)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Volatility & RSI
    df['Volatility_10'] = df['Return_1d'].rolling(window=10).std()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Target: 1 als volgende dag hoger sluit
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df.dropna()

# ==========================================
# LSTM MODEL (optioneel)
# ==========================================
if PYTORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=2):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return self.sigmoid(out)

    def create_sequences(X, y, seq_length=10):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X.iloc[i:i+seq_length].values)
            y_seq.append(y.iloc[i+seq_length])
        return np.array(X_seq), np.array(y_seq)

# ==========================================
# MODEL TRAINING
# ==========================================
def train_all_models(df, ticker, use_lstm=True):
    """Train alle modellen voor één aandeel"""
    
    feature_cols = ['Return_1d', 'Return_3d', 'Return_5d', 'SMA_5', 'SMA_10', 
                    'SMA_20', 'Volatility_10', 'RSI_14']
    
    if len(df) < 100:
        return None, f"Niet genoeg data ({len(df)} rijen)"
    
    # Chronologische split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df['Target']
    X_test, y_test = test_df[feature_cols], test_df['Target']
    
    # Traditionele modellen
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, 
                                     eval_metric='logloss', verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, 
                                       verbose=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    results = []
    
    # Train traditionele modellen
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results.append({
                "Ticker": ticker,
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4)
            })
        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Model": name,
                "Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0
            })
    
    # LSTM Model (optioneel)
    if use_lstm and PYTORCH_AVAILABLE:
        try:
            seq_length = 10
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
            
            if len(X_train_seq) >= 50:
                # Normaliseer data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(
                    X_train_seq.reshape(-1, X_train_seq.shape[-1])
                ).reshape(X_train_seq.shape)
                X_test_scaled = scaler.transform(
                    X_test_seq.reshape(-1, X_test_seq.shape[-1])
                ).reshape(X_test_seq.shape)
                
                # PyTorch tensors
                X_train_t = torch.FloatTensor(X_train_scaled)
                y_train_t = torch.FloatTensor(y_train_seq)
                X_test_t = torch.FloatTensor(X_test_scaled)
                
                train_dataset = TensorDataset(X_train_t, y_train_t)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
                
                # Train LSTM
                lstm_model = LSTMModel(input_size=len(feature_cols))
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
                
                lstm_model.train()
                for epoch in range(10):
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = lstm_model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate
                lstm_model.eval()
                with torch.no_grad():
                    lstm_preds = (lstm_model(X_test_t).squeeze().numpy() > 0.5).astype(int)
                
                results.append({
                    "Ticker": ticker,
                    "Model": "LSTM",
                    "Accuracy": round(accuracy_score(y_test_seq, lstm_preds), 4),
                    "Precision": round(precision_score(y_test_seq, lstm_preds, zero_division=0), 4),
                    "Recall": round(recall_score(y_test_seq, lstm_preds, zero_division=0), 4),
                    "F1-Score": round(f1_score(y_test_seq, lstm_preds, zero_division=0), 4)
                })
        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Model": "LSTM",
                "Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0
            })
    
    return results, None

# ==========================================
# HOOFDLOGICA
# ==========================================
if start_button:
    # Parse tickers
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    
    if not tickers:
        st.error("❌ Geen tickers ingevoerd!")
        st.stop()
    
    st.info(f"📊 Analyseren van {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Data ophalen
    status_text.text("📥 Data ophalen...")
    try:
        all_data = yf.download(
            tickers, 
            period=period,
            progress=False,
            threads=True
        )
        
        if isinstance(all_data.columns, pd.MultiIndex):
            close_data = all_data['Close']
            volume_data = all_data['Volume']
        else:
            close_data = all_data[['Close']]
            volume_data = all_data[['Volume']]
        
    except Exception as e:
        st.error(f"❌ Fout bij ophalen data: {e}")
        st.stop()
    
    # Verwerk elke ticker
    all_results = []
    successful_tickers = 0
    failed_tickers = []
    
    for idx, ticker in enumerate(tickers, 1):
        status_text.text(f"[{idx}/{len(tickers)}] Verwerken {ticker}...")
        progress_bar.progress(idx / len(tickers))
        
        try:
            # Haal data op
            if isinstance(close_data, pd.DataFrame) and ticker in close_data.columns:
                df_close = close_data[ticker].dropna()
                df_volume = volume_data[ticker].dropna() if ticker in volume_data.columns else pd.Series(0, index=df_close.index)
            else:
                df_close = close_data['Close'].dropna()
                df_volume = volume_data['Volume'].dropna()
            
            # Maak features
            df = create_features(df_close, df_volume)
            
            # Train modellen
            results, error = train_all_models(df, ticker, use_lstm=use_lstm)
            
            if error:
                failed_tickers.append(f"{ticker}: {error}")
            else:
                all_results.extend(results)
                successful_tickers += 1
                
        except Exception as e:
            failed_tickers.append(f"{ticker}: {str(e)[:50]}")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Analyse voltooid!")
    
    # ==========================================
    # RESULTATEN TONEN
    # ==========================================
    st.success(f"✅ Succesvol verwerkt: {successful_tickers}/{len(tickers)} tickers")
    
    if failed_tickers:
        with st.expander("⚠️ Mislukte tickers"):
            for fail in failed_tickers:
                st.write(f"• {fail}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Tabs voor verschillende weergaven
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Gemiddelde Performance", 
            "🏆 Beste Model per Aandeel", 
            "🌟 Top Performers", 
            "📋 Alle Resultaten"
        ])
        
        with tab1:
            st.subheader("Gemiddelde Performance per Model")
            avg_results = results_df.groupby('Model').agg({
                'Accuracy': 'mean',
                'Precision': 'mean',
                'Recall': 'mean',
                'F1-Score': 'mean'
            }).round(4)
            st.dataframe(avg_results, use_container_width=True)
            
            # Visualisatie
            st.subheader("F1-Score per Model")
            st.bar_chart(avg_results['F1-Score'])
        
        with tab2:
            st.subheader("Beste Model per Aandeel (F1-Score)")
            best_per_ticker = results_df.loc[results_df.groupby('Ticker')['F1-Score'].idxmax()]
            st.dataframe(
                best_per_ticker[['Ticker', 'Model', 'Accuracy', 'F1-Score']], 
                use_container_width=True
            )
        
        with tab3:
            st.subheader("Top 10 Aandelen (hoogste gemiddelde F1-Score)")
            top_tickers = results_df.groupby('Ticker')['F1-Score'].mean().sort_values(ascending=False).head(10)
            
            top_data = []
            for ticker, score in top_tickers.items():
                best_model = results_df[results_df['Ticker'] == ticker].loc[
                    results_df[results_df['Ticker'] == ticker]['F1-Score'].idxmax()
                ]
                top_data.append({
                    'Ticker': ticker,
                    'Gemiddelde F1': round(score, 4),
                    'Beste Model': best_model['Model'],
                    'Beste F1': best_model['F1-Score']
                })
            
            st.dataframe(pd.DataFrame(top_data), use_container_width=True)
            st.bar_chart(top_tickers)
        
        with tab4:
            st.subheader("Alle Resultaten")
            st.dataframe(results_df, use_container_width=True)
            
            # Download knop
            csv = results_df.to_csv(index=False).encode('utf-8')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download als CSV",
                data=csv,
                file_name=f"stock_predictions_{timestamp}.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ Geen resultaten om weer te geven.")
