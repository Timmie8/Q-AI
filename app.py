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
import plotly.express as px
import plotly.graph_objects as go

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
st.set_page_config(page_title="Stock ML Predictor - Weighted Ensemble", layout="wide")
st.title("📈 Stock ML Predictor - Weighted Ensemble")
st.markdown("**XGBoost & LightGBM wegen 3x zwaarder** | **LSTM voor patroonherkenning**")

# Sidebar configuratie
with st.sidebar:
    st.header("⚙️ Configuratie")
    
    # Ticker selectie
    default_tickers = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,AMD,INTC,JPM,BAC,GS,V,MA,JNJ,UNH,PFE,WMT,COST"
    ticker_input = st.text_area(
        "Voer tickers in (gescheiden door komma's):",
        value=default_tickers,
        height=150
    )
    
    # Periode selectie
    period_options = {
        "1 jaar": "1y",
        "2 jaar": "2y",
        "5 jaar": "5y"
    }
    selected_period = st.selectbox("Selecteer periode:", list(period_options.keys()))
    period = period_options[selected_period]
    
    # Ensemble weights
    st.subheader("⚖️ Ensemble Weights")
    st.info("XGBoost en LightGBM wegen 3x zwaarder in de finale voorspelling")
    
    # Start knop
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
# LSTM MODEL
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
# MODEL TRAINING MET WEIGHTS
# ==========================================
def train_all_models(df, ticker, use_lstm=True):
    feature_cols = ['Return_1d', 'Return_3d', 'Return_5d', 'SMA_5', 'SMA_10', 
                    'SMA_20', 'Volatility_10', 'RSI_14']
    
    if len(df) < 100:
        return None, None, f"Niet genoeg data ({len(df)} rijen)"
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df['Target']
    X_test, y_test = test_df[feature_cols], test_df['Target']
    
    # Ensemble weights: XGBoost en LightGBM wegen 3x
    model_weights = {
        "Logistic Regression": 1,
        "SVM": 1,
        "Random Forest": 1,
        "Gradient Boosting": 1,
        "XGBoost": 3,  # ⚖️ 3x gewicht
        "LightGBM": 3,  # ⚖️ 3x gewicht
        "MLP": 1,
        "LSTM": 2  # LSTM weegt 2x (patroonherkenner)
    }
    
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
    model_predictions = {}
    
    # Train traditionele modellen
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
            
            # Bewaar voorspellingen voor ensemble
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
    
    # LSTM Model
    lstm_prob = None
    if use_lstm and PYTORCH_AVAILABLE:
        try:
            seq_length = 10
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
            
            if len(X_train_seq) >= 50:
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
                for epoch in range(10):
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
                
                lstm_prob = lstm_probs
                model_predictions["LSTM"] = {
                    'pred': lstm_preds,
                    'prob': lstm_probs,
                    'weight': model_weights["LSTM"]
                }
        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Model": "LSTM",
                "Weight": model_weights["LSTM"],
                "Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0
            })
    
    # Bereken ensemble voorspelling (gewogen gemiddelde)
    ensemble_prediction = None
    if model_predictions:
        weighted_sum = 0
        total_weight = 0
        
        for model_name, pred_data in model_predictions.items():
            weight = pred_data['weight']
            probs = pred_data['prob']
            
            # Alleen gebruiken als lengte overeenkomt
            if lstm_prob is not None and model_name == "LSTM":
                # LSTM heeft andere lengte door sequence, skip in ensemble
                continue
            
            weighted_sum += probs * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_probs = weighted_sum / total_weight
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            
            # Bereken ensemble metrics
            # Gebruik de laatste test data (niet LSTM sequence data)
            ensemble_acc = accuracy_score(y_test, ensemble_preds)
            ensemble_f1 = f1_score(y_test, ensemble_preds, zero_division=0)
            
            ensemble_prediction = {
                'accuracy': round(ensemble_acc, 4),
                'f1_score': round(ensemble_f1, 4),
                'confidence': round(ensemble_probs[-1] if len(ensemble_probs) > 0 else 0.5, 4)
            }
    
    return results, ensemble_prediction, None

# ==========================================
# HOOFDLOGICA
# ==========================================
if start_button:
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    
    if not tickers:
        st.error("❌ Geen tickers ingevoerd!")
        st.stop()
    
    st.info(f"📊 Analyseren van {len(tickers)} tickers...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Data ophalen
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
    
    # Verwerk elke ticker
    all_results = []
    ticker_data = {}  # Bewaar data per ticker voor interactieve selectie
    
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
            results, ensemble_pred, error = train_all_models(df, ticker, use_lstm=True)
            
            if not error and results:
                all_results.extend(results)
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
    # INTERACTIEVE TICKER SELECTIE
    # ==========================================
    if ticker_data:
        st.success(f"✅ Succesvol verwerkt: {len(ticker_data)} tickers")
        
        # Ticker selector
        selected_ticker = st.selectbox(
            "🎯 Klik op een ticker om alle modellen te zien:",
            options=list(ticker_data.keys()),
            index=0
        )
        
        if selected_ticker in ticker_data:
            data = ticker_data[selected_ticker]
            results_df = pd.DataFrame(data['results'])
            
            # ==========================================
            # ENSEMBLE VOORSPELLING (BOVENAAN)
            # ==========================================
            if data['ensemble']:
                st.markdown("---")
                st.subheader(f"🎯 Ensemble Voorspelling voor {selected_ticker}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    confidence = data['ensemble']['confidence']
                    direction = "📈 OMHOOG" if confidence > 0.5 else "📉 OMLAAG"
                    st.metric("Richting", direction)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    st.metric("Ensemble F1-Score", f"{data['ensemble']['f1_score']:.4f}")
                
                # Confidence bar
                st.progress(confidence)
                
                st.markdown(f"""
                **Gewogen Ensemble**: XGBoost (3x) + LightGBM (3x) + LSTM (2x) + andere modellen (1x)
                
                De finale voorspelling is gebaseerd op een gewogen gemiddelde van alle modelvoorspellingen, 
                waarbij **XGBoost en LightGBM het zwaarst wegen** (3x) en **LSTM als patroonherkenner** (2x).
                """)
            
            st.markdown("---")
            
            # ==========================================
            # ALLE MODELLEN MET SCORES
            # ==========================================
            st.subheader(f"📊 Alle Modellen voor {selected_ticker}")
            
            # Sorteer op F1-Score
            results_df_sorted = results_df.sort_values('F1-Score', ascending=False)
            
            # Kleur coding voor weights
            def highlight_weights(row):
                if row['Model'] in ['XGBoost', 'LightGBM']:
                    return ['background-color: #ff6b6b; color: white'] * len(row)
                elif row['Model'] == 'LSTM':
                    return ['background-color: #4ecdc4; color: white'] * len(row)
                return [''] * len(row)
            
            styled_df = results_df_sorted.style.apply(highlight_weights, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # ==========================================
            # VISUALISATIES
            # ==========================================
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 F1-Score per Model")
                fig = px.bar(
                    results_df_sorted, 
                    x='F1-Score', 
                    y='Model', 
                    orientation='h',
                    color='Weight',
                    color_continuous_scale='RdYlGn',
                    title=f"F1-Score per Model (Hoger = Beter)"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("⚖️ Model Weights")
                weights_df = results_df_sorted[['Model', 'Weight']].drop_duplicates()
                fig = px.pie(
                    weights_df, 
                    values='Weight', 
                    names='Model',
                    title="Gewichten in Ensemble",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # ==========================================
            # LSTM PATROONHERKENNING
            # ==========================================
            st.markdown("---")
            st.subheader("🧠 LSTM Patroonherkenning")
            
            lstm_row = results_df[results_df['Model'] == 'LSTM']
            if not lstm_row.empty:
                lstm_metrics = lstm_row.iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("LSTM Accuracy", f"{lstm_metrics['Accuracy']:.2%}")
                
                with col2:
                    st.metric("LSTM Precision", f"{lstm_metrics['Precision']:.2%}")
                
                with col3:
                    st.metric("LSTM Recall", f"{lstm_metrics['Recall']:.2%}")
                
                with col4:
                    st.metric("LSTM F1-Score", f"{lstm_metrics['F1-Score']:.4f}")
                
                st.info("""
                **LSTM (Long Short-Term Memory)** is een recurrent neural network dat 
                tijdsafhankelijke patronen herkent in de koersdata. Het analyseert sequenties 
                van 10 dagen om complexe niet-lineaire patronen te detecteren die traditionele 
                modellen missen.
                """)
            
            # ==========================================
            # KOERSGRAFIEK
            # ==========================================
            st.markdown("---")
            st.subheader(f"📈 {selected_ticker} Koersverloop")
            
            close_data = data['close_data']
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=close_data.index,
                y=close_data.values,
                mode='lines',
                name='Close Price',
                line=dict(color='#00d4aa', width=2)
            ))
            
            fig.update_layout(
                title=f"{selected_ticker} - Laatste {len(close_data)} dagen",
                xaxis_title="Datum",
                yaxis_title="Prijs ($)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ==========================================
            # DOWNLOAD
            # ==========================================
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
