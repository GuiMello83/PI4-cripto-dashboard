import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math

st.set_page_config(page_title="IA Preditiva", layout="centered")

st.title("🧠 IA Preditiva com LSTM")
st.markdown("Este painel utiliza uma rede neural LSTM para prever preços de criptomoedas com base em séries temporais.")

# ----------------------------
# Seleção de moeda
# ----------------------------
coin = st.selectbox("Escolha a moeda:", ["bitcoin", "ethereum", "solana"])
symbol_map = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT"
}
symbol = symbol_map[coin]

# ----------------------------
# Requisição à API Binance
# ----------------------------
url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": symbol,
    "interval": "1d",
    "limit": 120  # últimos 120 dias
}

response = requests.get(url, params=params)
data = response.json()

if not isinstance(data, list):
    st.error("Erro ao carregar dados da Binance.")
else:
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    df["data"] = pd.to_datetime(df["open_time"], unit="ms")
    df["preco"] = df["close"].astype(float)
    df = df[["data", "preco"]].sort_values("data")

    # ----------------------------
    # Pré-processamento
    # ----------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie = df["preco"].values.reshape(-1, 1)
    serie_scaled = scaler.fit_transform(serie)

    def create_sequences(data, window_size=30):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    WINDOW = 30
    X, y = create_sequences(serie_scaled, WINDOW)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ----------------------------
    # Modelo LSTM
    # ----------------------------
    model = Sequential([
        LSTM(64, input_shape=(WINDOW, 1)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=80,
        batch_size=16,
        callbacks=[es],
        verbose=0
    )

    # ----------------------------
    # Avaliação
    # ----------------------------
    pred_test = model.predict(X_test)
    pred_test_inv = scaler.inverse_transform(pred_test)
    y_test_inv = scaler.inverse_transform(y_test)

    rmse = math.sqrt(mean_squared_error(y_test_inv, pred_test_inv))
    mae = mean_absolute_error(y_test_inv, pred_test_inv)

    st.subheader("📊 Métricas de Avaliação")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")

    # ----------------------------
    # Gráfico de Previsão
    # ----------------------------
    st.subheader("📈 Preço Real vs Previsto (Teste)")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(y_test_inv.flatten(), label="Real", color="black")
    ax1.plot(pred_test_inv.flatten(), label="Previsto (LSTM)", color="green", linestyle="--")
    ax1.set_title(f"{coin.capitalize()} - Real vs Previsto")
    ax1.set_xlabel("Amostras de Teste")
    ax1.set_ylabel("Preço (USD)")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # ----------------------------
    # Curva de Perda
    # ----------------------------
    st.subheader("📉 Curva de Perda (Loss)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(history.history['loss'], label="Treino")
    ax2.plot(history.history['val_loss'], label="Validação")
    ax2.set_title("Curva de Perda")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("MSE")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
