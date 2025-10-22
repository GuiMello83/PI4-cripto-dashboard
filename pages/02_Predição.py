import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Predição de Criptomoedas", layout="centered")

st.title("🔮 Predição de Preços de Criptomoedas")
st.markdown("Este painel utiliza regressão linear simples para prever os preços médios mensais das criptomoedas nos próximos 3 meses, com dados atualizados da API da Binance.")

# ----------------------------
# Seleção de moeda
# ----------------------------
coin = st.selectbox("Escolha a moeda para prever:", ["bitcoin", "ethereum", "solana"])
symbol_map = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT"
}
symbol = symbol_map[coin]

# ----------------------------
# Requisição à API Binance
# ----------------------------
url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=1000"
params = {
    "symbol": symbol,
    "interval": "1d",
    "limit": 90  # últimos 90 dias
}

response = requests.get(url, params=params)
data = response.json()

if not isinstance(data, list):
    st.error("Erro ao carregar dados da Binance.")
else:
    # ----------------------------
    # Tratamento dos dados
    # ----------------------------
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    df["data"] = pd.to_datetime(df["open_time"], unit="ms")
    df["preco"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df["ano_mes"] = df["data"].dt.to_period("M")
    media_mensal = df.groupby("ano_mes")[["preco", "volume"]].mean()

    # ----------------------------
    # Modelo preditivo
    # ----------------------------
    media_mensal.reset_index(inplace=True)
    media_mensal["timestamp"] = media_mensal["ano_mes"].dt.to_timestamp().astype(np.int64) // 10**9
    X = media_mensal[["timestamp"]]
    y = media_mensal["preco"]

    modelo = LinearRegression()
    modelo.fit(X, y)

    # Prever próximos 3 meses
    futuro = pd.date_range(media_mensal["ano_mes"].max().to_timestamp() + pd.offsets.MonthBegin(),
                           periods=3, freq="MS")
    futuro_ts = np.array(futuro.astype(np.int64) // 10**9).reshape(-1, 1)
    previsoes = modelo.predict(futuro_ts)

    # ----------------------------
    # Gráfico
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(media_mensal["ano_mes"].dt.to_timestamp(), media_mensal["preco"], label="Histórico", marker="o")
    ax.plot(futuro, previsoes, label="Previsão", linestyle="--", color="red")
    ax.set_title(f"{coin.capitalize()} - Preço médio + Previsão")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Preço médio (USD)")
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)
