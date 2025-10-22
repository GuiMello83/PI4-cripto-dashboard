import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Visualização de Criptomoedas", layout="centered")

st.title("📊 Visualização de Criptomoedas")
st.markdown("Este painel mostra a média mensal de preço e volume de uma criptomoeda com dados atualizados da API da Binance.")

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
    # Gráfico
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(media_mensal.index.to_timestamp(), media_mensal["preco"], marker="o", color="green", linewidth=2)
    ax.set_title(f"{coin.capitalize()} - Preço médio mensal")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Preço médio (USD)")
    ax.grid(True)
    plt.xticks(rotation=45)

    st.pyplot(fig)
