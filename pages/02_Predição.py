import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Predição de Criptomoedas", layout="centered")

st.title("🔮 Predição de Preços de Criptomoedas")
st.markdown("Este painel utiliza regressão linear simples para prever os preços médios mensais das criptomoedas nos próximos 3 meses.")

# ----------------------------
# Seleção de moeda
# ----------------------------
coin = st.selectbox("Escolha a moeda para prever:", ["bitcoin", "ethereum", "solana"])
start_date = int(datetime(2024, 12, 1).timestamp())
end_date = int(datetime(2025, 2, 28).timestamp())

# ----------------------------
# Requisição à API
# ----------------------------
url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range"
params = {
    "vs_currency": "usd",
    "from": start_date,
    "to": end_date
}

response = requests.get(url, params=params)
data = response.json()

if "prices" not in data:
    st.error("Erro ao carregar dados da API.")
else:
    # ----------------------------
    # Preparar dados
    # ----------------------------
    df_precos = pd.DataFrame(data["prices"], columns=["timestamp_ms", "preco"])
    df_mercado = pd.DataFrame(data["market_caps"], columns=["timestamp_ms", "capitalizacao_mercado"])
    df_volume = pd.DataFrame(data["total_volumes"], columns=["timestamp_ms", "volume"])

    df = df_precos.merge(df_mercado, on="timestamp_ms").merge(df_volume, on="timestamp_ms")
    df["data"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df = df[["data", "preco", "capitalizacao_mercado", "volume"]]

    df["ano_mes"] = df["data"].dt.to_period("M")
    media_mensal = df.groupby("ano_mes")[["preco", "capitalizacao_mercado", "volume"]].mean()

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
    futuro = pd.date_range("2025-03-01", "2025-05-01", freq="MS")
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
