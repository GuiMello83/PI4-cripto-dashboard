import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Análise de Criptomoedas", layout="centered")

st.title("📊 Visualização de Criptomoedas")
st.markdown("Este painel mostra a média mensal de preço, capitalização de mercado e volume de uma criptomoeda.")

# ----------------------------
# Seleção de moeda
# ----------------------------
coin = st.selectbox("Escolha a moeda:", ["bitcoin", "ethereum", "solana"])
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
    df_precos = pd.DataFrame(data["prices"], columns=["timestamp_ms", "preco"])
    df_mercado = pd.DataFrame(data["market_caps"], columns=["timestamp_ms", "capitalizacao_mercado"])
    df_volume = pd.DataFrame(data["total_volumes"], columns=["timestamp_ms", "volume"])

    df = df_precos.merge(df_mercado, on="timestamp_ms").merge(df_volume, on="timestamp_ms")
    df["data"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df = df[["data", "preco", "capitalizacao_mercado", "volume"]]

    df["ano_mes"] = df["data"].dt.to_period("M")
    media_mensal = df.groupby("ano_mes")[["preco", "capitalizacao_mercado", "volume"]].mean()

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
