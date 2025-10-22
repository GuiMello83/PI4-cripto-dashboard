import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os

st.set_page_config(page_title="Predição de Criptomoedas", layout="centered")

st.title("🔮 Predição de Preços de Criptomoedas")
st.markdown("Este painel utiliza regressão linear simples para prever os preços médios mensais das criptomoedas nos próximos 3 meses, com base em dados salvos localmente.")

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
csv_file = "dados_binance.csv"

# ----------------------------
# Verificação do arquivo
# ----------------------------

if not os.path.exists(csv_file):
    st.error(f"Arquivo de dados '{csv_file}' não encontrado. Execute o script 'dados_binance.py' para gerar os dados.")
    st.stop()


# ----------------------------
# Carregamento e filtragem
# ----------------------------
df = pd.read_csv(csv_file)
df["data"] = pd.to_datetime(df["data"])
df = df[df["symbol"] == symbol]

if df.empty:
    st.error(f"Não há dados disponíveis para {coin}.")
    st.stop()

df["ano_mes"] = df["data"].dt.to_period("M")
media_mensal = df.groupby("ano_mes")[["preco", "volume"]].mean().reset_index()

# ----------------------------
# Modelo preditivo
# ----------------------------
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
