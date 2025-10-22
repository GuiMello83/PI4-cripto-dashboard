import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Visualização de Criptomoedas", layout="centered")

st.title("📊 Visualização de Criptomoedas")
st.markdown("Este painel mostra a média mensal de preço e volume de uma criptomoeda com base em dados salvos localmente da API da Binance.")

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
