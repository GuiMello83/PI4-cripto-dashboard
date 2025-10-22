import requests
import pandas as pd

# Moedas desejadas
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
all_data = []

for symbol in symbols:
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": 365
    }

    response = requests.get(url, params=params)
    data = response.json()

    if not isinstance(data, list):
        print(f"Erro ao carregar dados da Binance para {symbol}: {response.status_code}")
        continue

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    df["data"] = pd.to_datetime(df["open_time"], unit="ms")
    df["preco"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["symbol"] = symbol

    all_data.append(df[["data", "preco", "volume", "symbol"]])

# Concatenar e salvar
final_df = pd.concat(all_data)
final_df.to_csv("dados_binance.csv", index=False)
print("✅ Dados salvos em dados_binance.csv para BTC, ETH e SOL")
