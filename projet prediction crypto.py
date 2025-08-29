import requests
import pandas as pd
import time
import joblib
import os
try:
    from xgboost import XGBClassifier
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "xgboost"])
    from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime

# === CONFIG ===
LOG_PATH = "logs"
MODEL_PATH = "crypto_explosion_model.pkl"
DATA_PATH = "last_crypto_scan.csv"

def get_top_cryptos(limit=1000):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "1h,24h,7d"
    }
    response = requests.get(url, params=params)
    return response.json()



from ta.momentum import RSIIndicator
from ta.trend import MACD

# Pour le sentiment Twitter
import os
import requests


def collect_data():
    coins = get_top_cryptos()
    data = []
    prices_for_rsi = []  # stockage pour RSI
    # === Paramètre de l'API Twitter ===
    BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
    def get_tweet_count(symbol):
        if not BEARER_TOKEN:
            return 0
        headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
        query = f"{symbol} crypto lang:en -is:retweet"
        url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=10"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("meta", {}).get("result_count", 0)
            else:
                return 0
        except:
            return 0

    for coin in coins:
        try:
            price_close = coin["current_price"]
            pct_change_24h = coin["price_change_percentage_24h"]
            market_cap = coin["market_cap"]
            volume = coin["total_volume"]

            # Estimation approximative du prix d'ouverture (à partir de la variation)
            price_open = price_close / (1 + pct_change_24h / 100)

            # Pour éviter division par zéro ou valeurs aberrantes
            if price_open == 0 or market_cap is None or volume is None:
                continue

            # Approximons volume_change par rapport à market_cap (proxy faible)
            volume_change = (volume / market_cap) * 100 if market_cap else 0
            tweet_mentions = get_tweet_count(coin['symbol'])
            label = 1 if pct_change_24h >= 50 else 0

            prices_for_rsi.append(price_close)

            data.append({
                "pct_change_1h": coin.get("price_change_percentage_1h_in_currency", 0),
                "pct_change_7d": coin.get("price_change_percentage_7d_in_currency", 0),
                "id": coin["id"],
                "symbol": coin["symbol"],
                "price_open": price_open,
                "price_close": price_close,
                "pct_change_24h": pct_change_24h,
                "volume_change": volume_change,
                "market_cap": market_cap,
                "label": label,
                "tweet_mentions": tweet_mentions
            })

            time.sleep(0.2)
        except Exception as e:
            print(f"Erreur avec {coin['id']}: {e}")
    df = pd.DataFrame(data)

    # RSI + MACD (basé sur price_close)
    if len(prices_for_rsi) >= 26:
        df["rsi"] = RSIIndicator(pd.Series(prices_for_rsi)).rsi()
        macd = MACD(pd.Series(prices_for_rsi))
        df["macd"] = macd.macd()
    else:
        df["rsi"] = 0
        df["macd"] = 0
    df.to_csv(DATA_PATH, index=False)
    date_str = datetime.now().strftime("%Y-%m-%d")
    df.to_csv(os.path.join(LOG_PATH, f"last_crypto_scan_{date_str}.csv"), index=False)
    return df

def train_model(df):
    print("\n✅ Colonnes disponibles dans le DataFrame:", df.columns.tolist())
    if df.empty:
        print("❌ Aucune donnée collectée. Vérifie ta connexion ou l'API CoinGecko.")
        exit()
    for col in ["price_open", "volume_change", "market_cap", "rsi", "macd", "tweet_mentions", "pct_change_1h", "pct_change_7d"]:
        if col not in df.columns:
            print(f"❌ Colonne manquante : {col}")
            exit()

    features = ["price_open", "volume_change", "market_cap", "rsi", "macd", "tweet_mentions", "pct_change_1h", "pct_change_7d"]
    X = df[features]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(model, MODEL_PATH)
    print("✅ Modèle entraîné avec les colonnes :", X.columns.tolist())
    return model

def predict_today(model, df):
    features = ["price_open", "volume_change", "market_cap", "rsi", "macd", "tweet_mentions"]
    df["prediction"] = model.predict(df[features])
    explosifs = df[df["prediction"] == 1]
    explosifs = explosifs.sort_values(by="pct_change_24h", ascending=False)

    print("\n🚀 Cryptos potentiellement explosives aujourd'hui :")
    if explosifs.empty:
        print("Aucune détectée.")
    else:
        print(explosifs[["symbol", "pct_change_24h", "volume_change", "market_cap"]])

    # Sauvegarde du log du jour
    os.makedirs(LOG_PATH, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    explosifs.to_csv(os.path.join(LOG_PATH, f"explosifs_{date_str}.csv"), index=False)

    # Affichage du top 5 des plus mentionnées sur Twitter
    top_tweeted = df.sort_values(by="tweet_mentions", ascending=False).head(5)
    print("\n🔥 Top 5 cryptos les plus tweetées :")
    print(top_tweeted[["symbol", "tweet_mentions", "pct_change_24h"]])

        # Enregistrement du top 5 des plus tweetées
    top_tweeted.to_csv(os.path.join(LOG_PATH, f"top_tweeted_{date_str}.csv"), index=False)
    return explosifs

if __name__ == "__main__":
    print("📡 Collecte des données...")
    df = collect_data()

    if not os.path.exists(MODEL_PATH):
        print("🧠 Entraînement initial du modèle...")
        model = train_model(df)
    else:
        model = joblib.load(MODEL_PATH)

    print("🔮 Prédiction des cryptos explosives...")
    predict_today(model, df)
    print("✅ Terminé.")


