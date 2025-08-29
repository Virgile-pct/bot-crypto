import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURATION ===
LOG_PATH = "logs"

def load_logs():
    dataframes = []
    for file in os.listdir(LOG_PATH):
        if file.startswith("last_crypto_scan_") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(LOG_PATH, file))
            if "label" in df.columns and "prediction" in df.columns:
                dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

def backtest():
    df = load_logs()
    if df.empty:
        print("❌ Aucun fichier de log compatible trouvé dans le dossier 'logs'.")
        return

    y_true = df["label"]
    y_pred = df["prediction"]

    print("✅ Backtest sur", len(df), "prédictions totales")
    print("\n=== Rapport de classification ===")
    print(classification_report(y_true, y_pred))
    print("\n=== Matrice de confusion ===")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    backtest()