"""
Stock Dashboard Streamlit
- Récupère les données soit depuis des fichiers JSON locaux (ex: Hermes.json) soit via yfinance
- Calcule RSI (14), SMA(5) et SMA(20)
- Affiche graphiques (bougies, SMA, RSI)
- Donne un signal d'achat/vente simple basé sur RSI et tendance SMA
- Affiche une NOTE NUMÉRIQUE finale (score) et une interprétation (Strong Buy/Buy/Hold/Sell/Strong Sell)

Pour exécuter:
pip install streamlit yfinance pandas plotly
streamlit run stock_dashboard_streamlit.py

"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Dashboard Actions — RSI & SMA")

# ---------- Utilities ----------

def load_from_json(name, ticker):
    filename = f"{name}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                payload = json.load(f)
            records = payload.get("data", [])
            if not records:
                return None
            df = pd.DataFrame(records)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]) 
                df = df.set_index("Date")
            return df
        except Exception as e:
            st.warning(f"Impossible de lire {filename} : {e}")
            return None
    return None


def download_yfinance(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df


def compute_sma(df, window):
    return df["Close"].rolling(window=window).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def score_from_indicators(latest_rsi, sma5_latest, sma20_latest):
    """Calcule une note numérique basée sur RSI et SMA crossover.
    Méthode simple et pédagogique :
      - RSI: survente -> positif, surachat -> négatif
      - SMA crossover: tendance haussière -> positif, baissière -> négatif
    Retourne tuple (score, label, rating_5)
    score: valeur entre -100 et 100
    rating_5: note normalisée entre 0 et 5
    """
    if latest_rsi is None or (sma5_latest is None) or (sma20_latest is None):
        return None, "No data", None

    score = 0.0
    # RSI contribution
    if latest_rsi < 30:
        score += 50  # forte survente
    elif latest_rsi < 50:
        score += 20  # légèrement positif
    elif latest_rsi <= 70:
        score += 0   # neutre
    else:
        score -= 50  # surachat -> négatif

    # SMA contribution
    if sma5_latest > sma20_latest:
        score += 30
    elif sma5_latest < sma20_latest:
        score -= 30

    # Normalisation (cap)
    if score > 100:
        score = 100
    if score < -100:
        score = -100

    # Label mapping
    if score >= 40:
        label = "Strong Buy"
    elif score >= 10:
        label = "Buy"
    elif score > -10:
        label = "Hold"
    elif score >= -39:
        label = "Sell"
    else:
        label = "Strong Sell"

    # Convertir score [-100,100] en note sur 5 (0 = très mauvais, 5 = excellent)
    # mapping linéaire : -100 -> 0, 100 -> 5
    rating_5 = round(((score + 100) / 200) * 5, 2)

    return score, label, rating_5

# ---------- Sidebar ----------
st.sidebar.title("Paramètres")
companies = {
    "Hermes": "RMS.PA",
    "Dassault_Systèmes": "DSY.PA",
    "Sopra_Steria": "SOP.PA",
    "TotalEnergies": "TTE.PA",
    "Airbus": "AIR.PA"
}

selected = st.sidebar.selectbox("Choisir une entreprise", list(companies.keys()))
period_days = st.sidebar.selectbox("Période (jours)", [30, 90, 180, 365], index=3)
period_str = f"{period_days}d"
interval = st.sidebar.selectbox("Intervalle", ["1d", "1wk"], index=0)

# ---------- Data loading ----------
st.title("Dashboard technique — RSI & SMA (5)")

ticker = companies[selected]

df = load_from_json(selected, ticker)
if df is None:
    df = download_yfinance(ticker, period=f"{period_days}d", interval=interval)
    if df is None:
        st.error("Impossible de récupérer des données pour ce ticker.")
        st.stop()
else:
    start = pd.to_datetime("today") - pd.Timedelta(days=period_days)
    df = df[df.index >= start]

for col in ["Open", "High", "Low", "Close", "Volume"]:
    if col not in df.columns:
        st.error(f"Colonne manquante: {col}")
        st.stop()

# Calculs techniques
df = df.sort_index()
df["SMA_5"] = compute_sma(df, 5)
df["SMA_20"] = compute_sma(df, 20)
df["RSI_14"] = compute_rsi(df["Close"], period=14)

latest = df.iloc[-1]
latest_rsi = float(latest["RSI_14"]) if not pd.isna(latest["RSI_14"]) else None
sma5_latest = float(latest["SMA_5"]) if not pd.isna(latest["SMA_5"]) else None
sma20_latest = float(latest["SMA_20"]) if not pd.isna(latest["SMA_20"]) else None

score, label, rating = score_from_indicators(latest_rsi, sma5_latest, sma20_latest)

# ---------- Layout ----------
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"{selected} — {ticker}")
    st.write(f"Dernière date : {df.index[-1].date()}")

    # Candlestick + SMA overlays
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Prix"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="SMA 5", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(width=1)))
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df[["Close", "SMA_5", "SMA_20", "RSI_14"]].tail(10))

with col2:
    st.markdown("### Indicateurs")
    st.metric("RSI(14)", f"{latest_rsi:.2f}" if latest_rsi is not None else "N/A")
    st.metric("SMA5", f"{sma5_latest:.2f}" if sma5_latest is not None else "N/A")
    st.metric("SMA20", f"{sma20_latest:.2f}" if sma20_latest is not None else "N/A")
    st.markdown("---")
    #st.markdown("### Note finale (score)")
    if score is None:
        st.info("Pas assez de données pour calculer la note.")
    else:
        # affichage clair de la note, du label et de la note sur 5
        # st.subheader(f"Score: {score:.1f} / 100")
        st.metric("Note (sur 5)", f"{rating:.2f} / 5")
        # affichage étoilé simple
        try:
            full_stars = int(round(rating))
            stars = "★" * full_stars + "☆" * (5 - full_stars)
        except Exception:
            stars = "N/A"
        st.write(f"Interprétation: **{label}**  ")
        st.write(f"Note visuelle: {stars}")

        if score >= 40:
            st.success(f"{label} — Recommandation: ACHETER")
        elif score >= 10:
            st.success(f"{label} — Recommandation: ACHETER (suivre)")
        elif score > -10:
            st.info(f"{label} — Recommandation: NE RIEN FAIRE (HOLD)")
        elif score >= -39:
            st.error(f"{label} — Recommandation: VENDRE")
        else:
            st.error(f"{label} — Recommandation: VENDRE FORTEMENT")

    st.markdown("---")
    st.markdown("### Règles utilisées pour la note:")
    st.markdown("""
    - **RSI**  
    - < 30 → +50  
    - 30–50 → +20  
    - 50–70 → 0  
    - > 70 → -50  

    - **Tendance (SMA)**  
    - SMA5 > SMA20 → +30  
    - SMA5 < SMA20 → -30  
    """)


# RSI chart
st.subheader("RSI (14)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI 14"))
fig2.add_hline(y=70, line_dash="dash", annotation_text="Overbought 70", annotation_position="top left")
fig2.add_hline(y=30, line_dash="dash", annotation_text="Oversold 30", annotation_position="bottom left")
fig2.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

# Footer: option d'export JSON
st.markdown("---")
if st.button("Exporter les données affichées en JSON"):
    export_payload = {
        "entreprise": selected,
        "ticker": ticker,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "score": score,
        "label": label,
        "rating_5": rating,
        "data": df.reset_index().to_dict(orient="records")
    }
    outname = f"{selected}_export.json"
    with open(outname, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=4, default=str)
    st.success(f"Fichier exporté : {outname}")

st.write("NB: Ceci est un outil pédagogique. Ne prend pas de décisions financières sans conseil adapté.")
