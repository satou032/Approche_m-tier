import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np 
import feedparser 
import re 
import urllib.parse 
import requests 
from textblob import TextBlob 

# Configuration Streamlit
st.set_page_config(layout="wide", page_title="Dashboard d'Analyse Technique")
st.title("📊 Dashboard d'Analyse Technique")

# ----------------------------------------------------------------------
# Dictionnaires de Sentiment (Extraits pour la logique d'analyse)
# ----------------------------------------------------------------------
POSITIVE_KEYWORDS = {
    'profit': 4, 'bénéfice': 4, 'croissance': 5, 'succès': 5, 'revenus': 4, 'chiffre d\'affaires': 5,
    'hausse': 4, 'augmentation': 4, 'contrat': 4, 'partenariat': 3, 'fort': 4, 'solide': 4, 
    'excellent': 5, 'dépasser': 5, 'record': 4, 'stratégique': 2, 'leader': 3, 'investissement': 3
}
NEGATIVE_KEYWORDS = {
    'perte': 5, 'pertes': 5, 'déficit': 4, 'déclin': 4, 'baisse': 4, 'chute': 5, 'tomber': 3,
    'licenciement': 5, 'suppression d\'emplois': 4, 'échec': 4, 'insuffisant': 3, 'négatif': 4, 
    'faible': 3, 'dégradation': 4, 'pression': 3, 'crise': 5, 'critique': 4, 'retard': 3, 'retire': 1
}

# ---------------------------
# Choix entreprise, technique et période
# ---------------------------
companies = {
    "TotalEnergies": "TTE.PA", 
    "Airbus": "AIR.PA", 
    "Hermes": "RMS.PA",
    "Dassault Systèmes": "DSY.PA",
    "Sopra Steria": "SOP.PA"
}
selected_company = st.selectbox("Choisir une entreprise", list(companies.keys()))

techniques = [
    "Graphe Classique",
    "Leçon 1 : Les Tendances",
    "Leçon 2 : Les Moyennes Mobiles",
    "Leçon 3 : La MACD",
    "Leçon 4 : Les Bollingers",
    "Leçon 5 : Le Stochastique",
    "Leçon 6 : Les Chandeliers",
    "Leçon 7 : Le RSI",
    "Leçon 8 : Le Mouvement Directionnel",
    "Leçon 9 : Les Volumes",
    "Leçon 14 : L'épaule-tête-épaule"
]
selected_technique = st.selectbox("Choisir une technique", techniques)

period_options = {
    "1 Jour (Intraday)": "1d",
    "5 Jours (1 Semaine)": "5d",
    "1 Mois": "1mo",
    "3 Mois": "3mo",
    "6 Mois": "6mo",
    "1 An": "1y",
    "5 Ans": "5y",
    "Max (Historique complet)": "max"
}
selected_period_label = st.selectbox("Choisir la période d'analyse", list(period_options.keys()), index=2) 
selected_period_yf = period_options[selected_period_label]


# ---------------------------
# Télécharger les données
# ---------------------------
ticker = companies[selected_company]
if selected_period_yf == "1d":
    df = yf.download(ticker, period="1d", interval="5m", auto_adjust=True)
else:
    df = yf.download(ticker, period=selected_period_yf, interval="1d", auto_adjust=True) 
    
df.reset_index(inplace=True)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns] 
if "Datetime" in df.columns:
    df.rename(columns={"Datetime": "Date"}, inplace=True)
df.dropna(subset=['Close', 'Volume'], inplace=True)
if df.empty:
    st.error("Aucune donnée disponible pour cette entreprise/période.")
    st.stop()


# ---------------------------
# Calculs techniques (omission pour concision, le code reste inchangé)
# ---------------------------

df["SMA_5"] = df["Close"].rolling(5).mean()
df["SMA_20"] = df["Close"].rolling(20).mean() 

rolling_std = df["Close"].rolling(20).std().squeeze()
if isinstance(rolling_std, pd.DataFrame):
    rolling_std = rolling_std.iloc[:, 0]
df["BB_upper"] = df["SMA_20"] + 2 * rolling_std
df["BB_lower"] = df["SMA_20"] - 2 * rolling_std

delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

df["L14"] = df["Low"].rolling(14).min().squeeze()  
df["H14"] = df["High"].rolling(14).max().squeeze() 
df["Stoch_K"] = 100 * (df["Close"] - df["L14"]) / (df["H14"] - df["L14"]) 
df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

df["plus_DM"] = df["High"].diff().clip(lower=0)
df["minus_DM"] = -df["Low"].diff().clip(upper=0)
df["TR"] = df[["High", "Low", "Close"]].apply(lambda x: max(x["High"] - x["Low"], abs(x["High"] - x["Close"]), abs(x["Low"] - x["Close"])), axis=1)
df["plus_DI"] = 100 * (df["plus_DM"].ewm(alpha=1/14, adjust=False).mean() / df["TR"].ewm(alpha=1/14, adjust=False).mean())
df["minus_DI"] = 100 * (df["minus_DM"].ewm(alpha=1/14, adjust=False).mean() / df["TR"].ewm(alpha=1/14, adjust=False).mean())
df["DX"] = 100 * abs(df["plus_DI"] - df["minus_DI"]) / (df["plus_DI"] + df["minus_DI"])
df["ADX"] = df["DX"].ewm(alpha=1/14, adjust=False).mean()


# ---------------------------
# FONCTION D'ANALYSE DE SENTIMENT (AJOUT DU LABEL TEXTUEL)
# ---------------------------
def _get_news_label(score):
    """Retourne un label textuel basé sur le score de 1 à 5."""
    if score >= 4.0:
        return "Très Favorable"
    elif score >= 3.5:
        return "Favorable"
    elif score <= 2.0:
        return "Très Défavorable"
    elif score <= 2.5:
        return "Défavorable"
    else:
        return "Neutre"

def _enhance_sentiment_with_keywords(text, base_sentiment):
    """Améliore le score TextBlob avec des mots-clés pondérés."""
    text_lower = text.lower()
    positive_impact = 0
    negative_impact = 0
    
    for word, weight in POSITIVE_KEYWORDS.items():
        if word in text_lower:
            positive_impact += weight
    
    for word, weight in NEGATIVE_KEYWORDS.items():
        if word in text_lower:
            negative_impact += weight
    
    net_impact_scaled = (positive_impact - negative_impact) / 10 
    
    enhanced_sentiment = base_sentiment + net_impact_scaled
    limited_sentiment = max(-2.0, min(2.0, enhanced_sentiment))
    
    final_score_1_5 = ((limited_sentiment + 2) / 4) * 4 + 1 

    return max(1.0, min(5.0, final_score_1_5))

@st.cache_data(ttl=3600)
def get_news_score(company_name):
    search_query = f"{company_name} actions"
    encoded_query = urllib.parse.quote(search_query) 
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=fr&gl=FR&ceid=FR:fr"
    
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        # NOTE: Le label doit être renvoyé même en cas d'erreur
        return 3.0, "Erreur de connexion au flux RSS.", "Neutre" 
    
    if not feed.entries:
        return 3.0, "Aucune actualité récente trouvée.", "Neutre"

    score_sum = 0
    
    for entry in feed.entries:
        title = entry.title if entry.title else ""
        content = entry.summary if entry.summary else ""
        text = f"{title} {content}"
        
        analysis = TextBlob(text)
        base_sentiment = analysis.sentiment.polarity 
        final_article_score = _enhance_sentiment_with_keywords(text, base_sentiment)
        
        score_sum += final_article_score
        
    article_count = len(feed.entries)
    
    if article_count > 0:
        news_rating = score_sum / article_count
        news_label = _get_news_label(news_rating) # Déterminer le label
        return news_rating, f"{article_count} articles analysés.", news_label
    else:
        return 3.0, "Aucune actualité pertinente analysée.", "Neutre"


# ---------------------------
# Fonction d'analyse agrégée
# ---------------------------
def generate_score_signal(df, company_name):
    
    if df.empty or len(df) < 5: 
        return 3.0, "Hold", {}, "Données historiques insuffisantes (moins de 5 points de données)."
        
    latest = df.iloc[-1]
    if latest.isnull().any() and len(df) > 1:
        latest = df.iloc[-2]
    elif latest.isnull().any():
         return 3.0, "Hold", {}, "La dernière ligne de données est incomplète (NaN)."
         
    previous = df.iloc[-2] if len(df) >= 2 else None 

    scores = {}
    valid_indicator_count = 0
    
    # ... (Calculs des scores techniques omis pour concision - ils utilisent 'latest' et 'previous') ...

    # 1. SMA (5/20)
    if not pd.isna(latest.get("SMA_5")) and not pd.isna(latest.get("SMA_20")):
        score_sma = 3
        if latest["SMA_5"] > latest["SMA_20"]: score_sma = 4 
        elif latest["SMA_5"] < latest["SMA_20"]: score_sma = 2
        scores["SMA"] = score_sma
        valid_indicator_count += 1
    
    # 2. MACD
    if not pd.isna(latest.get("MACD")) and not pd.isna(latest.get("Signal")):
        score_macd = 3
        if latest["MACD"] > latest["Signal"]: score_macd = 4
        elif latest["MACD"] < latest["Signal"]: score_macd = 2
        scores["MACD"] = score_macd
        valid_indicator_count += 1
    
    # 3. Bollinger Bands
    if not pd.isna(latest.get("BB_lower")) and not pd.isna(latest.get("BB_upper")):
        score_bb = 3
        if latest["Close"] < latest["BB_lower"]: score_bb = 5 
        elif latest["Close"] > latest["BB_upper"]: score_bb = 1 
        scores["Bollinger"] = score_bb
        valid_indicator_count += 1
    
    # 4. RSI 14 
    if not pd.isna(latest.get("RSI_14")):
        score_rsi = 3
        if latest["RSI_14"] < 30: score_rsi = 5
        elif latest["RSI_14"] > 70: score_rsi = 1
        scores["RSI"] = score_rsi
        valid_indicator_count += 1
    
    # 5. Stochastique (%K)
    if not pd.isna(latest.get("Stoch_K")):
        score_stoch = 3
        if latest["Stoch_K"] < 20: score_stoch = 5
        elif latest["Stoch_K"] > 80: score_stoch = 1
        scores["Stochastique"] = score_stoch
        valid_indicator_count += 1

    # 6. ADX (Mouvement Directionnel)
    if not pd.isna(latest.get("ADX")) and not pd.isna(latest.get("plus_DI")) and not pd.isna(latest.get("minus_DI")):
        score_adx = 3
        if latest["ADX"] > 25: 
            if latest["plus_DI"] > latest["minus_DI"]: score_adx = 4 
            elif latest["minus_DI"] > latest["plus_DI"]: score_adx = 2 
        scores["ADX"] = score_adx
        valid_indicator_count += 1
    
    # 7. SCORE ACTUS (Actualités) - Récupération du label textuel
    news_score, news_status, news_label = get_news_score(company_name)
    scores["Actualités (Fund.)"] = news_score
    valid_indicator_count += 1 

    # --- Agrégation Finale ---
    
    if valid_indicator_count <= 1: 
        return 3.0, "Hold", scores, "Seul le score d'actualité est disponible (manque données techniques)."

    final_rating = sum(scores.values()) / valid_indicator_count
    
    # Détermination du signal
    if final_rating > 4.2: final_signal = "Acheter Fort"
    elif final_rating > 2.9: final_signal = "Acheter"
    elif final_rating < 1.8: final_signal = "Vendre Fort"
    elif final_rating < 2.25: final_signal = "Vendre"
    else: final_signal = "Hold"
        
    # Retourne le label d'actualité
    return final_rating, final_signal, scores, news_label


# ----------------------------------------------------
# Layout : Graphe Classique (en haut) + Colonnes (Graphe Technique / Note)
# ----------------------------------------------------

# Appel de la fonction d'analyse agrégée
final_rating, final_signal, individual_scores, news_label = generate_score_signal(df, selected_company)

# --- 1. Graphe Classique (Toujours en haut) ---
st.subheader(f"📈 {selected_company} — Graphique classique (Prix de Clôture) sur {selected_period_label}")
fig_classique = go.Figure()
fig_classique.add_trace(go.Scatter(
    x=df["Date"], y=df["Close"], mode="lines", name="Prix Close", line=dict(color="blue")
))
st.plotly_chart(fig_classique, use_container_width=True, key=f"classique_{selected_company}")

st.markdown("---") 

# --- 2. Colonnes pour Technique et Note ---
col1, col2 = st.columns([3, 1]) 

with col2:
    st.subheader("Analyse Synthétique")

    # Affichage conditionnel basé sur le message d'état
    if individual_scores.get("Actualités (Fund.)") is None:
         st.warning(f"⚠️ Analyse limitée ou impossible. {individual_scores.get('Actualités (Fund.)', 'Veuillez vérifier les données.')}")
    else:
        # Note Finale
        st.metric("Note Technique Finale / 5", f"{final_rating:.2f}")
        full_stars = int(round(final_rating))
        st.write("Note visuelle:", "★" * full_stars + "☆" * (5 - full_stars))
        
        st.markdown("---")
        
        # Signal Final
        if "Fort" in final_signal:
             if "Acheter" in final_signal:
                st.success(f"Signal : **{final_signal}** 🚀")
             else:
                st.error(f"Signal : **{final_signal}** 🔻")
        elif final_signal == "Acheter":
            st.success(f"Signal : **{final_signal}**")
        elif final_signal == "Vendre":
            st.error(f"Signal : **{final_signal}**")
        else:
            st.warning(f"Signal : **{final_signal}**")

        st.markdown("---")
        st.caption("Détail des scores (5=Achat Fort, 1=Vente Forte) :")
        
        # Afficher le détail des scores individuels
        for indicator, score in individual_scores.items():
            if indicator == "Actualités (Fund.)":
                # AFFICHAGE DU LABEL DÉSIRÉ
                emoji = {'Très Favorable': '🟢', 'Favorable': '🟩', 'Neutre': '🟡', 'Défavorable': '🟧', 'Très Défavorable': '🔴'}.get(news_label, '⚪')
                st.markdown(f"- **{indicator}**: {emoji} **{news_label}** ({score:.1f})")
            else:
                color = 'green' if score >= 4 else ('red' if score <= 2 else 'orange')
                st.markdown(f"- **{indicator}**: <span style='color:{color};'>{'★' * int(round(score))} ({score:.1f})</span>", unsafe_allow_html=True)
            
with col1:
    # Graphe technique (à gauche)
    if selected_technique != "Graphe Classique":
        st.subheader(f"Technique : {selected_technique}")
        fig_technique = go.Figure()

        # Leçon 1 (Tendances) et Leçon 2 (Moyennes Mobiles)
        if selected_technique in ["Leçon 1 : Les Tendances", "Leçon 2 : Les Moyennes Mobiles"]:
            fig_technique.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Prix"))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["SMA_5"], name="SMA 5", line=dict(color="blue")))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20", line=dict(color="orange"))) 

        elif selected_technique == "Leçon 3 : La MACD":
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal"))
            
        elif selected_technique == "Leçon 4 : Les Bollingers":
            fig_technique.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Prix"))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], name="BB Upper", line=dict(color="red")))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], name="BB Lower", line=dict(color="green")))

        elif selected_technique == "Leçon 5 : Le Stochastique":
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"], name="%K"))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"], name="%D"))
            fig_technique.add_hline(y=80, line_dash="dash", annotation_text="Overbought", line=dict(color="blue"))
            fig_technique.add_hline(y=20, line_dash="dash", annotation_text="Oversold", line=dict(color="orange"))

        elif selected_technique == "Leçon 6 : Les Chandeliers":
            fig_technique.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Bougies"))

        elif selected_technique == "Leçon 7 : Le RSI":
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["RSI_14"], name="RSI 14"))
            fig_technique.add_hline(y=70, line_dash="dash", annotation_text="Overbought", line=dict(color="red"))
            fig_technique.add_hline(y=30, line_dash="dash", annotation_text="Oversold", line=dict(color="green"))

        elif selected_technique == "Leçon 8 : Le Mouvement Directionnel":
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["plus_DI"], name="+DI"))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["minus_DI"], name="-DI"))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["ADX"], name="ADX"))
        
        elif selected_technique in ["Leçon 9 : Les Volumes"]:
            fig_technique.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume"))
            
        elif selected_technique == "Leçon 14 : L'épaule-tête-épaule":
             fig_technique.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Bougies"))
             st.info("Détection visuelle du pattern Épaule-Tête-Épaule non automatisée dans ce script.")


        st.plotly_chart(fig_technique, use_container_width=True,
                        key=f"technique_{selected_technique}_{selected_company}")
        
    st.dataframe(df.tail(10)) 

st.write("⚠️ Ceci est un outil pédagogique. Ne prend pas de décisions financières sans conseil adapté.")