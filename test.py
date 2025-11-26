import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np 
import feedparser 
import re 
import urllib.parse 

st.set_page_config(layout="wide", page_title="Dashboard d'Analyse Technique")
st.title("📊 Dashboard d'Analyse Technique")

# ---------------------------
# Choix entreprise et technique
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

# ---------------------------
# Télécharger les données (période 1 mois conservée)
# ---------------------------
ticker = companies[selected_company]
df = yf.download(ticker, period="1mo", interval="1d", auto_adjust=True) 
df.reset_index(inplace=True)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns] 

# ---------------------------
# Calculs techniques
# ---------------------------

df["SMA_5"] = df["Close"].rolling(5).mean()
df["SMA_20"] = df["Close"].rolling(20).mean() # Peut générer NaN au début

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
# FONCTION : Analyse d'Actualités
# ---------------------------
@st.cache_data(ttl=3600)
def get_news_score(company_name):
    search_query = f"{company_name} actions"
    encoded_query = urllib.parse.quote(search_query) 
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=fr&gl=FR&ceid=FR:fr"
    
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        # st.error(f"Erreur lors de la récupération du flux RSS : {e}")
        return 3.0, "Erreur de connexion au flux RSS."
    
    total_articles = len(feed.entries)
    
    if total_articles == 0:
        return 3.0, "Aucune actualité récente trouvée."

    positive_keywords = r'gagne|hausse|progresse|record|succès|croissance|achat|augmentation|fort|solide'
    negative_keywords = r'perd|chute|baisse|déclin|difficulté|vendre|crise|scandale|recul|procès|pénurie'

    score_sum = 0
    article_count = 0

    for entry in feed.entries:
        title = entry.title.lower()
        article_score = 3.0
        
        pos_count = len(re.findall(positive_keywords, title))
        neg_count = len(re.findall(negative_keywords, title))

        if pos_count > neg_count:
            article_score = 5.0
        elif neg_count > pos_count:
            article_score = 1.0
        
        score_sum += article_score
        article_count += 1
        
    if article_count > 0:
        news_rating = score_sum / article_count
        return news_rating, f"{article_count} articles analysés."
    else:
        return 3.0, "Aucune actualité pertinente analysée."


# ---------------------------
# Fonction d'analyse agrégée (CORRIGÉE)
# ---------------------------
def generate_score_signal(df, company_name):
    
    # Nouvelle condition pour le nombre de jours minimum (ex: 20 jours)
    if df.empty or len(df) < 20: 
        return 3.0, "Hold", {}, "Données historiques insuffisantes (moins de 20 jours)."
        
    # Vérification de la validité de la dernière ligne
    latest = df.iloc[-1]
    if latest.isnull().any():
         return 3.0, "Hold", {}, "La dernière ligne de données est incomplète (NaN)."
         
    previous = df.iloc[-2] if len(df) >= 2 else None 

    scores = {}
    valid_indicator_count = 0
    
    # 1. SMA (5/20)
    # Rendre la vérification plus robuste en incluant un test de NaN pour chaque indicateur
    if not pd.isna(latest.get("SMA_5")) and not pd.isna(latest.get("SMA_20")):
        score_sma = 3
        if latest["SMA_5"] > latest["SMA_20"]:
            score_sma = 4 
            if previous is not None and not pd.isna(previous["SMA_5"]) and previous["SMA_5"] < latest["SMA_5"]: score_sma = 5 
        elif latest["SMA_5"] < latest["SMA_20"]:
            score_sma = 2
            if previous is not None and not pd.isna(previous["SMA_5"]) and previous["SMA_5"] > latest["SMA_5"]: score_sma = 1 
        scores["SMA (5/20)"] = score_sma
        valid_indicator_count += 1
    
    # 2. MACD
    if not pd.isna(latest.get("MACD")) and not pd.isna(latest.get("Signal")):
        score_macd = 3
        if latest["MACD"] > latest["Signal"]:
            score_macd = 4
            if previous is not None and not pd.isna(previous["MACD"]) and not pd.isna(previous["Signal"]) and previous["MACD"] < previous["Signal"]: score_macd = 5 
        elif latest["MACD"] < latest["Signal"]:
            score_macd = 2
            if previous is not None and not pd.isna(previous["MACD"]) and not pd.isna(previous["Signal"]) and previous["MACD"] > previous["Signal"]: score_macd = 1 
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
    
    # 7. SCORE ACTUS (Actualités)
    news_score, news_status = get_news_score(company_name)
    scores["Actualités (Fund.)"] = news_score
    valid_indicator_count += 1 

    # --- Agrégation Finale ---
    
    if valid_indicator_count <= 1: # Si seul le score d'actualité est valide, on retourne Hold
        return 3.0, "Hold", {}, "Seul le score d'actualité est disponible (manque données techniques)."

    final_rating = sum(scores.values()) / valid_indicator_count
    
    # Détermination du signal
    if final_rating > 4.2: final_signal = "Acheter Fort"
    elif final_rating > 2.89: final_signal = "Acheter"
    elif final_rating < 1.8: final_signal = "Vendre Fort"
    elif final_rating < 2.25: final_signal = "Vendre"
    else: final_signal = "Hold"
        
    return final_rating, final_signal, scores, news_status


# ----------------------------------------------------
# Layout : Graphe Classique (en haut) + Colonnes (Graphe Technique / Note)
# ----------------------------------------------------

# Appel de la fonction d'analyse agrégée
final_rating, final_signal, individual_scores, news_status = generate_score_signal(df, selected_company)

# --- 1. Graphe Classique (Toujours en haut) ---
st.subheader(f"📈 {selected_company} — Graphique classique (Prix de Clôture) sur 1 Mois")
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
    if individual_scores.get('Actualités (Fund.)') is None and final_signal == "Hold":
         # Si le score d'actualité n'a pas été calculé ET que le signal est Hold par défaut
         st.warning("⚠️ Chargement en cours ou données insuffisantes pour l'analyse technique.")
         st.markdown(f"*{individual_scores.get('Actualités (Fund.)', 'Veuillez patienter ou vérifier la connexion.')}*") # Affichage du statut
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
                st.markdown(f"- **{indicator}**: {'⭐' * int(round(score))} ({score:.1f}) _({news_status})_")
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
            fig_technique.add_hline(y=80, line_dash="dash", annotation_text="Overbought")
            fig_technique.add_hline(y=20, line_dash="dash", annotation_text="Oversold")

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