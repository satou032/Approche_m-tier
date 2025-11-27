import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import numpy as np 
import feedparser 
import re 
import urllib.parse 
import requests 
from textblob import TextBlob 

logo_esigelec_path = "esigelec.png" 
second_logo_path = "logo_equipe2.png" 


col_logo_left, col_spacer, col_logo_right = st.columns([1, 5, 1]) 

with col_logo_left:
    st.image(logo_esigelec_path, width=200) # Ajustez la largeur si nécessaire

with col_logo_right:
    
    st.image(second_logo_path, width=200)

# Configuration Streamlit
st.set_page_config(layout="wide", page_title="Dashboard d'Analyse Technique")
st.title("📊 Objectif Gain")



# ----------------------------------------------------------------------
# Dictionnaires Statiques (Sentiment et Options)
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
companies = {
    "TotalEnergies": "TTE.PA", 
    "Airbus": "AIR.PA", 
    "Hermes": "RMS.PA",
    "Dassault Systèmes": "DSY.PA",
    "Sopra Steria": "SOP.PA"
}
techniques = [
    "Graphe Classique",
    "Leçon 1 : Les Tendances",
    "Leçon 2 : Les Moyennes Mobiles",
    "Leçon 3 : La MACD",
    "Leçon 4 : Les Bollingers",
    "Leçon 5 : Le Stochastique",
    "Leçon 6 : Les Chandeliers",
    "Leçon 7 : Le RSI",
    "Leçon 9 : Les Volumes",
    "Leçon 14 : L'épaule-tête-épaule",
    "Leçon 15 : Le Momentum"
]
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

# ---------------------------
# Sélecteurs
# ---------------------------
selected_company = st.selectbox("Choisir une entreprise", list(companies.keys()))
selected_technique = st.selectbox("Choisir une technique", techniques)
selected_period_label = st.selectbox("Choisir la période d'analyse", list(period_options.keys()), index=2) 
selected_period_yf = period_options[selected_period_label]

# ---------------------------
# FONCTIONS UTILITAIRES DE SENTIMENT (omises pour la concision)
# ---------------------------

def _get_news_label(score):
    if score >= 4.0: return "Très Favorable"
    elif score >= 3.5: return "Favorable"
    elif score <= 2.0: return "Très Défavorable"
    elif score <= 2.5: return "Défavorable"
    else: return "Neutre"

def _enhance_sentiment_with_keywords(text, base_sentiment):
    text_lower = text.lower()
    positive_impact = 0
    negative_impact = 0
    for word, weight in POSITIVE_KEYWORDS.items():
        if word in text_lower: positive_impact += weight
    for word, weight in NEGATIVE_KEYWORDS.items():
        if word in text_lower: negative_impact += weight
    net_impact_scaled = (positive_impact - negative_impact) / 10 
    enhanced_sentiment = base_sentiment + net_impact_scaled
    limited_sentiment = max(-2.0, min(2.0, enhanced_sentiment))
    final_score_1_5 = ((limited_sentiment + 2) / 4) * 4 + 1 
    return max(1.0, min(5.0, final_score_1_5))

@st.cache_data(ttl=300) # Cache de 5 minutes
def get_news_score(company_name):
    search_query = f"{company_name} actions"
    encoded_query = urllib.parse.quote(search_query) 
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=fr&gl=FR&ceid=FR:fr"
    
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
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
        news_label = _get_news_label(news_rating) 
        return news_rating, f"{article_count} articles analysés.", news_label
    else:
        return 3.0, "Aucune actualité pertinente analysée.", "Neutre"

# ---------------------------
# Téléchargement et Calculs (Mise en cache pour la stabilité)
# ---------------------------
@st.cache_data(ttl=60) # Cache de 1 minutes
def load_and_calculate_data(ticker, period_yf):
    
    if period_yf == "1d":
        df = yf.download(ticker, period="1d", interval="1m", auto_adjust=True)
    elif period_yf == "5d":
        # Pour 5 jours, on passe en 30 minutes pour avoir assez de bougies
        # (environ 70 bougies sur 5 jours, suffisant pour la SMA 20 et les Bollingers)
        df = yf.download(ticker, period="5d", interval="30m", auto_adjust=True)
    else:
        df = yf.download(ticker, period=period_yf, interval="1d", auto_adjust=True) 
        
    df.reset_index(inplace=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns] 
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])

    if df["Date"].dt.tz is not None:
        # On convertit vers le fuseau de Paris
        df["Date"] = df["Date"].dt.tz_convert("Europe/Paris")
    else:
        # Si pas de timezone, on suppose UTC et on convertit
        df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("Europe/Paris")

    df.dropna(subset=['Close', 'Volume'], inplace=True)
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculs techniques
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20, min_periods=20).mean() 

    rolling_std = df["Close"].rolling(20, min_periods=20).std().squeeze()
    if isinstance(rolling_std, pd.DataFrame): rolling_std = rolling_std.iloc[:, 0]
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
    df["MACDH"] = df["MACD"] - df["Signal"] 
    
    # MOMENTUM (Dynamique)
    df['Momentum_10'] = df['Close'].diff(10)
    df['Momentum_5'] = df['Close'].diff(5) 
    df['Momentum_1'] = df['Close'].diff(1) # Pour période ultra-courte (5 jours ou moins)

    df["L14"] = df["Low"].rolling(14).min().squeeze()  
    df["H14"] = df["High"].rolling(14).max().squeeze() 
    df["Stoch_K"] = 100 * (df["Close"] - df["L14"]) / (df["H14"] - df["L14"]) 
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # Mouvement Directionnel (Gardé pour l'agrégation, retiré de la liste de choix)
    df["plus_DM"] = df["High"].diff().clip(lower=0)
    df["minus_DM"] = -df["Low"].diff().clip(upper=0)
    df["TR"] = df[["High", "Low", "Close"]].apply(lambda x: max(x["High"] - x["Low"], abs(x["High"] - x["Close"]), abs(x["Low"] - x["Close"])), axis=1)
    df["plus_DI"] = 100 * (df["plus_DM"].ewm(alpha=1/14, adjust=False).mean() / df["TR"].ewm(alpha=1/14, adjust=False).mean())
    df["minus_DI"] = 100 * (df["minus_DM"].ewm(alpha=1/14, adjust=False).mean() / df["TR"].ewm(alpha=1/14, adjust=False).mean())
    df["DX"] = 100 * abs(df["plus_DI"] - df["minus_DI"]) / (df["plus_DI"] + df["minus_DI"])
    df["ADX"] = df["DX"].ewm(alpha=1/14, adjust=False).mean()
    
    return df
    

# ---------------------------
# Fonction d'analyse agrégée (Scoring)
# ---------------------------
def generate_score_signal(df, company_name):
    
    if df.empty or len(df) < 5: 
        return 3.0, "Hold", {}, "Données historiques insuffisantes (moins de 5 points de données)."
        
    latest = df.iloc[-1]
    if latest.isnull().all():
         latest = df.iloc[-2] if len(df) >= 2 else latest 

    previous = df.iloc[-2] if len(df) >= 2 else None 

    scores = {}
    valid_indicator_count = 0
    
    # Choix dynamique du Momentum pour le scoring
    if len(df) <= 5: 
        mom_column = 'Momentum_1'
        mom_key = 'Momentum (1j)'
    elif len(df) <= 10:
        mom_column = 'Momentum_5'
        mom_key = 'Momentum (5j)'
    else:
        mom_column = 'Momentum_10'
        mom_key = 'Momentum (10j)'
        
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
    
    # 4. RSI 14 (Logique proportionnelle)
    if not pd.isna(latest.get("RSI_14")):
        rsi = latest["RSI_14"]
        if rsi < 30: score_rsi = 5.0
        elif rsi > 70: score_rsi = 1.0
        elif rsi >= 50: score_rsi = 3.0 + 1.0 * ((rsi - 50.0) / 20.0)
        else: score_rsi = 3.0 - 1.0 * ((50.0 - rsi) / 20.0)
        scores["RSI"] = score_rsi
        valid_indicator_count += 1
    
    # 5. Stochastique (%K) (Logique proportionnelle)
    if not pd.isna(latest.get("Stoch_K")):
        stoch_k = latest["Stoch_K"]
        if stoch_k < 20: score_stoch = 5.0
        elif stoch_k > 80: score_stoch = 1.0
        elif stoch_k >= 50: score_stoch = 3.0 + 1.5 * ((stoch_k - 50.0) / 30.0)
        else: score_stoch = 3.0 - 1.5 * ((50.0 - stoch_k) / 30.0)
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
        
    # 7. MOMENTUM (Dynamique)
    if not pd.isna(latest.get(mom_column)):
        momentum = latest[mom_column]
        if momentum > 0: score_mom = 4.0 
        elif momentum < 0: score_mom = 2.0
        else: score_mom = 3.0
        scores[mom_key] = score_mom
        valid_indicator_count += 1
    
    # 8. SCORE ACTUS (Actualités) - Note qu'il est toujours le dernier pour la moyenne
    news_score, news_status, news_label = get_news_score(company_name)
    scores["Actualités (Fund.)"] = news_score
    valid_indicator_count += 1 

    # --- Agrégation Finale ---
    
    if valid_indicator_count <= 1: 
        return 3.0, "Hold", scores, news_label

    final_rating = sum(scores.values()) / valid_indicator_count
    
    # Détermination du signal
    if final_rating > 4.2: final_signal = "Acheter Fort"
    elif final_rating > 3.4: final_signal = "Acheter"
    elif final_rating < 1.8: final_signal = "Vendre Fort"
    elif final_rating < 2.6: final_signal = "Vendre"
    else: final_signal = "Ne Rien Faire" 
        
    return final_rating, final_signal, scores, news_label


# ----------------------------------------------------
# LOGIQUE PRINCIPALE ET MISE À JOUR MANUELLE
# ----------------------------------------------------

# Bouton pour forcer le rechargement des données et vider le cache
if st.button("Actualiser les Données (Dernières cotations disponibles)"):
    st.cache_data.clear() 
    st.rerun() 
    
# 1. Charger/Calculer les données (Utilise le cache de 5 minutes)
df = load_and_calculate_data(companies[selected_company], selected_period_yf)

if df.empty:
    st.info("Chargement en cours ou données non disponibles pour cette période. Veuillez réessayer.")
    st.stop()

# 2. Lancer l'analyse agrégée
final_rating, final_signal, individual_scores, news_label = generate_score_signal(df, selected_company)

# --- 1. Graphe Classique (Affiche UNIQUEMENT LE PRIX) ---
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

    # Affichage conditionnel
    if len([s for i, s in individual_scores.items() if i != 'Actualités (Fund.)']) < 2 and final_rating == 3.0:
         st.warning(f"⚠️ Analyse technique limitée ou impossible.")
         if 'Actualités (Fund.)' in individual_scores:
             st.metric("Score Actualités / 5", f"{individual_scores['Actualités (Fund.)']:.2f}")
         st.markdown(f"*{news_label}*")
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
                emoji = {'Très Favorable': '🟢', 'Favorable': '🟩', 'Neutre': '🟡', 'Défavorable': '🟧', 'Très Défavorable': '🔴'}.get(news_label, '⚪')
                st.markdown(f"- **{indicator}**: {emoji} **{news_label}** ({score:.1f})")
            else:
                color = 'green' if score >= 4 else ('red' if score <= 2 else 'orange')
                st.markdown(f"- **{indicator}**: <span style='color:{color};'>{'★' * int(round(score))} ({score:.1f})</span>", unsafe_allow_html=True)
            
with col1:
    # Graphe technique (à gauche) - Rendu des autres Leçons
    if selected_technique != "Graphe Classique":
        st.subheader(f"Technique : {selected_technique}")
        
        # Leçon 1 (Tendances) et Leçon 2 (Moyennes Mobiles)
        if selected_technique in ["Leçon 1 : Les Tendances", "Leçon 2 : Les Moyennes Mobiles"]:
            fig_technique = go.Figure()
            fig_technique.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Prix"))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["SMA_5"], name="SMA 5", line=dict(color="blue")))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20", line=dict(color="orange"))) 

        # Leçon 3 : La MACD (Graphique Principal et Histogramme)
        elif selected_technique == "Leçon 3 : La MACD":
            df["Couleur_MACDH"] = np.where(df["MACDH"] >= 0, 'green', 'red')
            
            fig_technique = make_subplots(rows=2, cols=1, 
                                          row_heights=[0.7, 0.3], 
                                          shared_xaxes=True, 
                                          vertical_spacing=0.05)

            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color='blue')), row=1, col=1)
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal", line=dict(color='red')), row=1, col=1)
            fig_technique.update_yaxes(title_text="MACD / Signal", row=1, col=1)
            fig_technique.update_layout(xaxis_rangeslider_visible=False)

            fig_technique.add_trace(go.Bar(x=df["Date"], y=df["MACDH"], 
                                           name="Histogramme", marker_color=df["Couleur_MACDH"]), row=2, col=1)
            fig_technique.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=2, col=1)
            fig_technique.update_yaxes(title_text="MACDH", row=2, col=1)
            
        # Leçon 4 : Les Bollingers
        elif selected_technique == "Leçon 4 : Les Bollingers":
            fig_technique = go.Figure()
            fig_technique.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Prix"))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], name="BB Upper", line=dict(color="red")))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], name="BB Lower", line=dict(color="green")))

        # Leçon 5 : Le Stochastique
        elif selected_technique == "Leçon 5 : Le Stochastique":
            fig_technique = go.Figure()
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_K"], name="%K"))
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["Stoch_D"], name="%D"))
            fig_technique.add_hline(y=80, line_dash="dash", annotation_text="Surachat", line=dict(color="blue"))
            fig_technique.add_hline(y=20, line_dash="dash", annotation_text="Survente", line=dict(color="orange"))

        # Leçon 6 : Les Chandeliers
        elif selected_technique == "Leçon 6 : Les Chandeliers":
            fig_technique = go.Figure()
            fig_technique.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Bougies"))

        # Leçon 7 : Le RSI
        elif selected_technique == "Leçon 7 : Le RSI":
            fig_technique = go.Figure()
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["RSI_14"], name="RSI 14"))
            fig_technique.add_hline(y=70, line_dash="dash", annotation_text="Surachat", line=dict(color="red"))
            fig_technique.add_hline(y=30, line_dash="dash", annotation_text="Survente", line=dict(color="green"))

        # Leçon 15 : Le Momentum (Affiche Prix + Momentum, conforme à l'exemple)
        elif selected_technique == "Leçon 15 : Le Momentum":
            
            # Déterminer la colonne Momentum et le label en fonction de la période
            if len(df) <= 5:
                mom_col = 'Momentum_1'
                mom_label = 'Momentum (1 jour)'
            elif len(df) <= 10:
                mom_col = 'Momentum_5'
                mom_label = 'Momentum (5 jours)'
            else:
                mom_col = 'Momentum_10'
                mom_label = 'Momentum (10 jours)'
                
            fig_technique = make_subplots(rows=2, cols=1, 
                                          row_heights=[0.7, 0.3], 
                                          shared_xaxes=True, 
                                          vertical_spacing=0.05)

            # Trace 1 (Prix)
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Prix Clôture", line=dict(color='blue')), row=1, col=1)
            
            # Trace 2 (Momentum)
            fig_technique.add_trace(go.Scatter(x=df["Date"], y=df[mom_col], 
                                               name=mom_label, 
                                               line=dict(color='orange')), 
                                    row=2, col=1)
            fig_technique.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=2, col=1) # Ligne zéro du Momentum
            fig_technique.update_yaxes(title_text="Momentum", row=2, col=1)

            fig_technique.update_layout(height=600, xaxis_rangeslider_visible=False) 
        
        # Leçon 9 : Les Volumes (Gardé ici car il était dans la liste d'origine, même s'il est retiré du selectbox principal)
        elif selected_technique in ["Leçon 9 : Les Volumes"]:
            fig_technique = go.Figure()
            fig_technique.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume"))
            
        # Leçon 14 : L'épaule-tête-épaule
        elif selected_technique == "Leçon 14 : L'épaule-tête-épaule":
            fig_technique = go.Figure()
            
            fig_technique.add_trace(go.Scatter(
                x=df["Date"], 
                y=df["Close"], 
                mode="lines", 
                name="Prix de Clôture", 
                line=dict(color='blue')))
            
            lookback = min(len(df), 100) 
            df_subset = df.tail(lookback).copy()
            
            if len(df_subset) >= 2:
                df_subset.loc[:, 'X'] = np.arange(len(df_subset))
                slope, intercept = np.polyfit(df_subset['X'].values, df_subset['Close'].values, 1)
                
                neckline_y = intercept + slope * df_subset['X']
                
                fig_technique.add_trace(go.Scatter(
                    x=df_subset["Date"], 
                    y=neckline_y, 
                    mode="lines", 
                    name="Ligne de Cou (Support Dynamique)", 
                    line=dict(color='red', dash='dash', width=2)
                ))
            
            st.info("NOTE: La Ligne de Cou utilise une Régression Linéaire pour illustrer le support dynamique de la figure ETE.")


        # Afficher le graphique
        st.plotly_chart(fig_technique, use_container_width=True,
                        key=f"technique_{selected_technique}_{selected_company}")
        
    st.dataframe(df.tail(10)) 

st.write("⚠️ Ceci est un outil pédagogique. Ne prend pas de décisions financières sans conseil adapté.")