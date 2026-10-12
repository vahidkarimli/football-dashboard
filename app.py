import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import requests

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Universal Live Sports Analytics & Value Betting Engine", layout="wide")

st.title("⚽ Universal Live Sports Analytics & Value Betting Engine")
st.caption("Poisson Statistical Modeling • Real-Time Odds API Integration • Expected Value (+EV) Scanner")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Data & League Settings")

# League Mappings for CollectAPI and The Odds API
LEAGUES = {
    "Turkish Süper Lig": {
        "collect_key": "super-lig", 
        "odds_key": "soccer_turkey_super_league"
    },
    "English Premier League": {
        "collect_key": "ingiltere-premier-ligi", 
        "odds_key": "soccer_epl"
    },
    "Spanish La Liga": {
        "collect_key": "ispanya-la-liga", 
        "odds_key": "soccer_spain_la_liga"
    },
    "Italian Serie A": {
        "collect_key": "italya-serie-a", 
        "odds_key": "soccer_italy_serie_a"
    },
    "German Bundesliga": {
        "collect_key": "almanya-bundesliga", 
        "odds_key": "soccer_germany_bundesliga"
    },
    "UEFA Champions League": {
        "collect_key": "sampiyonlar-ligi", 
        "odds_key": "soccer_uefa_champs_league"
    }
}

selected_league_name = st.sidebar.selectbox("Select League / Competition:", list(LEAGUES.keys()))
league_info = LEAGUES[selected_league_name]

# API Credentials Input
st.sidebar.subheader("🔑 Live API Keys")
collect_api_key = st.sidebar.text_input("CollectAPI Key:", type="password", help="For live league standings")
odds_api_key = st.sidebar.text_input("The Odds API Key:", type="password", help="For live bookmaker odds (the-odds-api.com)")

# --- LIVE DATA FETCHING FUNCTIONS ---
@st.cache_data(ttl=600)
def fetch_live_standings(api_key, league_slug):
    """Fetches real standings from CollectAPI."""
    if not api_key:
        return None
    url = f"https://api.collectapi.com/football/league?league={league_slug}"
    headers = {"authorization": f"apikey {api_key}", "content-type": "application/json"}
    try:
        r = requests.get(url, headers=headers)
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data
        elif isinstance(data, dict) and data.get("success") and data.get("result"):
            return data.get("result")
    except Exception:
        pass
    return None

@st.cache_data(ttl=300)
def fetch_live_odds(api_key, sport_key):
    """Fetches real-time upcoming match odds from The Odds API."""
    if not api_key:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# --- STATISTICAL MODEL ENGINE ---
def calculate_poisson_probabilities(home_xg, away_xg, max_goals=5):
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    
    score_matrix = np.outer(home_probs, away_probs)
    
    home_win = np.sum(np.tril(score_matrix, -1))
    draw = np.sum(np.diag(score_matrix))
    away_win = np.sum(np.triu(score_matrix, 1))
    
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total, score_matrix

def calculate_expected_value(model_prob, decimal_odds):
    if decimal_odds <= 1.0:
        return 0.0
    return (model_prob * decimal_odds) - 1.0

def calculate_kelly_stake(model_prob, decimal_odds, bankroll, fractional=0.25):
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    f_kelly = (b * model_prob - q) / b
    if f_kelly <= 0:
        return 0.0
    return max(0.0, round(bankroll * f_kelly * fractional, 2))

# --- DATA PROCESSOR (LIVE OR MOCK FALLBACK) ---
raw_standings = fetch_live_standings(collect_api_key, league_info["collect_key"])

if raw_standings:
    # Process live CollectAPI data into a clean DataFrame
    processed = []
    for item in raw_standings:
        played = int(item.get("play", item.get("played", 20)))
        gf = int(item.get("goalfor", item.get("gf", 30)))
        ga = int(item.get("goalagainst", item.get("ga", 25)))
        processed.append({
            "Rank": int(item.get("rank", 1)),
            "Team": item.get("team", item.get("teamName", "Unknown")),
            "Played": played,
            "W": int(item.get("win", 10)),
            "D": int(item.get("draw", 5)),
            "L": int(item.get("lose", 5)),
            "GF": gf,
            "GA": ga,
            "GD": int(item.get("goaldistance", gf - ga)),
            "Points": int(item.get("point", 30)),
            "Avg Goals Scored": round(gf / max(1, played), 2),
            "Avg Goals Conceded": round(ga / max(1, played), 2)
        })
    df_standings = pd.DataFrame(processed).sort_values(by="Rank")
else:
    # Fallback Mock Engine if no CollectAPI key is provided
    teams_dict = {
        "Turkish Süper Lig": ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", "Başakşehir", "Kasımpaşa", "Samsunspor", "Göztepe"],
        "English Premier League": ["Arsenal", "Manchester City", "Liverpool", "Chelsea", "Aston Villa", "Tottenham", "Newcastle", "Brighton"],
        "Spanish La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Bilbao", "Villarreal", "Real Sociedad", "Real Betis", "Valencia"],
        "Italian Serie A": ["Inter Milan", "AC Milan", "Juventus", "Atalanta", "Napoli", "Lazio", "Fiorentina", "Roma"],
        "German Bundesliga": ["Bayern Munich", "Bayer Leverkusen", "Borussia Dortmund", "RB Leipzig", "Eintracht Frankfurt", "Stuttgart"],
        "UEFA Champions League": ["Real Madrid", "Manchester City", "Bayern Munich", "PSG", "Barcelona", "Inter Milan", "Arsenal", "Leverkusen"]
    }
    league_teams = teams_dict.get(selected_league_name, ["Team A", "Team B", "Team C", "Team D"])
    standings = []
    for rank, team in enumerate(league_teams, 1):
        standings.append({
            "Rank": rank, "Team": team, "Played": 25, "W": 18 - rank, "D": 4, "L": 3 + rank,
            "GF": 50 - (rank * 2), "GA": 20 + rank, "GD": (50 - (rank * 2)) - (20 + rank),
            "Points": 58 - (rank * 3), "Avg Goals Scored": round((50 - (rank * 2)) / 25, 2),
            "Avg Goals Conceded": round((20 + rank) / 25, 2)
        })
    df_standings = pd.DataFrame(standings)
    if not collect_api_key:
        st.sidebar.info("💡 Tip: Enter a CollectAPI key to load live team standings.")

# Fetch live odds data
live_odds_data = fetch_live_odds(odds_api_key, league_info["odds_key"])

# --- DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs(["🔮 Match Predictor Engine", "💰 Live Value Bet (+EV) Scanner", "🏆 League Standings"])

# --- TAB 1: MATCH PREDICTOR ---
with tab1:
    st.subheader(f"🔮 Match Predictor — {selected_league_name}")
    
    col1, col2, col3 = st.columns(3)
    teams = df_standings["Team"].tolist()
    with col1:
        home_team = st.selectbox("Home Team:", teams, index=0)
    with col2:
        away_team = st.selectbox("Away Team:", teams, index=1 if len(teams) > 1 else 0)
    with col3:
        bankroll = st.number_input("Your Total Bankroll ($):", min_value=10.0, value=1000.0, step=50.0)

    if home_team == away_team:
        st.warning("⚠️ Please select two different teams.")
    else:
        home_row = df_standings[df_standings["Team"] == home_team].iloc[0]
        away_row = df_standings[df_standings["Team"] == away_team].iloc[0]
        
        league_avg_goals = df_standings["Avg Goals Scored"].mean()
        
        home_attack = home_row["Avg Goals Scored"] / league_avg_goals
        away_defense = away_row["Avg Goals Conceded"] / league_avg_goals
        
        away_attack = away_row["Avg Goals Scored"] / league_avg_goals
        home_defense = home_row["Avg Goals Conceded"] / league_avg_goals
        
        home_xg = max(0.2, round(home_attack * away_defense * league_avg_goals * 1.15, 2))
        away_xg = max(0.2, round(away_attack * home_defense * league_avg_goals, 2))
        
        p_home, p_draw, p_away, score_grid = calculate_poisson_probabilities(home_xg, away_xg)
        
        st.markdown("---")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Home xG", home_xg)
        mc2.metric("Away xG", away_xg)
        mc3.metric("Predicted Winner", home_team if p_home > p_away else (away_team if p_away > p_home else "Draw"))
        mc4.metric("Model Confidence", f"{max(p_home, p_away) * 100:.1f}%")

        st.write("#### 📊 Outcome Probabilities")
        pb1, pb2, pb3 = st.columns(3)
        pb1.metric(f"🏠 {home_team} Win", f"{p_home * 100:.1f}%")
        pb2.metric("🤝 Draw", f"{p_draw * 100:.1f}%")
        pb3.metric(f"✈️ {away_team} Win", f"{p_away * 100:.1f}%")

        # Check if live odds exist for this specific match
        default_home_odds, default_draw_odds, default_away_odds = 2.10, 3.40, 3.20
        match_live_found = False
        
        if live_odds_data:
            for game in live_odds_data:
                h_name = game.get("home_team", "")
                a_name = game.get("away_team", "")
                if home_team.lower() in h_name.lower() or away_team.lower() in a_name.lower():
                    bookmakers = game.get("bookmakers", [])
                    if bookmakers:
                        markets = bookmakers[0].get("markets", [])
                        for m in markets:
                            if m.get("key") == "h2h":
                                outcomes = m.get("outcomes", [])
                                for o in outcomes:
                                    name = o.get("name")
                                    price = o.get("price")
                                    if home_team.lower() in name.lower():
                                        default_home_odds = price
                                        match_live_found = True
                                    elif away_team.lower() in name.lower():
                                        default_away_odds = price
                                        match_live_found = True
                                    elif "draw" in name.lower():
                                        default_draw_odds = price

        if match_live_found:
            st.success("🟢 Live Bookmaker Odds successfully loaded from The Odds API!")
        else:
            st.info("ℹ️ Using standard odds inputs (Enter your Odds API key in sidebar for automated live lines).")

        st.write("#### 💡 Odds & Expected Value (+EV) Analyzer")
        oc1, oc2, oc3 = st.columns(3)
        odds_home = oc1.number_input(f"{home_team} Odds:", min_value=1.01, value=float(default_home_odds), step=0.05)
        odds_draw = oc2.number_input("Draw Odds:", min_value=1.01, value=float(default_draw_odds), step=0.05)
        odds_away = oc3.number_input(f"{away_team} Odds:", min_value=1.01, value=float(default_away_odds), step=0.05)
        
        ev_home = calculate_expected_value(p_home, odds_home)
        ev_draw = calculate_expected_value(p_draw, odds_draw)
        ev_away = calculate_expected_value(p_away, odds_away)
        
        stake_home = calculate_kelly_stake(p_home, odds_home, bankroll)
        stake_draw = calculate_kelly_stake(p_draw, odds_draw, bankroll)
        stake_away = calculate_kelly_stake(p_away, odds_away, bankroll)
        
        analysis_data = [
            {"Outcome": f"{home_team} Win", "Model Prob": f"{p_home*100:.1f}%", "Bookie Odds": odds_home, "Implied Prob": f"{(1/odds_home)*100:.1f}%", "Expected Value (EV)": f"{ev_home*100:+.1f}%", "Kelly Stake": f"${stake_home:.2f}" if ev_home > 0 else "$0.00", "Status": "🔥 VALUE BET" if ev_home > 0.05 else ("✅ Fair" if ev_home > 0 else "❌ No Value")},
            {"Outcome": "Draw", "Model Prob": f"{p_draw*100:.1f}%", "Bookie Odds": odds_draw, "Implied Prob": f"{(1/odds_draw)*100:.1f}%", "Expected Value (EV)": f"{ev_draw*100:+.1f}%", "Kelly Stake": f"${stake_draw:.2f}" if ev_draw > 0 else "$0.00", "Status": "🔥 VALUE BET" if ev_draw > 0.05 else ("✅ Fair" if ev_draw > 0 else "❌ No Value")},
            {"Outcome": f"{away_team} Win", "Model Prob": f"{p_away*100:.1f}%", "Bookie Odds": odds_away, "Implied Prob": f"{(1/odds_away)*100:.1f}%", "Expected Value (EV)": f"{ev_away*100:+.1f}%", "Kelly Stake": f"${stake_away:.2f}" if ev_away > 0 else "$0.00", "Status": "🔥 VALUE BET" if ev_away > 0.05 else ("✅ Fair" if ev_away > 0 else "❌ No Value")},
        ]
        st.dataframe(pd.DataFrame(analysis_data), hide_index=True, use_container_width=True)

        st.write("#### 🎯 Scoreline Probability Matrix")
        scores_list = []
        for h in range(6):
            for a in range(6):
                prob = score_grid[h][a]
                scores_list.append({"Scoreline": f"{h} - {a}", "Probability": prob, "Percentage": f"{prob*100:.2f}%"})
        df_scores = pd.DataFrame(scores_list).sort_values(by="Probability", ascending=False).head(8)
        st.dataframe(df_scores[["Scoreline", "Percentage"]], hide_index=True, use_container_width=True)

# --- TAB 2: LIVE VALUE BET SCANNER ---
with tab2:
    st.subheader(f"💰 Live Value Bet Scanner (+EV) — {selected_league_name}")
    
    if live_odds_data:
        st.success(f"Scanning {len(live_odds_data)} live fixtures fetched from The Odds API...")
        scanned_fixtures = []
        
        for game in live_odds_data:
            h_team = game.get("home_team")
            a_team = game.get("away_team")
            bookmakers = game.get("bookmakers", [])
            
            if not bookmakers:
                continue
            
            # Extract odds from first available bookmaker
            markets = bookmakers[0].get("markets", [])
            h_odds, d_odds, a_odds = 2.0, 3.3, 3.0
            for m in markets:
                if m.get("key") == "h2h":
                    for o in m.get("outcomes", []):
                        name = o.get("name")
                        price = o.get("price")
                        if name == h_team:
                            h_odds = price
                        elif name == a_team:
                            a_odds = price
                        elif "draw" in name.lower():
                            d_odds = price
            
            # Simple model approximation based on table standings if team names match
            h_match = df_standings[df_standings["Team"].str.contains(h_team[:4], case=False, na=False)]
            a_match = df_standings[df_standings["Team"].str.contains(a_team[:4], case=False, na=False)]
            
            if not h_match.empty and not a_match.empty:
                h_xg = h_match.iloc[0]["Avg Goals Scored"] * 1.
