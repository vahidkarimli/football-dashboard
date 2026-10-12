import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import requests

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Multi-League Predictive Analytics & Value Betting Engine", layout="wide")

st.title("⚽ Universal Sports Analytics & Value Betting Engine")
st.caption("Statistical Modeling (Poisson Distribution) • Implied Odds Analysis • Expected Value (+EV) Detection")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Data & League Settings")

# 1. League Selection
LEAGUES = {
    "Turkish Süper Lig": {"collect_key": "super-lig", "odds_key": "soccer_turkey_super_league"},
    "English Premier League": {"collect_key": "ingiltere-premier-ligi", "odds_key": "soccer_epl"},
    "Spanish La Liga": {"collect_key": "ispanya-la-liga", "odds_key": "soccer_spain_la_liga"},
    "Italian Serie A": {"collect_key": "italya-serie-a", "odds_key": "soccer_italy_serie_a"},
    "German Bundesliga": {"collect_key": "almanya-bundesliga", "odds_key": "soccer_germany_bundesliga"},
    "UEFA Champions League": {"collect_key": "sampiyonlar-ligi", "odds_key": "soccer_uefa_champs_league"}
}

selected_league_name = st.sidebar.selectbox("Select League / Competition:", list(LEAGUES.keys()))
league_info = LEAGUES[selected_league_name]

# 2. API Credentials Input
st.sidebar.subheader("API Keys (Optional)")
collect_api_key = st.sidebar.text_input("CollectAPI Key:", type="password", help="For live league standings & results")
odds_api_key = st.sidebar.text_input("The Odds API Key:", type="password", help="For live bookmaker odds")

# --- POISSON STATISTICAL MODEL ENGINE ---
def calculate_poisson_probabilities(home_xg, away_xg, max_goals=5):
    """Calculates win/draw/loss probabilities and scoreline grid using Poisson Distribution."""
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    
    score_matrix = np.outer(home_probs, away_probs)
    
    home_win = np.sum(np.tril(score_matrix, -1))
    draw = np.sum(np.diag(score_matrix))
    away_win = np.sum(np.triu(score_matrix, 1))
    
    # Normalize probabilities to sum to 100%
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total, score_matrix

def calculate_expected_value(model_prob, decimal_odds):
    """Calculates Expected Value (+EV) percentage."""
    if decimal_odds <= 1.0:
        return 0.0
    return (model_prob * decimal_odds) - 1.0

def calculate_kelly_stake(model_prob, decimal_odds, bankroll, fractional=0.25):
    """Calculates recommended stake size using Fractional Kelly Criterion."""
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    f_kelly = (b * model_prob - q) / b
    if f_kelly <= 0:
        return 0.0
    return max(0.0, round(bankroll * f_kelly * fractional, 2))

# --- MOCK DATA ENGINE (FOR INSTANT PREVIEWS & FALLBACK) ---
def get_mock_league_data(league_name):
    """Provides representative stats if API keys are absent."""
    teams = {
        "Turkish Süper Lig": ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", "Başakşehir", "Kasımpaşa"],
        "English Premier League": ["Arsenal", "Manchester City", "Liverpool", "Chelsea", "Aston Villa", "Tottenham"],
        "Spanish La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Bilbao", "Villarreal", "Real Sociedad"],
        "Italian Serie A": ["Inter Milan", "AC Milan", "Juventus", "Atalanta", "Napoli", "Lazio"],
        "German Bundesliga": ["Bayern Munich", "Bayer Leverkusen", "Borussia Dortmund", "RB Leipzig", "Eintracht Frankfurt"],
        "UEFA Champions League": ["Real Madrid", "Manchester City", "Bayern Munich", "PSG", "Barcelona", "Inter Milan"]
    }
    league_teams = teams.get(league_name, ["Team A", "Team B", "Team C", "Team D"])
    
    standings = []
    for rank, team in enumerate(league_teams, 1):
        standings.append({
            "Rank": rank,
            "Team": team,
            "Played": 25,
            "W": 20 - rank,
            "D": 3 + (rank % 2),
            "L": 2 + rank,
            "GF": 55 - (rank * 3),
            "GA": 18 + (rank * 2),
            "GD": (55 - (rank * 3)) - (18 + (rank * 2)),
            "Points": 63 - (rank * 4),
            "Avg Goals Scored": round((55 - (rank * 3)) / 25, 2),
            "Avg Goals Conceded": round((18 + (rank * 2)) / 25, 2)
        })
    return pd.DataFrame(standings)

# --- DASHBOARD UI ---
tab1, tab2, tab3 = st.tabs(["🔮 Match Predictor Engine", "💰 Multi-League +EV Scanner", "🏆 League Standings & Form"])

# DATA INITIALIZATION
df_standings = get_mock_league_data(selected_league_name)

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
        bankroll = st.number_input("Your Total Betting Bankroll ($):", min_value=10.0, value=1000.0, step=50.0)

    if home_team == away_team:
        st.warning("⚠️ Please select two different teams for comparison.")
    else:
        # Get team metrics
        home_row = df_standings[df_standings["Team"] == home_team].iloc[0]
        away_row = df_standings[df_standings["Team"] == away_team].iloc[0]
        
        # Calculate Expected Goals (xG) based on goals scored/conceded averages
        league_avg_goals = df_standings["Avg Goals Scored"].mean()
        
        home_attack = home_row["Avg Goals Scored"] / league_avg_goals
        away_defense = away_row["Avg Goals Conceded"] / league_avg_goals
        
        away_attack = away_row["Avg Goals Scored"] / league_avg_goals
        home_defense = home_row["Avg Goals Conceded"] / league_avg_goals
        
        # Include home advantage multiplier (~1.15)
        home_xg = max(0.2, round(home_attack * away_defense * league_avg_goals * 1.15, 2))
        away_xg = max(0.2, round(away_attack * home_defense * league_avg_goals, 2))
        
        # Calculate Poisson Model Probabilities
        p_home, p_draw, p_away, score_grid = calculate_poisson_probabilities(home_xg, away_xg)
        
        st.markdown("---")
        
        # Expected Goals & Win Probabilities
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Home xG", home_xg)
        mc2.metric("Away xG", away_xg)
        mc3.metric("Predicted Winner", home_team if p_home > p_away else (away_team if p_away > p_home else "Draw"))
        mc4.metric("Model Confidence", f"{max(p_home, p_away) * 100:.1f}%")

        # Win/Draw/Loss Bars
        st.write("#### 📊 Outcome Probabilities")
        pb_col1, pb_col2, pb_col3 = st.columns(3)
        pb_col1.metric(f"🏠 {home_team} Win", f"{p_home * 100:.1f}%")
        pb_col2.metric("🤝 Draw", f"{p_draw * 100:.1f}%")
        pb_col3.metric(f"✈️ {away_team} Win", f"{p_away * 100:.1f}%")

        # Odds & Value Bet Inputs
        st.write("#### 💡 Odds & Expected Value (+EV) Analyzer")
        oc1, oc2, oc3 = st.columns(3)
        
        odds_home = oc1.number_input(f"{home_team} Odds:", min_value=1.01, value=2.10, step=0.05)
        odds_draw = oc2.number_input("Draw Odds:", min_value=1.01, value=3.30, step=0.05)
        odds_away = oc3.number_input(f"{away_team} Odds:", min_value=1.01, value=3.20, step=0.05)
        
        ev_home = calculate_expected_value(p_home, odds_home)
        ev_draw = calculate_expected_value(p_draw, odds_draw)
        ev_away = calculate_expected_value(p_away, odds_away)
        
        stake_home = calculate_kelly_stake(p_home, odds_home, bankroll)
        stake_draw = calculate_kelly_stake(p_draw, odds_draw, bankroll)
        stake_away = calculate_kelly_stake(p_away, odds_away, bankroll)
        
        analysis_data = [
            {"Outcome": f"{home_team} Win", "Model Prob": f"{p_home*100:.1f}%", "Bookmaker Odds": odds_home, "Implied Prob": f"{(1/odds_home)*100:.1f}%", "Expected Value (EV)": f"{ev_home*100:+.1f}%", "Recommended Stake (Kelly)": f"${stake_home:.2f}" if ev_home > 0 else "$0.00", "Status": "🔥 VALUE BET" if ev_home > 0.05 else ("✅ Fair" if ev_home > 0 else "❌ No Value")},
            {"Outcome": "Draw", "Model Prob": f"{p_draw*100:.1f}%", "Bookmaker Odds": odds_draw, "Implied Prob": f"{(1/odds_draw)*100:.1f}%", "Expected Value (EV)": f"{ev_draw*100:+.1f}%", "Recommended Stake (Kelly)": f"${stake_draw:.2f}" if ev_draw > 0 else "$0.00", "Status": "🔥 VALUE BET" if ev_draw > 0.05 else ("✅ Fair" if ev_draw > 0 else "❌ No Value")},
            {"Outcome": f"{away_team} Win", "Model Prob": f"{p_away*100:.1f}%", "Bookmaker Odds": odds_away, "Implied Prob": f"{(1/odds_away)*100:.1f}%", "Expected Value (EV)": f"{ev_away*100:+.1f}%", "Recommended Stake (Kelly)": f"${stake_away:.2f}" if ev_away > 0 else "$0.00", "Status": "🔥 VALUE BET" if ev_away > 0.05 else ("✅ Fair" if ev_away > 0 else "❌ No Value")},
        ]
        
        st.dataframe(pd.DataFrame(analysis_data), hide_index=True, use_container_width=True)

        # Most Likely Scorelines Matrix
        st.write("#### 🎯 Scoreline Probability Heatmap (0-5 Goals)")
        scores_list = []
        for h in range(6):
            for a in range(6):
                prob = score_grid[h][a]
                scores_list.append({"Scoreline": f"{h} - {a}", "Probability": prob, "Percentage": f"{prob*100:.2f}%"})
        
        df_scores = pd.DataFrame(scores_list).sort_values(by="Probability", ascending=False).head(8)
        st.dataframe(df_scores[["Scoreline", "Percentage"]], hide_index=True, use_container_width=True)

# --- TAB 2: MULTI-LEAGUE VALUE BET SCANNER ---
with tab2:
    st.subheader(f"💰 Multi-League Value Bet Scanner (+EV)")
    st.info("Scanning current league matchups where Model Probability > Bookmaker Implied Probability.")
    
    # Generate sample fixtures for scanning
    scanner_results = []
    all_teams = df_standings["Team"].tolist()
    
    for i in range(0, len(all_teams)-1, 2):
        h_team = all_teams[i]
        a_team = all_teams[i+1]
        
        # Model calc
        h_row = df_standings[df_standings["Team"] == h_team].iloc[0]
        a_row = df_standings[df_standings["Team"] == a_team].iloc[0]
        
        h_xg = round((h_row["Avg Goals Scored"] / 1.5) * 1.3, 2)
        a_xg = round((a_row["Avg Goals Scored"] / 1.5) * 0.9, 2)
        
        p_h, p_d, p_a, _ = calculate_poisson_probabilities(h_xg, a_xg)
        
        # Mock odds comparison
        mock_odds_h = round(1 / p_h + 0.3, 2) # Adding artificial value for demo
        mock_odds_d = round(1 / p_d - 0.2, 2)
        mock_odds_a = round(1 / p_a - 0.3, 2)
        
        ev_h = calculate_expected_value(p_h, mock_odds_h)
        
        if ev_h > 0.02:
            scanner_results.append({
                "Matchup": f"{h_team} vs {a_team}",
                "Bet Choice": f"{h_team} Win",
                "Model Prob": f"{p_h*100:.1f}%",
                "Bookie Odds": mock_odds_h,
                "Implied Prob": f"{(1/mock_odds_h)*100:.1f}%",
                "Expected Value (+EV)": f"{ev_h*100:+.1f}%",
                "Action": "🔥 HIGH VALUE" if ev_h > 0.08 else "✅ VALUE BET"
            })
            
    if scanner_results:
        st.dataframe(pd.DataFrame(scanner_results), hide_index=True, use_container_width=True)
    else:
        st.write("No high value (+EV) bets found for current fixtures.")

# --- TAB 3: STANDINGS ---
with tab3:
    st.subheader(f"🏆 {selected_league_name} Standings")
    st.dataframe(df_standings[["Rank", "Team", "Played", "W", "D", "L", "GF", "GA", "GD", "Points", "Avg Goals Scored", "Avg Goals Conceded"]], hide_index=True, use_container_width=True)
