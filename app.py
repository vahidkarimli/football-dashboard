import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import requests
from dataclasses import dataclass
from typing import Optional

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Universal Live Sports Analytics & Context-Aware +EV Engine", layout="wide")

st.title("⚽ Universal Live Sports Analytics & Context-Aware Engine")
st.caption("Poisson Modeling • Match Context & Lineups • Live Odds API • Multi-Market +EV Scanner")

# --- MATCH CONTEXT DATA MODEL ---
@dataclass
class MatchContext:
    home_lineup_confirmed: bool = False
    away_lineup_confirmed: bool = False
    home_missing_players: int = 0
    away_missing_players: int = 0
    home_rest_days: Optional[int] = 7
    away_rest_days: Optional[int] = 7
    home_manager_changed: bool = False
    away_manager_changed: bool = False
    context_data_quality: float = 1.0  # 1.0 = High quality, lower = missing data

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Data & League Settings")

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

st.sidebar.subheader("🔑 Live API Credentials")
collect_api_key = st.sidebar.text_input("CollectAPI Key:", type="password", help="For live league standings")
odds_api_key = st.sidebar.text_input("The Odds API Key:", type="password", help="For live bookmaker odds (the-odds-api.com)")

# --- API DATA FETCHING WITH TIMEOUTS & RETRIES ---
@st.cache_data(ttl=600)
def fetch_live_standings(api_key, league_slug):
    if not api_key:
        return None, "Demo Data (No API Key Provided)"
    url = f"https://api.collectapi.com/football/league?league={league_slug}"
    headers = {"authorization": f"apikey {api_key}", "content-type": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data, "Live CollectAPI Feed"
        elif isinstance(data, dict) and data.get("success") and data.get("result"):
            return data.get("result"), "Live CollectAPI Feed"
    except Exception as e:
        return None, f"API Error: {str(e)}"
    return None, "Fallback Demo Data"

@st.cache_data(ttl=300)
def fetch_live_odds(api_key, sport_key):
    if not api_key:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {"apiKey": api_key, "regions": "eu,uk", "markets": "h2h", "oddsFormat": "decimal"}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# --- STATISTICAL MODEL & MARKET ENGINES ---
def calculate_poisson_probabilities(home_xg, away_xg, max_goals=10):
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    score_matrix = np.outer(home_probs, away_probs)
    
    home_win = np.sum(np.tril(score_matrix, -1))
    draw = np.sum(np.diag(score_matrix))
    away_win = np.sum(np.triu(score_matrix, 1))
    
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total, score_matrix

def calculate_extended_markets(score_matrix, home_xg, away_xg):
    max_goals = score_matrix.shape[0] - 1
    btts_yes = 0.0
    over_2_5 = 0.0
    clean_sheet_home = 0.0
    clean_sheet_away = 0.0
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = score_matrix[h][a]
            if h > 0 and a > 0:
                btts_yes += prob
            if (h + a) > 2.5:
                over_2_5 += prob
            if a == 0:
                clean_sheet_home += prob
            if h == 0:
                clean_sheet_away += prob

    btts_no = 1.0 - btts_yes
    under_2_5 = 1.0 - over_2_5
    
    est_home_corners = round(home_xg * 2.8 + 2.1, 1)
    est_away_corners = round(away_xg * 2.5 + 1.8, 1)
    
    return {
        "BTTS Yes": round(btts_yes, 3),
        "BTTS No": round(btts_no, 3),
        "Over 2.5 Goals": round(over_2_5, 3),
        "Under 2.5 Goals": round(under_2_5, 3),
        "Home Clean Sheet": round(clean_sheet_home, 3),
        "Away Clean Sheet": round(clean_sheet_away, 3),
        "Est. Total Corners": round(est_home_corners + est_away_corners, 1),
        "Est. Home SOT": round(home_xg * 1.6 + 1.2, 1),
        "Est. Away SOT": round(away_xg * 1.5 + 1.0, 1)
    }

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

# --- DATA PROCESSOR (LIVE OR FALLBACK) ---
raw_standings, data_status = fetch_live_standings(collect_api_key, league_info["collect_key"])

if raw_standings:
    processed = []
    for item in raw_standings:
        played = int(item.get("play", item.get("played", 20)))
        gf = int(item.get("goalfor", item.get("gf", 30)))
        ga = int(item.get("goalagainst", item.get("ga", 25)))
        processed.append({
            "Rank": int(item.get("rank", 1)),
            "Team": item.get("team", item.get("teamName", "Unknown")),
            "Played": played, "W": int(item.get("win", 10)), "D": int(item.get("draw", 5)), "L": int(item.get("lose", 5)),
            "GF": gf, "GA": ga, "GD": int(item.get("goaldistance", gf - ga)), "Points": int(item.get("point", 30)),
            "Avg Goals Scored": round(gf / max(1, played), 2), "Avg Goals Conceded": round(ga / max(1, played), 2)
        })
    df_standings = pd.DataFrame(processed).sort_values(by="Rank")
else:
    teams_dict = {
        "Turkish Süper Lig": ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", "Başakşehir", "Kasımpaşa"],
        "English Premier League": ["Arsenal", "Manchester City", "Liverpool", "Chelsea", "Aston Villa", "Tottenham"],
        "Spanish La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Bilbao", "Villarreal", "Real Sociedad"],
        "Italian Serie A": ["Inter Milan", "AC Milan", "Juventus", "Atalanta", "Napoli", "Lazio"],
        "German Bundesliga": ["Bayern Munich", "Bayer Leverkusen", "Borussia Dortmund", "RB Leipzig", "Eintracht Frankfurt"],
        "UEFA Champions League": ["Real Madrid", "Manchester City", "Bayern Munich", "PSG", "Barcelona", "Inter Milan"]
    }
    league_teams = teams_dict.get(selected_league_name, ["Team A", "Team B"])
    standings = []
    for rank, team in enumerate(league_teams, 1):
        standings.append({
            "Rank": rank, "Team": team, "Played": 25, "W": 18 - rank, "D": 4, "L": 3 + rank,
            "GF": 50 - (rank * 2), "GA": 20 + rank, "GD": (50 - (rank * 2)) - (20 + rank),
            "Points": 58 - (rank * 3), "Avg Goals Scored": round((50 - (rank * 2)) / 25, 2),
            "Avg Goals Conceded": round((20 + rank) / 25, 2)
        })
    df_standings = pd.DataFrame(standings)

live_odds_data = fetch_live_odds(odds_api_key, league_info["odds_key"])

# Data Reliability Banner
if "Live" in data_status:
    st.success(f"🟢 Active Data Source: {data_status}")
else:
    st.info(f"ℹ️ Active Data Source: {data_status}")

# --- DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Match Predictor & Context", "💰 Live +EV Scanner", "🏆 Standings", "📖 Methodology & Model Info"])

# --- TAB 1: MATCH PREDICTOR & CONTEXT ---
with tab1:
    st.subheader(f"🔮 Match Predictor & Context Analysis — {selected_league_name}")
    
    col1, col2, col3 = st.columns(3)
    teams = df_standings["Team"].tolist()
    with col1:
        home_team = st.selectbox("Home Team:", teams, index=0)
    with col2:
        away_team = st.selectbox("Away Team:", teams, index=1 if len(teams) > 1 else 0)
    with col3:
        bankroll = st.number_input("Your Bankroll ($):", min_value=10.0, value=1000.0, step=50.0)

    if home_team == away_team:
        st.warning("⚠️ Select two different teams.")
    else:
        # Context adjustment input panel
        st.write("#### 🛡️ Match Context & Lineup Adjustments")
        cc1, cc2, cc3, cc4 = st.columns(4)
        home_missing = cc1.number_input(f"{home_team} Missing Key Players:", min_value=0, max_value=8, value=0)
        away_missing = cc2.number_input(f"{away_team} Missing Key Players:", min_value=0, max_value=8, value=0)
        home_rest = cc3.number_input(f"{home_team} Rest Days:", min_value=1, max_value=14, value=7)
        away_rest = cc4.number_input(f"{away_team} Rest Days:", min_value=1, max_value=14, value=7)
        
        ctx = MatchContext(
            home_missing_players=home_missing,
            away_missing_players=away_missing,
            home_rest_days=home_rest,
            away_rest_days=away_rest
        )

        # Statistical calculations
        home_row = df_standings[df_standings["Team"] == home_team].iloc[0]
        away_row = df_standings[df_standings["Team"] == away_team].iloc[0]
        league_avg_goals = df_standings["Avg Goals Scored"].mean()
        
        home_attack = home_row["Avg Goals Scored"] / league_avg_goals
        away_defense = away_row["Avg Goals Conceded"] / league_avg_goals
        away_attack = away_row["Avg Goals Scored"] / league_avg_goals
        home_defense = home_row["Avg Goals Conceded"] / league_avg_goals
        
        # Base xG
        base_home_xg = max(0.2, home_attack * away_defense * league_avg_goals * 1.15)
        base_away_xg = max(0.2, away_attack * home_defense * league_avg_goals)
        
        # Adjust xG based on MatchContext (Missing players reduce xG / Fatigue penalty for rest < 4 days)
        home_xg_adj = base_home_xg * (1.0 - (ctx.home_missing_players * 0.05)) * (0.92 if ctx.home_rest_days < 4 else 1.0)
        away_xg_adj = base_away_xg * (1.0 - (ctx.away_missing_players * 0.05)) * (0.92 if ctx.away_rest_days < 4 else 1.0)
        
        home_xg = round(max(0.2, home_xg_adj), 2)
        away_xg = round(max(0.2, away_xg_adj), 2)
        
        p_home, p_draw, p_away, score_grid = calculate_poisson_probabilities(home_xg, away_xg)
        extended_markets = calculate_extended_markets(score_grid, home_xg, away_xg)
        
        st.markdown("---")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Context Home xG", home_xg, delta=round(home_xg - base_home_xg, 2))
        mc2.metric("Context Away xG", away_xg, delta=round(away_xg - base_away_xg, 2))
        mc3.metric("Predicted Winner", home_team if p_home > p_away else (away_team if p_away > p_home else "Draw"))
        mc4.metric("Model Confidence", f"{max(p_home, p_away) * 100:.1f}%")

        # 1X2 Market & Odds
        st.write("#### 📊 Match Winner (1X2) & Value Bet Analysis")
        default_home_odds, default_draw_odds, default_away_odds = 2.10, 3.40, 3.20
        
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
            {"Outcome": f"{home_team} Win", "Model Prob": f"{p_home*100:.1f}%", "Odds": odds_home, "EV": f"{ev_home*100:+.1f}%", "Kelly Stake": f"${stake_home:.2f}" if ev_home > 0 else "$0.00", "Status": "🔥 VALUE" if ev_home > 0.05 else ("✅ Fair" if ev_home > 0 else "❌ None")},
            {"Outcome": "Draw", "Model Prob": f"{p_draw*100:.1f}%", "Odds": odds_draw, "EV": f"{ev_draw*100:+.1f}%", "Kelly Stake": f"${stake_draw:.2f}" if ev_draw > 0 else "$0.00", "Status": "🔥 VALUE" if ev_draw > 0.05 else ("✅ Fair" if ev_draw > 0 else "❌ None")},
            {"Outcome": f"{away_team} Win", "Model Prob": f"{p_away*100:.1f}%", "Odds": odds_away, "EV": f"{ev_away*100:+.1f}%", "Kelly Stake": f"${stake_away:.2f}" if ev_away > 0 else "$0.00", "Status": "🔥 VALUE" if ev_away > 0.05 else ("✅ Fair" if ev_away > 0 else "❌ None")}
        ]
        st.dataframe(pd.DataFrame(analysis_data), hide_index=True, use_container_width=True)

        # Extended Markets Table
        st.write("#### 🎯 Additional Market Projections")
        markets_data = [
            {"Market": "BTTS", "Selection": "Yes", "Probability": f"{extended_markets['BTTS Yes']*100:.1f}%", "Rating": "🔥 Strong" if extended_markets['BTTS Yes'] > 0.55 else "Standard"},
            {"Market": "BTTS", "Selection": "No", "Probability": f"{extended_markets['BTTS No']*100:.1f}%", "Rating": "🔥 Strong" if extended_markets['BTTS No'] > 0.55 else "Standard"},
            {"Market": "Goals", "Selection": "Over 2.5 Goals", "Probability": f"{extended_markets['Over 2.5 Goals']*100:.1f}%", "Rating": "🔥 Strong" if extended_markets['Over 2.5 Goals'] > 0.52 else "Standard"},
            {"Market": "Clean Sheet", "Selection": f"{home_team} Clean Sheet", "Probability": f"{extended_markets['Home Clean Sheet']*100:.1f}%", "Rating": "Defensive Model"},
            {"Market": "Corners", "Selection": f"Est Total: {extended_markets['Est. Total Corners']}", "Probability": "Model Est", "Rating": "Stats Prop"},
            {"Market": "Shots on Target", "Selection": f"{home_team}: {extended_markets['Est. Home SOT']} | {away_team}: {extended_markets['Est. Away SOT']}", "Probability": "Model Est", "Rating": "Player Prop"}
        ]
        st.dataframe(pd.DataFrame(markets_data), hide_index=True, use_container_width=True)

        # Top Scorelines
        st.write("#### 🎯 Most Likely Scorelines (0–10 Matrix)")
        scores_list = []
        for h in range(7):
            for a in range(7):
                prob = score_grid[h][a]
                scores_list.append({"Scoreline": f"{h} - {a}", "Probability": prob, "Percentage": f"{prob*100:.2f}%"})
        df_scores = pd.DataFrame(scores_list).sort_values(by="Probability", ascending=False).head(6)
        st.dataframe(df_scores[["Scoreline", "Percentage"]], hide_index=True, use_container_width=True)

# --- TAB 2: LIVE VALUE BET SCANNER ---
with tab2:
    st.subheader(f"💰 Live Value Bet Scanner (+EV) — {selected_league_name}")
    if live_odds_data:
        st.success(f"Scanning {len(live_odds_data)} live fixtures from The Odds API...")
        scanned_fixtures = []
        for game in live_odds_data:
            h_team, a_team = game.get("home_team"), game.get("away_team")
            bookmakers = game.get("bookmakers", [])
            if not bookmakers: 
                continue
            
            markets = bookmakers[0].get("markets", [])
            h_odds, a_odds = 2.0, 3.0
            for m in markets:
                if m.get("key") == "h2h":
                    for o in m.get("outcomes", []):
                        if o.get("name") == h_team: h_odds = o.get("price")
                        elif o.get("name") == a_team: a_odds = o.get("price")
            
            h_xg, a_xg = 1.4, 1.1
            p_h, p_d, p_a, _ = calculate_poisson_probabilities(h_xg, a_xg)
            ev_h = calculate_expected_value(p_h, h_odds)
            
            if ev_h > 0.03:
                scanned_fixtures.append({"Matchup": f"{h_team} vs {a_team}", "Selection": f"{h_team} Win", "Model Prob": f"{p_h*100:.1f}%", "Odds": h_odds, "Expected Value (+EV)": f"{ev_h*100:+.1f}%", "Rating": "🔥 HIGH VALUE"})
                
        if scanned_fixtures:
            st.dataframe(pd.DataFrame(scanned_fixtures), hide_index=True, use_container_width=True)
        else:
            st.info("No +EV opportunities found in current live feed.")
    else:
        st.info("ℹ️ Enter **The Odds API Key** in the sidebar to enable automated live odds scanning across fixtures.")

# --- TAB 3: STANDINGS ---
with tab3:
    st.subheader(f"🏆 {selected_league_name} Standings")
    st.dataframe(df_standings[["Rank", "Team", "Played", "W", "D", "L", "GF", "GA", "GD", "Points"]], hide_index=True, use_container_width=True)

# --- TAB 4: METHODOLOGY ---
with tab4:
    st.subheader("📖 Model Assumptions, Scope & Methodology")
    st.markdown("""
    * **Poisson Modeling**: Match outcome probabilities are calculated using independent Poisson distributions for home and away goal generation.
    * **Context Layer (`MatchContext`)**: Base Expected Goals (xG) are dynamically adjusted based on starting player availability (missing key starters reduce xG by ~5% per player) and rest day fatigue (rest < 4 days applies a 8% xG penalty).
    * **Value Scanner (+EV)**: Expected Value is calculated as $EV = (P_{\text{model}} \times \text{Odds}) - 1$. Positive EV bets are highlighted when $EV > 3\%$.
    * **Stake Management**: Kelly Criterion bets use a fractional sizing model ($0.25 \times \text{Full Kelly}$) to preserve bankroll during variance.
    """)from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson

st.set_page_config(
    page_title="Universal Live Football Analytics",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Universal Live Football Analytics")
st.caption(
    "Match probabilities · Recent form · Team context · Data provenance · Model uncertainty"
)
st.info(
    "This dashboard is for football analysis and statistical projections. It does not provide betting picks or staking advice. "
    "Predictions are estimates, not guarantees."
)

LEAGUES: dict[str, dict[str, str | None]] = {
    "Turkish Süper Lig": {"collect_key": "super-lig", "football_data_code": None},
    "English Premier League": {"collect_key": "ingiltere-premier-ligi", "football_data_code": "PL"},
    "Spanish La Liga": {"collect_key": "ispanya-la-liga", "football_data_code": "PD"},
    "Italian Serie A": {"collect_key": "italya-serie-a", "football_data_code": "SA"},
    "German Bundesliga": {"collect_key": "almanya-bundesliga", "football_data_code": "BL1"},
    "UEFA Champions League": {"collect_key": "sampiyonlar-ligi", "football_data_code": "CL"},
}

MOCK_TEAMS: dict[str, list[str]] = {
    "Turkish Süper Lig": ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", "Başakşehir", "Kasımpaşa"],
    "English Premier League": ["Arsenal", "Manchester City", "Liverpool", "Chelsea", "Aston Villa", "Tottenham"],
    "Spanish La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Bilbao", "Villarreal", "Real Sociedad"],
    "Italian Serie A": ["Inter Milan", "AC Milan", "Juventus", "Atalanta", "Napoli", "Lazio"],
    "German Bundesliga": ["Bayern Munich", "Bayer Leverkusen", "Borussia Dortmund", "RB Leipzig", "Eintracht Frankfurt"],
    "UEFA Champions League": ["Real Madrid", "Manchester City", "Bayern Munich", "PSG", "Barcelona", "Inter Milan"],
}


def safe_int(item: dict[str, Any], keys: tuple[str, ...], default: int = 0) -> int:
    for key in keys:
        value = item.get(key)
        try:
            if value is not None and value != "":
                return int(float(value))
        except (TypeError, ValueError):
            continue
    return default


@st.cache_data(ttl=600, show_spinner=False)
def fetch_collectapi_standings(api_key: str, league_slug: str) -> tuple[list[dict[str, Any]], str]:
    if not api_key:
        return [], "No CollectAPI key supplied"
    url = "https://api.collectapi.com/football/league"
    headers = {"authorization": f"apikey {api_key}", "content-type": "application/json"}
    try:
        response = requests.get(url, params={"league": league_slug}, headers=headers, timeout=12)
        if response.status_code != 200:
            return [], f"CollectAPI returned HTTP {response.status_code}"
        payload = response.json()
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("result") or payload.get("data") or []
        else:
            rows = []
        if not isinstance(rows, list) or not rows:
            return [], "CollectAPI response contained no standings rows"
        return rows, "Live CollectAPI data"
    except (requests.RequestException, ValueError) as exc:
        return [], f"CollectAPI request failed: {type(exc).__name__}"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_recent_matches(token: str, competition_code: str) -> tuple[list[dict[str, Any]], str]:
    if not token:
        return [], "No Football-Data.org token supplied"
    if not competition_code:
        return [], "This competition is not configured for Football-Data.org"
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": token}
    try:
        response = requests.get(url, headers=headers, params={"status": "FINISHED"}, timeout=15)
        if response.status_code != 200:
            return [], f"Football-Data.org returned HTTP {response.status_code}"
        payload = response.json()
        matches = payload.get("matches", []) if isinstance(payload, dict) else []
        return (matches, "Live Football-Data.org results") if matches else ([], "No finished matches returned")
    except (requests.RequestException, ValueError) as exc:
        return [], f"Football-Data.org request failed: {type(exc).__name__}"


def normalize_standings(rows: list[dict[str, Any]]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for item in rows:
        team = item.get("team") or item.get("teamName") or item.get("name")
        if isinstance(team, dict):
            team = team.get("name")
        if not team:
            continue
        played = safe_int(item, ("play", "played", "playedGames"), 0)
        gf = safe_int(item, ("goalfor", "gf", "goalsFor"), 0)
        ga = safe_int(item, ("goalagainst", "ga", "goalsAgainst"), 0)
        normalized.append({
            "Rank": safe_int(item, ("rank", "position"), len(normalized) + 1),
            "Team": str(team),
            "Played": played,
            "W": safe_int(item, ("win", "won"), 0),
            "D": safe_int(item, ("draw", "drawn"), 0),
            "L": safe_int(item, ("lose", "lost"), 0),
            "GF": gf,
            "GA": ga,
            "GD": safe_int(item, ("goaldistance", "goalDifference"), gf - ga),
            "Points": safe_int(item, ("point", "points"), 0),
            "Avg Goals Scored": gf / played if played else np.nan,
            "Avg Goals Conceded": ga / played if played else np.nan,
        })
    if not normalized:
        return pd.DataFrame()
    return pd.DataFrame(normalized).sort_values("Rank").reset_index(drop=True)


def make_demo_standings(league_name: str) -> pd.DataFrame:
    """Clearly labelled illustrative data; never represent this as current standings."""
    rows = []
    teams = MOCK_TEAMS.get(league_name, ["Demo Team A", "Demo Team B"])
    for rank, team in enumerate(teams, start=1):
        played = 25
        gf = max(12, 50 - rank * 2)
        ga = 20 + rank
        rows.append({
            "Rank": rank, "Team": team, "Played": played,
            "W": max(0, 18 - rank), "D": 4, "L": 3 + rank,
            "GF": gf, "GA": ga, "GD": gf - ga,
            "Points": 58 - rank * 3,
            "Avg Goals Scored": gf / played,
            "Avg Goals Conceded": ga / played,
        })
    return pd.DataFrame(rows)


def recent_form_table(matches: list[dict[str, Any]], team_names: list[str], limit: int = 5) -> pd.DataFrame:
    parsed: list[dict[str, Any]] = []
    for match in matches:
        home = (match.get("homeTeam") or {}).get("name")
        away = (match.get("awayTeam") or {}).get("name")
        score = match.get("score", {}).get("fullTime", {})
        hg, ag = score.get("home"), score.get("away")
        if home and away and hg is not None and ag is not None:
            parsed.append({
                "date": match.get("utcDate", ""), "home": home, "away": away,
                "hg": int(hg), "ag": int(ag),
            })
    parsed.sort(key=lambda row: row["date"], reverse=True)
    out = []
    for team in team_names:
        played = []
        for match in parsed:
            if match["home"] == team or match["away"] == team:
                is_home = match["home"] == team
                gf, ga = (match["hg"], match["ag"]) if is_home else (match["ag"], match["hg"])
                result = "W" if gf > ga else "D" if gf == ga else "L"
                played.append({"date": match["date"][:10], "opponent": match["away"] if is_home else match["home"], "venue": "H" if is_home else "A", "gf": gf, "ga": ga, "result": result})
                if len(played) == limit:
                    break
        out.append({
            "Team": team,
            "Matches available": len(played),
            "Last 5 (newest first)": " · ".join(x["result"] for x in played) if played else "Unavailable",
            "Goals for / match": round(np.mean([x["gf"] for x in played]), 2) if played else np.nan,
            "Goals against / match": round(np.mean([x["ga"] for x in played]), 2) if played else np.nan,
            "Recent matches": "; ".join(f"{x['date']} {x['venue']} vs {x['opponent']} {x['gf']}-{x['ga']} ({x['result']})" for x in played) if played else "No matching team names in recent results",
        })
    return pd.DataFrame(out)


def poisson_analysis(home_xg: float, away_xg: float, max_goals: int = 10) -> tuple[dict[str, float], np.ndarray]:
    # 0..10 score grid; include the tail mass by normalizing the truncated grid.
    home_probs = poisson.pmf(np.arange(max_goals + 1), home_xg)
    away_probs = poisson.pmf(np.arange(max_goals + 1), away_xg)
    grid = np.outer(home_probs, away_probs)
    grid = grid / grid.sum() if grid.sum() else grid
    home_win = float(np.tril(grid, -1).sum())
    draw = float(np.trace(grid))
    away_win = float(np.triu(grid, 1).sum())
    total_goals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
    btts = float(grid[(np.arange(max_goals + 1)[:, None] > 0) & (np.arange(max_goals + 1)[None, :] > 0)].sum())
    over_15 = float(grid[total_goals >= 2].sum())
    over_25 = float(grid[total_goals >= 3].sum())
    over_35 = float(grid[total_goals >= 4].sum())
    home_scores = float(grid.sum(axis=1)[1:].sum())
    away_scores = float(grid.sum(axis=0)[1:].sum())
    home_clean_sheet = float(grid[:, 0].sum())
    away_clean_sheet = float(grid[0, :].sum())
    markets = {
        "Home win": home_win, "Draw": draw, "Away win": away_win,
        "BTTS — Yes": btts, "BTTS — No": 1 - btts,
        "Over 1.5 goals": over_15, "Under 1.5 goals": 1 - over_15,
        "Over 2.5 goals": over_25, "Under 2.5 goals": 1 - over_25,
        "Over 3.5 goals": over_35, "Under 3.5 goals": 1 - over_35,
        "Home team scores": home_scores, "Away team scores": away_scores,
        "Home clean sheet": home_clean_sheet, "Away clean sheet": away_clean_sheet,
    }
    return markets, grid


def most_likely_scores(grid: np.ndarray, count: int = 5) -> pd.DataFrame:
    cells = [(i, j, float(grid[i, j])) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    cells.sort(key=lambda x: x[2], reverse=True)
    return pd.DataFrame([
        {"Score": f"{h}-{a}", "Probability": f"{p:.1%}"} for h, a, p in cells[:count]
    ])


with st.sidebar:
    st.header("⚙️ Data & competition")
    selected_league = st.selectbox("Competition", list(LEAGUES.keys()))
    st.subheader("Optional API credentials")
    collect_key = st.text_input("CollectAPI key", type="password", help="Used for league standings.")
    football_data_token = st.text_input("Football-Data.org token", type="password", help="Used for recent finished matches where competition coverage is available.")
    st.caption("Credentials are used for this session only. Do not commit secrets to GitHub.")
    st.divider()
    st.caption("Data cache: standings 10 min · match history 15 min")

league_info = LEAGUES[selected_league]
raw_standings, standings_status = fetch_collectapi_standings(collect_key, str(league_info["collect_key"]))
standings = normalize_standings(raw_standings)
if standings.empty:
    standings = make_demo_standings(selected_league)
    standings_mode = "SIMULATED DEMO DATA"
else:
    standings_mode = "LIVE API DATA"

matches, matches_status = fetch_recent_matches(football_data_token, str(league_info["football_data_code"] or ""))

with st.expander("Data source status", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        if standings_mode == "LIVE API DATA":
            st.success(f"Standings: {standings_mode}")
        else:
            st.warning(f"Standings: {standings_mode}. These rows are illustrative and must not be used as current league facts.")
        st.caption(standings_status)
    with c2:
        if matches:
            st.success("Recent results: LIVE API DATA")
        else:
            st.warning("Recent results: UNAVAILABLE")
        st.caption(matches_status)
    st.caption(f"Last checked in this session: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

if standings_mode != "LIVE API DATA":
    st.warning("The model is currently using demo standings. Add a valid CollectAPI key to calculate from live standings.")

team_list = standings["Team"].astype(str).tolist()
if len(team_list) < 2:
    st.error("At least two teams are required to run the analysis.")
    st.stop()

match_tab, form_tab, standings_tab, methodology_tab = st.tabs([
    "📊 Match analysis", "🕒 Recent form", "🏆 Standings", "ℹ️ Methodology & limitations"
])

with match_tab:
    st.subheader(f"Match analysis — {selected_league}")
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("Home team", team_list, index=0, key="home_team")
    with c2:
        away_index = 1 if len(team_list) > 1 else 0
        away_team = st.selectbox("Away team", team_list, index=away_index, key="away_team")

    st.markdown("#### Context notes (manual, evidence-led)")
    st.caption("Only enter information you have verified. These notes are shown in the report; they do not silently alter model probabilities.")
    context_col1, context_col2 = st.columns(2)
    with context_col1:
        home_lineup = st.selectbox("Home lineup status", ["Unknown", "Predicted", "Confirmed"], key="home_lineup")
        home_absences = st.text_input("Home team absences / returns", placeholder="Player names and source, if verified")
        home_rest = st.number_input("Home team's rest days", min_value=0, max_value=21, value=4)
    with context_col2:
        away_lineup = st.selectbox("Away lineup status", ["Unknown", "Predicted", "Confirmed"], key="away_lineup")
        away_absences = st.text_input("Away team absences / returns", placeholder="Player names and source, if verified")
        away_rest = st.number_input("Away team's rest days", min_value=0, max_value=21, value=4)
    team_context_note = st.text_area("Other verified context", placeholder="Recent managerial change, schedule congestion, reported tactical change, or source links…", height=80)

    if home_team == away_team:
        st.warning("Choose two different teams to compare them.")
    else:
        home = standings.loc[standings["Team"] == home_team].iloc[0]
        away = standings.loc[standings["Team"] == away_team].iloc[0]
        usable = standings[(standings["Played"] > 0) & standings["Avg Goals Scored"].notna() & standings["Avg Goals Conceded"].notna()]
        league_goal_avg = float(usable["Avg Goals Scored"].mean()) if not usable.empty else 1.35
        league_concede_avg = float(usable["Avg Goals Conceded"].mean()) if not usable.empty else league_goal_avg
        league_goal_avg = max(0.2, league_goal_avg)
        league_concede_avg = max(0.2, league_concede_avg)
        home_attack = float(home["Avg Goals Scored"]) / league_goal_avg if pd.notna(home["Avg Goals Scored"]) else 1.0
        away_defense = float(away["Avg Goals Conceded"]) / league_concede_avg if pd.notna(away["Avg Goals Conceded"]) else 1.0
        away_attack = float(away["Avg Goals Scored"]) / league_goal_avg if pd.notna(away["Avg Goals Scored"]) else 1.0
        home_defense = float(home["Avg Goals Conceded"]) / league_concede_avg if pd.notna(home["Avg Goals Conceded"]) else 1.0
        home_xg = float(np.clip(league_goal_avg * home_attack * away_defense * 1.10, 0.15, 4.5))
        away_xg = float(np.clip(league_goal_avg * away_attack * home_defense, 0.15, 4.5))
        markets, grid = poisson_analysis(home_xg, away_xg)

        st.divider()
        if standings_mode != "LIVE API DATA":
            st.error("Interpret the following figures as a demo of the model only: current standings data is not available.")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Home expected goals", f"{home_xg:.2f}")
        metric_cols[1].metric("Away expected goals", f"{away_xg:.2f}")
        metric_cols[2].metric("Expected total goals", f"{home_xg + away_xg:.2f}")
        metric_cols[3].metric("Model input source", "Live" if standings_mode == "LIVE API DATA" else "Demo")

        st.markdown("#### Scoreline and goal-distribution projections")
        left, right = st.columns([1, 1])
        with left:
            score_df = pd.DataFrame(grid, index=[f"Home {i}" for i in range(grid.shape[0])], columns=[f"Away {i}" for i in range(grid.shape[1])])
            st.dataframe(score_df.style.format("{:.1%}"), use_container_width=True, height=360)
            st.caption("Score grid is normalized over 0–10 goals per team; very high scores outside this range are omitted.")
        with right:
            st.dataframe(most_likely_scores(grid), hide_index=True, use_container_width=True)
            market_rows = [{"Projection": name, "Estimated probability": f"{prob:.1%}"} for name, prob in markets.items()]
            st.dataframe(pd.DataFrame(market_rows), hide_index=True, use_container_width=True, height=360)

        st.markdown("#### Match context record")
        context_rows = [
            {"Item": "Home lineup", "Status / notes": home_lineup},
            {"Item": "Away lineup", "Status / notes": away_lineup},
            {"Item": "Home absences / returns", "Status / notes": home_absences or "Not provided"},
            {"Item": "Away absences / returns", "Status / notes": away_absences or "Not provided"},
            {"Item": "Home rest days", "Status / notes": str(home_rest)},
            {"Item": "Away rest days", "Status / notes": str(away_rest)},
            {"Item": "Other context", "Status / notes": team_context_note or "Not provided"},
        ]
        st.dataframe(pd.DataFrame(context_rows), hide_index=True, use_container_width=True)
        st.caption("Manual context is documented for transparency but is not yet used as a numerical adjustment. This prevents unsupported news or morale assumptions from distorting the statistical baseline.")

with form_tab:
    st.subheader("Recent match form")
    st.write("The table uses completed match scores from Football-Data.org when a token is provided and competition coverage is available.")
    if matches:
        form_df = recent_form_table(matches, team_list, limit=5)
        st.dataframe(form_df, hide_index=True, use_container_width=True)
        st.caption("Team names must match the provider's names exactly. Rows marked unavailable mean no matching team name was found in the returned match set.")
    else:
        st.warning("Recent results are not connected yet. Add a Football-Data.org token in the sidebar, or use a supported competition. No recent form is fabricated.")
        st.markdown("**Competition coverage configured in this app:** Premier League, La Liga, Serie A, Bundesliga, and Champions League. Turkish Süper Lig needs another suitable results provider.")

with standings_tab:
    st.subheader(f"Standings — {selected_league}")
    st.dataframe(standings[["Rank", "Team", "Played", "W", "D", "L", "GF", "GA", "GD", "Points"]], hide_index=True, use_container_width=True)
    st.caption(f"Source status: {standings_mode}. Last checked: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.")

with methodology_tab:
    st.subheader("Model methodology and limitations")
    st.markdown("""
**Current baseline**

- Uses goals-for and goals-against averages from the selected standings source.
- Scales attack and defensive rates relative to league averages.
- Applies a simple home-advantage multiplier to home expected goals.
- Uses independent Poisson distributions to produce a 0–10 goals score matrix.
- Derives 1X2, BTTS, totals, scoring, and clean-sheet probabilities from that matrix.

**Important limitations**

- Standings aggregates are not a substitute for match-by-match xG, opponent strength, or recent form.
- The independent Poisson assumption does not model all score dependence or tactical effects.
- Corners and shots-on-target are deliberately not estimated here without reliable event-level data.
- Team morale is not directly observable. News should be sourced, timestamped, and separated from interpretation.
- Confirmed lineups and injury reports should only be marked confirmed when the provider has explicitly confirmed them.
- Demo standings are illustrative only and are visibly labelled; never treat them as current league data.

**Recommended validation before trusting projections**

Use a time-ordered historical holdout, compare against simple baselines, and track Brier score, log loss, and calibration by probability bucket. Do not evaluate a model using information published after the match kickoff.
""")
