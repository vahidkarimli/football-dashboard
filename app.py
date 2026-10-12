from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson

st.set_page_config(
    page_title="Universal Live Football Analytics & Predictions",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Universal Live Football Analytics & Match Predictions")
st.caption(
    "Match probabilities · Recent form · Automated injury/context lookup · Data provenance"
)
st.info(
    "This dashboard provides match analysis, statistical projections, and automated injury insights based on available live data feeds."
)

LEAGUES: dict[str, dict[str, str | None]] = {
    "Turkish Süper Lig": {"collect_key": "super-lig", "football_data_code": None, "api_football_id": 203},
    "English Premier League": {"collect_key": "ingiltere-premier-ligi", "football_data_code": "PL", "api_football_id": 39},
    "Spanish La Liga": {"collect_key": "ispanya-la-liga", "football_data_code": "PD", "api_football_id": 140},
    "Italian Serie A": {"collect_key": "italya-serie-a", "football_data_code": "SA", "api_football_id": 135},
    "German Bundesliga": {"collect_key": "almanya-bundesliga", "football_data_code": "BL1", "api_football_id": 78},
    "UEFA Champions League": {"collect_key": "sampiyonlar-ligi", "football_data_code": "CL", "api_football_id": 2},
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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_api_football_injuries(api_key: str, league_id: int) -> list[dict[str, Any]]:
    """Fetches active injuries from API-Football (v3.football.api-sports.io)."""
    if not api_key:
        return []
    url = "https://v3.football.api-sports.io/injuries"
    headers = {"x-apisports-key": api_key}
    params = {"league": league_id, "season": 2026}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=12)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", [])
    except Exception:
        pass
    return []


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
            "Recent matches": "; ".join(f"{x['date']} {x['venue']} vs {x['opponent']} {x['gf']}-{x['ga']} ({x['result']})" for x in played) if played else "No matching team names",
        })
    return pd.DataFrame(out)


def poisson_analysis(home_xg: float, away_xg: float, max_goals: int = 10) -> tuple[dict[str, float], np.ndarray]:
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
    st.header("⚙️ Data & Competition")
    selected_league = st.selectbox("Competition", list(LEAGUES.keys()))
    st.subheader("🔑 API Credentials")
    collect_key = st.text_input("CollectAPI Key", type="password", help="For live league standings.")
    football_data_token = st.text_input("Football-Data.org Token", type="password", help="For recent match results.")
    api_football_key = st.text_input("API-Football Key (API-Sports)", type="password", help="For automated team injuries and squad news.")
    st.caption("Credentials remain in session memory only.")

league_info = LEAGUES[selected_league]
raw_standings, standings_status = fetch_collectapi_standings(collect_key, str(league_info["collect_key"]))
standings = normalize_standings(raw_standings)
if standings.empty:
    standings = make_demo_standings(selected_league)
    standings_mode = "SIMULATED DEMO DATA"
else:
    standings_mode = "LIVE API DATA"

matches, matches_status = fetch_recent_matches(football_data_token, str(league_info["football_data_code"] or ""))

# Fetch automated injuries if key is provided
injuries_list = fetch_api_football_injuries(api_football_key, int(league_info["api_football_id"])) if api_football_key else []

with st.expander("Data Source Status", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success(f"Standings: {standings_mode}")
        st.caption(standings_status)
    with c2:
        st.success("Recent results: " + ("Live" if matches else "Unavailable"))
        st.caption(matches_status)
    with c3:
        st.success("Automated Injuries: " + ("Connected" if injuries_list else "Manual/Optional"))
        st.caption(f"{len(injuries_list)} injury reports loaded" if injuries_list else "Add API-Football key for auto-injuries")

team_list = standings["Team"].astype(str).tolist()
if len(team_list) < 2:
    st.error("At least two teams are required.")
    st.stop()

match_tab, form_tab, standings_tab, methodology_tab = st.tabs([
    "📊 Match Analysis & Predictions", "🕒 Recent form", "🏆 Standings", "ℹ️ Methodology"
])

with match_tab:
    st.subheader(f"Match Analysis & Predictions — {selected_league}")
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("Home team", team_list, index=0, key="home_team")
    with c2:
        away_index = 1 if len(team_list) > 1 else 0
        away_team = st.selectbox("Away team", team_list, index=away_index, key="away_team")

    # Automated injury lookup filtering
    home_injuries = [inj for inj in injuries_list if home_team.lower() in str(inj.get("team", {}).get("name", "")).lower()]
    away_injuries = [inj for inj in injuries_list if away_team.lower() in str(inj.get("team", {}).get("name", "")).lower()]

    st.markdown("#### 🛡️ Automated Team Conditions & Injuries")
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown(f"**{home_team} Reported Absences:**")
        if home_injuries:
            for inj in home_injuries:
                player_name = inj.get("player", {}).get("name", "Unknown Player")
                reason = inj.get("player", {}).get("reason", "Unknown")
                st.write(f"• `{player_name}` ({reason})")
        else:
            st.caption("No automated injuries found or API key not set.")
    with ic2:
        st.markdown(f"**{away_team} Reported Absences:**")
        if away_injuries:
            for inj in away_injuries:
                player_name = inj.get("player", {}).get("name", "Unknown Player")
                reason = inj.get("player", {}).get("reason", "Unknown")
                st.write(f"• `{player_name}` ({reason})")
        else:
            st.caption("No automated injuries found or API key not set.")

    if home_team == away_team:
        st.warning("Choose two different teams to compare them.")
    else:
        home = standings.loc[standings["Team"] == home_team].iloc[0]
        away = standings.loc[standings["Team"] == away_team].iloc[0]
        usable = standings[(standings["Played"] > 0) & standings["Avg Goals Scored"].notna() & standings["Avg Goals Conceded"].notna()]
        league_goal_avg = float(usable["Avg Goals Scored"].mean()) if not usable.empty else 1.35
        league_concede_avg = float(usable["Avg Goals Conceded"].mean()) if not usable.empty else league_goal_avg
        
        home_attack = float(home["Avg Goals Scored"]) / league_goal_avg if pd.notna(home["Avg Goals Scored"]) else 1.0
        away_defense = float(away["Avg Goals Conceded"]) / league_concede_avg if pd.notna(away["Avg Goals Conceded"]) else 1.0
        away_attack = float(away["Avg Goals Scored"]) / league_goal_avg if pd.notna(away["Avg Goals Scored"]) else 1.0
        home_defense = float(home["Avg Goals Conceded"]) / league_concede_avg if pd.notna(home["Avg Goals Conceded"]) else 1.0
        
        # Adjust xG automatically based on number of fetched injuries (~4% reduction per missing player)
        home_injury_penalty = max(0.7, 1.0 - (len(home_injuries) * 0.04))
        away_injury_penalty = max(0.7, 1.0 - (len(away_injuries) * 0.04))

        home_xg = float(np.clip(league_goal_avg * home_attack * away_defense * 1.10 * home_injury_penalty, 0.15, 4.5))
        away_xg = float(np.clip(league_goal_avg * away_attack * home_defense * away_injury_penalty, 0.15, 4.5))
        
        markets, grid = poisson_analysis(home_xg, away_xg)

        st.divider()
        metric_cols = st.columns(4)
        metric_cols[0].metric("Home expected goals (xG)", f"{home_xg:.2f}")
        metric_cols[1].metric("Away expected goals (xG)", f"{away_xg:.2f}")
        metric_cols[2].metric("Expected total goals", f"{home_xg + away_xg:.2f}")
        metric_cols[3].metric("Injury Impact", f"Home: -{len(home_injuries)*4}% | Away: -{len(away_injuries)*4}%")

        # Simplified Match Predictions & Insights Section
        st.markdown("#### 🎯 Match Predictions & Form Insights")
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            st.markdown(f"**Outcome Projections:**")
            st.write(f"• **{home_team} Win:** `{markets['Home win']:.1%}`")
            st.write(f"• **Draw:** `{markets['Draw']:.1%}`")
            st.write(f"• **{away_team} Win:** `{markets['Away win']:.1%}`")
            
            recommended_pick = max(
                [(home_team + " Win", markets['Home win']), 
                 ("Draw", markets['Draw']), 
                 (away_team + " Win", markets['Away win'])], 
                key=lambda x: x[1]
            )
            st.success(f"🔥 **Model Lean / Top Projection:** {recommended_pick[0]} ({recommended_pick[1]:.1%})")

        with pred_col2:
            st.markdown(f"**Goal & Market Projections:**")
            st.write(f"• **Both Teams to Score (BTTS):** Yes (`{markets['BTTS — Yes']:.1%}`) / No (`{markets['BTTS — No']:.1%}`)")
            st.write(f"• **Over 2.5 Goals:** `{markets['Over 2.5 goals']:.1%}`")
            st.write(f"• **Under 2.5 Goals:** `{markets['Under 2.5 goals']:.1%}`")
            top_btts = "BTTS — Yes" if markets['BTTS — Yes'] > 0.5 else "BTTS — No"
            top_goals = "Over 2.5 Goals" if markets['Over 2.5 goals'] > 0.5 else "Under 2.5 Goals"
            st.info(f"💡 **Key Trend Insights:** Favors **{top_btts}** and **{top_goals}** based on current performance and injury data.")

        st.markdown("#### Scoreline and Goal-Distribution Projections")
        left, right = st.columns([1, 1])
        with left:
            score_df = pd.DataFrame(grid, index=[f"Home {i}" for i in range(grid.shape[0])], columns=[f"Away {i}" for i in range(grid.shape[1])])
            st.dataframe(score_df.style.format("{:.1%}"), use_container_width=True, height=360)
        with right:
            st.dataframe(most_likely_scores(grid), hide_index=True, use_container_width=True)
            market_rows = [{"Projection": name, "Estimated probability": f"{prob:.1%}"} for name, prob in markets.items()]
            st.dataframe(pd.DataFrame(market_rows), hide_index=True, use_container_width=True, height=360)

with form_tab:
    st.subheader("Recent Match Form")
    if matches:
        form_df = recent_form_table(matches, team_list, limit=5)
        st.dataframe(form_df, hide_index=True, use_container_width=True)
    else:
        st.warning("Recent results require a Football-Data.org token in the sidebar.")

with standings_tab:
    st.subheader(f"Standings — {selected_league}")
    st.dataframe(standings[["Rank", "Team", "Played", "W", "D", "L", "GF", "GA", "GD", "Points"]], hide_index=True, use_container_width=True)

with methodology_tab:
    st.subheader("Model Methodology")
    st.markdown("""
- Combines live standings with automated injury lookups via API-Football.
- Automatically adjusts expected goals (xG) downward when key players are listed as injured or suspended.
- Uses independent Poisson distributions to compute match outcome probabilities and goal markets.
""")
