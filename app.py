import streamlit as st
import requests
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Football Analysis Dashboard", layout="wide")
st.title("📊 Football Match & Squad Analysis Tool (Live Season)")

# Input for User API Key in Sidebar
API_KEY = st.sidebar.text_input("Enter your Football-Data.org API Token:", type="password")

if not API_KEY:
    st.warning("⚠️ Please enter your free API token in the sidebar to start!")
    st.markdown("Get your free token instantly at [football-data.org](https://www.football-data.org/client/register)")
    st.stop()

# Headers for football-data.org v4 API
HEADERS = {
    "X-Auth-Token": API_KEY
}

# Free tier available competitions on football-data.org
LEAGUES = {
    "English Premier League (PL)": "PL",
    "Spanish La Liga (PD)": "PD",
    "Italian Serie A (SA)": "SA",
    "German Bundesliga (BL1)": "BL1",
    "French Ligue 1 (FL1)": "FL1",
    "UEFA Champions League (CL)": "CL"
}

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_teams_for_competition(league_code, token):
    url = f"https://api.football-data.org/v4/competitions/{league_code}/teams"
    response = requests.get(url, headers={"X-Auth-Token": token})
    if response.status_code == 200:
        teams_data = response.json().get("teams", [])
        return {team["name"]: team["id"] for team in teams_data}
    return {}

def get_team_matches(team_id):
    # Fetching matches for the team
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches?limit=15"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        return []
        
    data = response.json()
    return data.get("matches", [])

# --- MAIN APP LOGIC ---
selected_league_name = st.sidebar.selectbox("Select Competition:", list(LEAGUES.keys()))
league_code = LEAGUES[selected_league_name]

# Dynamically load teams for the chosen league
teams_dict = get_teams_for_competition(league_code, API_KEY)

if not teams_dict:
    st.error("Could not load teams. Please check your API token or try another competition.")
    st.stop()

selected_team_name = st.selectbox("Select Club:", sorted(list(teams_dict.keys())))
team_id = teams_dict[selected_team_name]

if st.button("Analyze Club Matches"):
    with st.spinner("Fetching current season match data..."):
        matches = get_team_matches(team_id)
        
        if not matches:
            st.warning("No match data available for this team.")
        else:
            match_table = []
            for m in matches:
                date = m["utcDate"][:10]
                home_club = m["homeTeam"]["name"]
                away_club = m["awayTeam"]["name"]
                status = m.get("status", "SCHEDULED")
                
                score_obj = m.get("score", {}).get("fullTime", {})
                home_goals = score_obj.get("home")
                away_goals = score_obj.get("away")
                
                if status == "FINISHED" and home_goals is not None and away_goals is not None:
                    score = f"{home_goals}–{away_goals}"
                    is_home = (m["homeTeam"]["id"] == team_id)
                    team_goals = home_goals if is_home else away_goals
                    opp_goals = away_goals if is_home else home_goals
                    
                    if team_goals > opp_goals:
                        res = "🟢 W"
                    elif team_goals == opp_goals:
                        res = "🟡 D"
                    else:
                        res = "🔴 L"
                else:
                    score = "Upcoming"
                    res = "⏳ Scheduled"
                
                match_table.append({
                    "Date": date,
                    "Matchup": f"{home_club} vs {away_club}",
                    "Status": status,
                    "Result": res,
                    "Score": score
                })
                
            df = pd.DataFrame(match_table)
            st.subheader(f"📋 Matches for {selected_team_name}")
            st.dataframe(df, use_container_width=True)
