import streamlit as st
import requests
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Football Analysis Dashboard", layout="wide")
st.title("📊 Football Match & Squad Analysis Tool (Live Season)")

# Input for User API Token in Sidebar
API_KEY = st.sidebar.text_input("Enter your Football-Data.org API Token:", type="password")

if not API_KEY:
    st.warning("⚠️ Please enter your free API token in the sidebar to start!")
    st.markdown("Get your free token instantly at [football-data.org](https://www.football-data.org/client/register)")
    st.stop()

# Headers for football-data.org v4 API
HEADERS = {
    "X-Auth-Token": API_KEY
}

# Popular Team ID mapping for instant, reliable lookups on the free tier
POPULAR_TEAMS = {
    "Arsenal FC": 57,
    "Manchester City FC": 65,
    "Real Madrid CF": 86,
    "FC Barcelona": 81,
    "Manchester United FC": 66,
    "Chelsea FC": 61,
    "Liverpool FC": 64,
    "Bayern Munich": 5,
    "Juventus FC": 109,
    "AC Milan": 98,
    "Inter Milan": 108,
    "Paris Saint-Germain FC": 524
}

# --- HELPER FUNCTIONS ---
def get_team_matches(team_id, count=7):
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches?limit={count}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        st.error(f"API Error ({response.status_code}): Please check your API token.")
        return []
        
    data = response.json()
    return data.get("matches", [])

# --- MAIN APP LOGIC ---
selected_team_name = st.selectbox("Select Club:", list(POPULAR_TEAMS.keys()))
team_id = POPULAR_TEAMS[selected_team_name]

if st.button("Analyze Club Matches"):
    with st.spinner("Fetching current season match data..."):
        matches = get_team_matches(team_id, count=7)
        
        if not matches:
            st.warning("No recent match data available for this team.")
        else:
            match_table = []
            for m in matches:
                date = m["utcDate"][:10]
                home_club = m["homeTeam"]["name"]
                away_club = m["awayTeam"]["name"]
                
                score_obj = m.get("score", {}).get("fullTime", {})
                home_goals = score_obj.get("home")
                away_goals = score_obj.get("away")
                
                if home_goals is not None and away_goals is not None:
                    score = f"{home_goals}–{away_goals}"
                    # Determine Result relative to selected team
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
                    "Status": m.get("status", "SCHEDULED"),
                    "Result": res,
                    "Score": score
                })
                
            df = pd.DataFrame(match_table)
            st.subheader(f"📋 Recent & Upcoming Matches for {selected_team_name}")
            st.dataframe(df, use_container_width=True)
