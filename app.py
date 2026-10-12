import streamlit as st
import requests
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Football Analysis Dashboard", layout="wide")
st.title("📊 Football Match & Squad Analysis Tool")

# Input for User API Key in Sidebar
API_KEY = st.sidebar.text_input("Enter your API-Football Key:", type="password")

if not API_KEY:
    st.warning("⚠️ Please enter your API-Football Key in the sidebar to start!")
    st.stop()

# Headers for direct API-Football access
HEADERS = {
    "x-apisports-key": API_KEY
}

# --- HELPER FUNCTIONS ---
def get_team_info(team_name):
    url = f"https://v3.football.api-sports.io/teams?search={team_name}"
    response = requests.get(url, headers=HEADERS).json()
    if response.get("response"):
        return response["response"][0]["team"]
    return None

def get_last_matches(team_id, count=7):
    url = f"https://v3.football.api-sports.io/fixtures?team={team_id}&last={count}"
    response = requests.get(url, headers=HEADERS).json()
    return response.get("response", [])

def get_match_stats(fixture_id):
    url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fixture_id}"
    response = requests.get(url, headers=HEADERS).json()
    
    stats_data = {"Home SoT": 0, "Away SoT": 0, "Home Corners": 0, "Away Corners": 0}
    
    if response.get("response") and len(response["response"]) == 2:
        home_stats = {item['type']: item['value'] for item in response["response"][0]["statistics"]}
        away_stats = {item['type']: item['value'] for item in response["response"][1]["statistics"]}
        
        stats_data["Home SoT"] = home_stats.get("Shots on Goal", 0) or 0
        stats_data["Away SoT"] = away_stats.get("Shots on Goal", 0) or 0
        stats_data["Home Corners"] = home_stats.get("Corner Kicks", 0) or 0
        stats_data["Away Corners"] = away_stats.get("Corner Kicks", 0) or 0
        
    return stats_data

def get_injuries(team_id):
    url = f"https://v3.football.api-sports.io/injuries?team={team_id}"
    response = requests.get(url, headers=HEADERS).json()
    return response.get("response", [])

# --- MAIN APP LOGIC ---
team_search = st.text_input("Enter Football Club Name (e.g., Galatasaray, Arsenal, Real Madrid):", "Galatasaray")

if st.button("Analyze Club"):
    with st.spinner("Fetching verified match data..."):
        team = get_team_info(team_search)
        
        if not team:
            st.error("Club not found! Please check the spelling.")
            st.stop()
            
        team_id = team["id"]
        st.header(f"⚽ {team['name']} ({team['country']})")
        st.image(team["logo"], width=80)
        
        # --- 1. LAST 7 MATCHES TABLE ---
        st.subheader("📋 Last 7 Matches (Verified Official Stats)")
        matches = get_last_matches(team_id, count=7)
        
        match_table = []
        for m in matches:
            fixture_id = m["fixture"]["id"]
            date = m["fixture"]["date"][:10]
            home_club = m["teams"]["home"]["name"]
            away_club = m["teams"]["away"]["name"]
            
            home_goals = m["goals"]["home"]
            away_goals = m["goals"]["away"]
            score = f"{home_goals}–{away_goals}"
            
            # Determine Result relative to searched team
            if (m["teams"]["home"]["id"] == team_id and home_goals > away_goals) or \
               (m["teams"]["away"]["id"] == team_id and away_goals > home_goals):
                res = "🟢 W"
            elif home_goals == away_goals:
                res = "🟡 D"
            else:
                res = "🔴 L"
                
            # Get SoT and Corners
            stats = get_match_stats(fixture_id)
            
            match_table.append({
                "Date": date,
                "Matchup": f"{home_club} – {away_club}",
                "Res": res,
                "Score": score,
                "Shots on Target": f"{stats['Home SoT']} – {stats['Away SoT']}",
                "Corners": f"{stats['Home Corners']} – {stats['Away Corners']}"
            })
            
        df = pd.DataFrame(match_table)
        st.dataframe(df, use_container_width=True)
        
        # --- 2. SQUAD INJURIES & MISSING PLAYERS ---
        st.subheader("🏥 Current Missing & Injured Players")
        injuries = get_injuries(team_id)
        
        if injuries:
            injury_data = []
            for inj in injuries[:10]:
                injury_data.append({
                    "Player": inj["player"]["name"],
                    "Reason": inj["player"]["reason"],
                    "Fixture Date": inj["fixture"]["date"][:10]
                })
            st.table(pd.DataFrame(injury_data))
        else:
            st.info("No major injuries reported currently.")
