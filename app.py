import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Turkish Süper Lig Dashboard", layout="wide")
st.title("📊 Turkish Süper Lig Live Dashboard")

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
    # Search specifically within Turkey (country code: TR)
    url = f"https://v3.football.api-sports.io/teams?search={team_name}&country=Turkey"
    response = requests.get(url, headers=HEADERS).json()
    if response.get("response"):
        return response["response"][0]["team"]
    return None

def get_team_fixtures_by_id(team_id, count=7):
    # Using last={count} and league=203 (Turkish Süper Lig) without restricted season parameters
    url = f"https://v3.football.api-sports.io/fixtures?team={team_id}&league=203&last={count}"
    response = requests.get(url, headers=HEADERS).json()
    
    # If the season parameter error triggers, let's catch it gracefully
    if "errors" in response and response["errors"]:
        if isinstance(response["errors"], dict):
            error_msg = response["errors"].get("plan", "API Plan restriction.")
            st.error(f"API restriction: {error_msg}")
        return []
        
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

# --- MAIN APP LOGIC ---
team_search = st.text_input("Enter Turkish Club Name (e.g., Galatasaray, Fenerbahce, Besiktas, Trabzonspor):", "Galatasaray")

if st.button("Analyze Süper Lig Club"):
    with st.spinner("Fetching live Süper Lig data..."):
        team = get_team_info(team_search)
        
        if not team:
            st.error("Club not found! Please check the spelling.")
            st.stop()
            
        team_id = team["id"]
        st.header(f"⚽ {team['name']} ({team['country']})")
        st.image(team["logo"], width=80)
        
        # --- 1. MATCHES TABLE ---
        st.subheader("📋 Recent Matches & Current Updates")
        matches = get_team_fixtures_by_id(team_id, count=7)
        
        if not matches:
            st.warning("No match data returned. Free tier keys may require checking live match days directly.")
        else:
            match_table = []
            for m in matches:
                fixture_id = m["fixture"]["id"]
                date = m["fixture"]["date"][:10]
                home_club = m["teams"]["home"]["name"]
                away_club = m["teams"]["away"]["name"]
                status = m["fixture"]["status"]["short"]
                
                home_goals = m["goals"]["home"] if m["goals"]["home"] is not None else 0
                away_goals = m["goals"]["away"] if m["goals"]["away"] is not None else 0
                score = f"{home_goals}–{away_goals}" if status in ["FT", "AET", "PEN", "1H", "2H", "HT"] else "Upcoming"
                
                if status in ["FT", "AET", "PEN"]:
                    if (m["teams"]["home"]["id"] == team_id and home_goals > away_goals) or \
                       (m["teams"]["away"]["id"] == team_id and away_goals > home_goals):
                        res = "🟢 W"
                    elif home_goals == away_goals:
                        res = "🟡 D"
                    else:
                        res = "🔴 L"
                else:
                    res = "⏳ Scheduled"
                    
                # Get SoT and Corners if match has data
                stats = get_match_stats(fixture_id) if status in ["FT", "AET", "PEN", "1H", "2H", "HT"] else {"Home SoT": 0, "Away SoT": 0, "Home Corners": 0, "Away Corners": 0}
                
                match_table.append({
                    "Date": date,
                    "Matchup": f"{home_club} – {away_club}",
                    "Status": status,
                    "Result": res,
                    "Score": score,
                    "Shots on Target": f"{stats['Home SoT']} – {stats['Away SoT']}",
                    "Corners": f"{stats['Home Corners']} – {stats['Away Corners']}"
                })
                
            df = pd.DataFrame(match_table)
            st.dataframe(df, use_container_width=True)
