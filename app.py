import streamlit as st
import requests
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Turkish Süper Lig Live Dashboard", layout="wide")
st.title("⚽ Turkish Süper Lig Live Dashboard")

# Sidebar for API Key
API_KEY = st.sidebar.text_input("Enter your CollectAPI Key:", type="password")

if not API_KEY:
    st.warning("⚠️ Please enter your CollectAPI key in the sidebar to load the live league data!")
    st.markdown("Get your free key at [collectapi.com](https://collectapi.com/)")
    st.stop()

HEADERS = {
    "authorization": f"apikey {API_KEY}",
    "content-type": "application/json"
}

# --- FETCH DATA FUNCTIONS ---
@st.cache_data(ttl=600)
def fetch_standings(key):
    url = "https://api.collectapi.com/football/league?league=super-lig"
    try:
        r = requests.get(url, headers={"authorization": f"apikey {key}", "content-type": "application/json"})
        data = r.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and data.get("success"):
            return data.get("result", [])
    except Exception as e:
        st.error(f"Error fetching standings: {e}")
    return []

@st.cache_data(ttl=600)
def fetch_results(key):
    url = "https://api.collectapi.com/football/results?league=super-lig"
    try:
        r = requests.get(url, headers={"authorization": f"apikey {key}", "content-type": "application/json"})
        data = r.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and data.get("success"):
            return data.get("result", [])
    except Exception as e:
        st.error(f"Error fetching results: {e}")
    return []

@st.cache_data(ttl=600)
def fetch_top_scorers(key):
    url = "https://api.collectapi.com/football/goalKings?league=super-lig"
    try:
        r = requests.get(url, headers={"authorization": f"apikey {key}", "content-type": "application/json"})
        data = r.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and data.get("success"):
            return data.get("result", [])
    except Exception as e:
        st.error(f"Error fetching top scorers: {e}")
    return []

# --- MAIN DASHBOARD INTERFACE ---
tab1, tab2, tab3 = st.tabs(["🏆 League Standings", "📅 Match Results", "⚽ Top Scorers"])

# TAB 1: STANDINGS
with tab1:
    st.subheader("Current Süper Lig Table")
    standings_data = fetch_standings(API_KEY)
    
    if standings_data:
        df_standings = pd.DataFrame(standings_data)
        
        # Select and rename columns for a clean presentation
        columns_map = {
            "rank": "Rank",
            "team": "Club",
            "play": "Played",
            "win": "W",
            "draw": "D",
            "lose": "L",
            "goalfor": "GF",
            "goalagainst": "GA",
            "goaldistance": "GD",
            "point": "Points"
        }
        
        # Filter available columns
        present_cols = [c for c in columns_map.keys() if c in df_standings.columns]
        df_clean = df_standings[present_cols].rename(columns=columns_map)
        
        # Interactive team highlight filter
        team_list = ["All Teams"] + list(df_clean["Club"].unique())
        selected_team = st.selectbox("Highlight Club:", team_list)
        
        if selected_team != "All Teams":
            df_display = df_clean[df_clean["Club"] == selected_team]
        else:
            df_display = df_clean

        st.dataframe(df_display, hide_index=True, use_container_width=True)
    else:
        st.info("No standings data available. Check your API Key.")

# TAB 2: RESULTS
with tab2:
    st.subheader("Recent & Fixture Results")
    results_data = fetch_results(API_KEY)
    
    if results_data:
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, hide_index=True, use_container_width=True)
    else:
        st.info("No match results available.")

# TAB 3: TOP SCORERS
with tab3:
    st.subheader("Süper Lig Goal Kings")
    scorers_data = fetch_top_scorers(API_KEY)
    
    if scorers_data:
        df_scorers = pd.DataFrame(scorers_data)
        st.dataframe(df_scorers, hide_index=True, use_container_width=True)
    else:
        st.info("No top scorers data available.")
