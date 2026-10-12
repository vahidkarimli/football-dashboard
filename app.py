import streamlit as st
import requests
import pandas as pd

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Turkish Süper Lig Dashboard", layout="wide")
st.title("📊 Turkish Süper Lig Live Dashboard (CollectAPI)")

# Input for CollectAPI Key in Sidebar
API_KEY = st.sidebar.text_input("Enter your CollectAPI Key:", type="password")

if not API_KEY:
    st.warning("⚠️ Please enter your free CollectAPI key in the sidebar to start!")
    st.markdown("Get your free key instantly at [collectapi.com](https://collectapi.com/)")
    st.stop()

# Headers for CollectAPI
HEADERS = {
    "authorization": f"apikey {API_KEY}",
    "content-type": "application/json"
}

# --- HELPER FUNCTIONS ---
def get_super_lig_standings():
    url = "https://api.collectapi.com/football/league?league=super-lig"
    try:
        response = requests.get(url, headers=HEADERS)
        data = response.json()
        
        # Display raw API output for debugging if needed
        with st.expander("🛠️ API Debug Inspector (Click to view raw API response)"):
            st.write("Status Code:", response.status_code)
            st.write("Response Data:", data)

        if isinstance(data, dict) and data.get("success"):
            return data.get("result", [])
        elif isinstance(data, dict) and "message" in data:
            st.error(f"CollectAPI Message: {data['message']}")
        else:
            st.error("Received invalid response format from CollectAPI.")
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
    return []

def get_super_lig_results():
    url = "https://api.collectapi.com/football/results?league=super-lig"
    try:
        response = requests.get(url, headers=HEADERS)
        data = response.json()
        
        if isinstance(data, dict) and data.get("success"):
            return data.get("result", [])
        elif isinstance(data, dict) and "message" in data:
            st.error(f"CollectAPI Message: {data['message']}")
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
    return []

# --- MAIN APP LOGIC ---
tab1, tab2 = st.tabs(["🏆 League Standings", "⚽ Recent Results"])

with tab1:
    st.subheader("Current Süper Lig Standings")
    if st.button("Load Standings"):
        with st.spinner("Fetching standings..."):
            standings = get_super_lig_standings()
            if standings:
                df_standings = pd.DataFrame(standings)
                st.dataframe(df_standings, use_container_width=True)

with tab2:
    st.subheader("Recent Match Results")
    if st.button("Load Results"):
        with st.spinner("Fetching match results..."):
            results = get_super_lig_results()
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
