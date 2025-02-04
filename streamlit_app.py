import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from datetime import datetime, timedelta
import requests
import os

#API code here


# Get the latest date
yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

today = datetime.today().strftime('%Y-%m-%d')

# ✅ **Cache Function to Fetch Air Quality Data** for preloaded gjson file 
@st.cache_data(ttl=3600)  # Cache expires after 1 hour
def fetch_air_quality_data():
    """Fetch air quality data for all districts and cache the results."""
    district_data = {}

    for district_name, (lat, lon) in coordinates_mapping.items():
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "european_aqi", "non_methane_volatile_organic_compounds", "secondary_inorganic_aerosol", "nitrogen_monoxide"],
            "timezone": "Europe/Warsaw",
            "start_date": today,
            "end_date": today,
            "domains": "cams_europe"
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "hourly" in data:
                hourly_data = data["hourly"]
                latest_index = -1  # Get the latest available hour

                district_data[district_name] = {
                    "PM10": hourly_data["pm10"][latest_index],
                    "PM2.5": hourly_data["pm2_5"][latest_index],
                    "CO": hourly_data["carbon_monoxide"][latest_index],
                    "NO2": hourly_data["nitrogen_dioxide"][latest_index],
                    "SO2": hourly_data["sulphur_dioxide"][latest_index],
                    "Ozone": hourly_data["ozone"][latest_index],
                    "AQI": hourly_data["european_aqi"][latest_index],
                    "NMVOC": hourly_data["non_methane_volatile_organic_compounds"][latest_index],
                    "NO": hourly_data["nitrogen_monoxide"][latest_index],
                    "SIA": hourly_data["secondary_inorganic_aerosol"][latest_index]
                }
        else:
            district_data[district_name] = None  # No data available

    return district_data

# ✅ **Cache Function for Custom Latitude & Longitude**
@st.cache_data(ttl=3600)
def fetch_air_quality_for_custom_location(lat, lon):
    """Fetch air quality data for any custom location worldwide."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", 
            "european_aqi", "non_methane_volatile_organic_compounds", "nitrogen_monoxide", "secondary_inorganic_aerosol"
        ],
        "timezone": "auto",
        "start_date": today,
        "end_date": today,
        "domains": "cams_global"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if "hourly" in data:
            hourly_data = data["hourly"]
            latest_index = -1

            return {
                "PM10": hourly_data["pm10"][latest_index],
                "PM2.5": hourly_data["pm2_5"][latest_index],
                "CO": hourly_data["carbon_monoxide"][latest_index],
                "NO2": hourly_data["nitrogen_dioxide"][latest_index],
                "SO2": hourly_data["sulphur_dioxide"][latest_index],
                "Ozone": hourly_data["ozone"][latest_index],
                "AQI": hourly_data["european_aqi"][latest_index],
                "NMVOC": hourly_data["non_methane_volatile_organic_compounds"][latest_index],
                "NO": hourly_data["nitrogen_monoxide"][latest_index],
                "SIA": hourly_data["secondary_inorganic_aerosol"][latest_index]
            }
    return None


# District Coordinates Mapping
coordinates_mapping = {
    'Bemowo': (52.2545, 20.9110),
    'Bielany': (52.2916, 20.9336),
    'Ursynow': (52.1501, 21.0504),
    'Wola': (52.2367, 20.9836),
    'Praga-Poludnie': (52.2445, 21.0708),
    'Srodmiescie': (52.2319, 21.0059),
    'Targowek': (52.2890, 21.0490),
    'Praga-Polnoc': (52.2561, 21.0360),
    'Ochota': (52.2196, 20.9772),
    'Mokotow': (52.1917, 21.0368)
}



# User interface code here

# Set page layout
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header("Map Options")

# Default map center (Example: Warsaw)
default_lat = 52.2298
default_lon = 21.0122
zoom_level = 12

# Sidebar for User Input (Custom Location)
st.sidebar.header("Enter Custom Location")
user_lat = st.sidebar.number_input("Latitude", value=default_lat, format="%.6f")
user_lon = st.sidebar.number_input("Longitude", value=default_lon, format="%.6f")

if "show_data" not in st.session_state:
    st.session_state.show_data = False  # Initially hidden

if "map_center" not in st.session_state:
    st.session_state.map_center = [user_lat, user_lon]  # Default to Warsaw

# Button to Toggle Air Quality Data
if st.sidebar.button("Toggle Air Quality Data"):
    st.session_state.show_data = not st.session_state.show_data  # Toggle state

    # If toggled ON, move the map center to the new location
    if st.session_state.show_data:
        st.session_state.map_center = [user_lat, user_lon]


# ✅ **Fetch Cached Air Quality Data Once Per Session**
air_quality_data = fetch_air_quality_data()

custom_location_data = fetch_air_quality_for_custom_location(user_lat, user_lon) if st.session_state.show_data else None

# ✅ Create the Map Centered on Warsaw
m = folium.Map(location=[default_lat, default_lon], zoom_start=zoom_level)

# ✅ Add Markers for Custom Location
if st.session_state.show_data  and custom_location_data:
    popup_html = f"""
    <b>🌍 Custom Location: ({user_lat}, {user_lon})</b><br>
    📅 <b>Date:</b> {today}<br>
    🏭 <b>AQI:</b> {custom_location_data["AQI"]}<br>
    🏭 <b>PM10:</b> {custom_location_data["PM10"]} µg/m³<br>
    ☁️ <b>PM2.5:</b> {custom_location_data["PM2.5"]} µg/m³<br>
    💨 <b>CO:</b> {custom_location_data["CO"]} ppm<br>
    ⚠️ <b>NO2:</b> {custom_location_data["NO2"]} ppm<br>
    ⚠️ <b>SO2:</b> {custom_location_data["SO2"]} ppm<br>
    🌞 <b>Ozone:</b> {custom_location_data["Ozone"]} ppm<br>
    🏭 <b>NMVOC:</b> {custom_location_data["NMVOC"]} ppm<br>
    ⚠️ <b>NO:</b> {custom_location_data["NO"]} ppm<br>
    🏭 <b>SIA:</b> {custom_location_data["SIA"]} µg/m³<br>
    """
    folium.Marker(
        [user_lat, user_lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip="Click for Air Quality Data",
        icon=folium.Icon(color="blue", icon="cloud"),
    ).add_to(m)


# ✅ Load and Display the Preloaded GeoJSON File
geojson_file_path = "map.geojson"

if os.path.exists(geojson_file_path) and not st.session_state.show_data :
    try:
        with open(geojson_file_path, "r", encoding="utf-8") as file:
            geojson_data = json.load(file)

        # Extract properties and dynamically update pop-ups
        def style_function(feature):
            return {
                "color": "black",  # Outline only
                "weight": 2,
                "fillOpacity": 0  # No fill color
            }

        def highlight_function(feature):
            return {
                "weight": 4,
                "color": "blue",
                "fillOpacity": 0.1
            }

        # Loop through each district in the GeoJSON file
        for feature in geojson_data["features"]:
            district_name = feature["properties"].get("district_name", "Unknown")
            air_quality = air_quality_data.get(district_name)

            if air_quality:
                popup_html = f"""
                <b>🌆 {district_name} - {today}</b><br>
                🏭 <b>AQI:</b> {air_quality["AQI"]}<br>
                🏭 <b>PM10:</b> {air_quality["PM10"]} µg/m³<br>
                ☁️ <b>PM2.5:</b> {air_quality["PM2.5"]} µg/m³<br>
                💨 <b>CO:</b> {air_quality["CO"]} ppm<br>
                ⚠️ <b>NO2:</b> {air_quality["NO2"]} ppm<br>
                ⚠️ <b>SO2:</b> {air_quality["SO2"]} ppm<br>
                🌞 <b>Ozone:</b> {air_quality["Ozone"]} ppm<br>
                🏭 <b>NMVOC:</b> {air_quality["NMVOC"]} ppm<br>
                ⚠️ <b>NO:</b> {air_quality["NO"]} ppm<br>
                🏭 <b>SIA:</b> {air_quality["SIA"]} µg/m³<br>
                """
            else:
                popup_html = f"<b>🌆 {district_name}</b><br>No data available"

            popup = folium.Popup(popup_html, max_width=300)

            folium.GeoJson(
                feature,
                name="Preloaded GeoJSON",
                popup=popup,
                tooltip="Hover for details",
                style_function=style_function,
                highlight_function=highlight_function
            ).add_to(m)

        st.success("✅ Preloaded GeoJSON file loaded successfully!")
        # ✅ Display the Map in Streamlit

    except json.JSONDecodeError:
        st.error("⚠️ Error decoding GeoJSON file. Check the format.")

else:
    st.error(f"⚠️ GeoJSON file '{geojson_file_path}' not found.")

st.title("📍 Interactive Air Quality Map of Warsaw (Today's Data, Cached)")
st_folium(m, width=900, height=600)
