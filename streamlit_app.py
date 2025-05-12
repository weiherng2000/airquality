import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from datetime import datetime, timedelta
import requests
import os
from retry_requests import retry
import requests_cache
import openmeteo_requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.losses import MeanSquaredError 







#API code here
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)



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
            "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "european_aqi", "non_methane_volatile_organic_compounds", "secondary_inorganic_aerosol", "nitrogen_monoxide","ammonia"],
            "timezone": "GMT",
            "start_date": today,
            "end_date": today
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
                    "SIA": hourly_data["secondary_inorganic_aerosol"][latest_index],
                    "NH3" : hourly_data["ammonia"][latest_index]
                }
        else:
            district_data[district_name] = None  # No data available

    return district_data

# ✅ **Cache Function for Custom Latitude & Longitude**
@st.cache_data(ttl=3600)
def fetch_latest_air_quality(lat, lon):
    """Fetch the latest air quality data for a given latitude and longitude."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust", "ammonia",
            "methane", "european_aqi", "formaldehyde", "non_methane_volatile_organic_compounds",
            "peroxyacyl_nitrates", "secondary_inorganic_aerosol", "nitrogen_monoxide"
        ],
        "timezone": "GMT",
        "start_date": today,
        "end_date": today
    }

    # Fetch data from Open-Meteo API
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # Process first location

    # Extract hourly data
    hourly = response.Hourly()
    hourly_time = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    # Create a DataFrame with all hourly data
    hourly_data = pd.DataFrame({
        "date": hourly_time,
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
        "carbon_monoxide": hourly.Variables(2).ValuesAsNumpy(),
        "carbon_dioxide": hourly.Variables(3).ValuesAsNumpy(),
        "nitrogen_dioxide": hourly.Variables(4).ValuesAsNumpy(),
        "sulphur_dioxide": hourly.Variables(5).ValuesAsNumpy(),
        "ozone": hourly.Variables(6).ValuesAsNumpy(),
        "aerosol_optical_depth": hourly.Variables(7).ValuesAsNumpy(),
        "dust": hourly.Variables(8).ValuesAsNumpy(),
        "ammonia": hourly.Variables(9).ValuesAsNumpy(),
        "methane": hourly.Variables(10).ValuesAsNumpy(),
        "european_aqi": hourly.Variables(11).ValuesAsNumpy(),
        "formaldehyde": hourly.Variables(12).ValuesAsNumpy(),
        "non_methane_volatile_organic_compounds": hourly.Variables(13).ValuesAsNumpy(),
        "peroxyacyl_nitrates": hourly.Variables(14).ValuesAsNumpy(),
        "secondary_inorganic_aerosol": hourly.Variables(15).ValuesAsNumpy(),
        "nitrogen_monoxide": hourly.Variables(16).ValuesAsNumpy()
    })

    # ✅ Get the latest available data (most recent timestamp)
    latest_index = -1  # Last row contains the latest available data
    latest_data = hourly_data.iloc[latest_index]  # Extract last row

    # ✅ Return the latest data as a dictionary
    return {
        "Date": latest_data["date"],
        "PM10": round(latest_data["pm10"], 1),
        "PM2.5": round(latest_data["pm2_5"], 1),
        "CO": round(latest_data["carbon_monoxide"], 1),
        "CO2": round(latest_data["carbon_dioxide"], 1),
        "NO2": round(latest_data["nitrogen_dioxide"], 1),
        "SO2": round(latest_data["sulphur_dioxide"], 1),
        "Ozone": round(latest_data["ozone"], 1),
        "AQI": round(latest_data["european_aqi"], 0),  # 🔥 Whole number (0 dp)
        "NMVOC": round(latest_data["non_methane_volatile_organic_compounds"], 1),
        "NO": round(latest_data["nitrogen_monoxide"], 1),
        "SIA": round(latest_data["secondary_inorganic_aerosol"], 1),
        "NH3" : round(latest_data["ammonia"], 1)
    }
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


def fetch_district_data(district):
    """Fetch hourly air quality data for a given district."""
    if district not in coordinates_mapping:
        raise ValueError(f"Invalid district: {district}. Available districts: {list(coordinates_mapping.keys())}")

    latitude, longitude = coordinates_mapping[district]

    # ✅ Get date ranges
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

    # ✅ API request parameters
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
            "ozone", "european_aqi", "non_methane_volatile_organic_compounds",
            "secondary_inorganic_aerosol", "nitrogen_monoxide" , "ammonia"
        ],
        "timezone": "Asia/Singapore",
        "start_date": start_date,
        "end_date": yesterday,
        "domains": "cams_europe"
    }

    # ✅ Fetch data from Open-Meteo API
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # Process first location

    # ✅ Process hourly data
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
        "carbon_monoxide": hourly.Variables(2).ValuesAsNumpy(),
        "nitrogen_dioxide": hourly.Variables(3).ValuesAsNumpy(),
        "sulphur_dioxide": hourly.Variables(4).ValuesAsNumpy(),
        "ozone": hourly.Variables(5).ValuesAsNumpy(),
        "european_aqi": hourly.Variables(6).ValuesAsNumpy(),
        "non_methane_volatile_organic_compounds": hourly.Variables(7).ValuesAsNumpy(),
        "secondary_inorganic_aerosol": hourly.Variables(8).ValuesAsNumpy(),
        "nitrogen_monoxide": hourly.Variables(9).ValuesAsNumpy(),
        "ammonia": hourly.Variables(10).ValuesAsNumpy()
    }

    # ✅ Convert to DataFrame
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    
    return hourly_dataframe

def convert_hourly_to_daily(df,district):

    # Ensure 'date' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Remove time portion (keep only the date)
    df["date"] = df["date"].dt.date

    # Aggregate by date (Compute daily mean for all numeric columns)
    daily_df = df.groupby("date").mean().reset_index()

    daily_df["district"] = district

    return daily_df

# Function to generate detailed AQI comments based on the new classification
def generate_aqi_comment(median_aqi):
    if median_aqi <= 25:
        return f"""
        **🟢 Classification:** Very Low  
        **📊 Median AQI:** {median_aqi:.1f}  
        **💨 Health Impact:** Air quality is excellent, with no health risks to the population.  
        **👥 Advice for General Population:** No restrictions; outdoor activities are highly encouraged.  
        **⚠️ Advice for Vulnerable Groups (Children, elderly, and those with respiratory conditions):** No precautions necessary; enjoy outdoor activities.
        """
    
    elif median_aqi <= 50:
        return f"""
        **🟢 Classification:** Low  
        **📊 Median AQI:** {median_aqi:.1f}  
        **💨 Health Impact:** Air quality is good, with minimal risk for the general population.  
        **👥 Advice for General Population:** Outdoor activities are safe and recommended.  
        **⚠️ Advice for Vulnerable Groups:** No restrictions, but individuals with extreme sensitivity should monitor for symptoms.
        """
    
    elif median_aqi <= 75:
        return f"""
        **🟡 Classification:** Medium  
        **📊 Median AQI:** {median_aqi:.1f}  
        **💨 Health Impact:** Air quality is acceptable, but there may be minor effects for sensitive individuals.  
        **👥 Advice for General Population:** Most people can continue normal activities without concern.  
        **⚠️ Advice for Vulnerable Groups:** Individuals with respiratory conditions should take short breaks from prolonged outdoor activities.
        """
    
    elif median_aqi <= 100:
        return f"""
        **🟠 Classification:** High  
        **📊 Median AQI:** {median_aqi:.1f}  
        **💨 Health Impact:** Some people may experience mild discomfort, such as throat irritation or coughing.  
        **👥 Advice for General Population:** Outdoor activities are still safe, but people with health conditions should take precautions.  
        **⚠️ Advice for Vulnerable Groups:** Avoid prolonged outdoor exertion; wear a mask if necessary.
        """
    
    else:
        return f"""
        **🔴 Classification:** Very High  
        **📊 Median AQI:** {median_aqi:.1f}  
        **💨 Health Impact:** Air pollution levels are concerning; everyone may experience health effects, especially sensitive individuals.  
        **👥 Advice for General Population:** Limit outdoor activities, especially strenuous exercise. Consider wearing a mask if staying outdoors.  
        **⚠️ Advice for Vulnerable Groups:** Stay indoors with windows closed; use air purifiers if possible. Seek medical attention if experiencing breathing difficulties.
        """
    
def get_aqi_color(aqi):
    if aqi <= 25:
        return "🟢"  # Very Low (Green)
    elif aqi <= 50:
        return "🟢"  # Low (Lighter Green)
    elif aqi <= 75:
        return "🟡"  # Medium (Yellow)
    elif aqi <= 100:
        return "🟠"  # High (Orange)
    else:
        return "🔴"  # Very High (Red)


# User interface code here

# Set page layout
st.set_page_config(layout="wide")

# Create Tabs
tab1, tab2 = st.tabs(["🌍 Map View", "📈 Forecast Graph"])

# Sidebar
st.sidebar.header("Options")

# Default map center (Example: Warsaw)
default_lat, default_lon = 52.2298, 21.0122  # Warsaw center
zoom_level = 12

# Sidebar for User Input (Custom Location)
st.sidebar.header("Enter Custom Location")
user_lat = st.sidebar.number_input(
    "Latitude", 
    min_value=-90.000000,  # Minimum allowed latitude
    max_value=90.000000,   # Maximum allowed latitude
    value=default_lat, 
    format="%.6f"
)

user_lon = st.sidebar.number_input(
    "Longitude", 
    min_value=-180.000000,  # Minimum allowed longitude
    max_value=180.000000,   # Maximum allowed longitude
    value=default_lon, 
    format="%.6f"
)

if "show_custom_location" not in st.session_state:
    st.session_state.show_custom_location = False

if "show_geojson" not in st.session_state:
    st.session_state.show_geojson = False


# Buttons to toggle map layers
if st.sidebar.button("Show Current Location Data"):
    st.session_state.show_custom_location = True
    st.session_state.show_geojson = False
    st.session_state.map_center = [user_lat, user_lon]

if st.sidebar.button("Show Warsaw Heatmap"):
    st.session_state.show_geojson = True
    st.session_state.show_custom_location = False
    st.session_state.map_center = [default_lat, default_lon]
    



# ✅ Create the Map Centered on Warsaw
map_center = st.session_state.get("map_center", [default_lat, default_lon])
m = folium.Map(location= map_center, zoom_start=zoom_level)




# ✅ Add Markers for Custom Location
# Show custom location data
if st.session_state.show_custom_location:
    custom_location_data = fetch_latest_air_quality(user_lat, user_lon)

    print(custom_location_data)

    popup_html = f"""
    <b>🌍 Custom Location: ({user_lat}, {user_lon})</b><br>
    📅 <b>Date:</b> {today}<br>
    🏭 <b>AQI:</b> {custom_location_data["AQI"]:.0f}<br>  <!-- Whole number -->
    🏭 <b>PM10:</b> {custom_location_data["PM10"]:.1f} µg/m³<br>
    ☁️ <b>PM2.5:</b> {custom_location_data["PM2.5"]:.1f} µg/m³<br>
    💨 <b>CO:</b> {custom_location_data["CO"]:.1f} ppm<br>
    ⚠️ <b>NO2:</b> {custom_location_data["NO2"]:.1f} ppm<br>
    ⚠️ <b>SO2:</b> {custom_location_data["SO2"]:.1f} ppm<br>
    🌞 <b>Ozone:</b> {custom_location_data["Ozone"]:.1f} ppm<br>
    🏭 <b>NMVOC:</b> {custom_location_data["NMVOC"]:.1f} ppm<br>
    ⚠️ <b>NO:</b> {custom_location_data["NO"]:.1f} ppm<br>
    🏭 <b>SIA:</b> {custom_location_data["SIA"]:.1f} µg/m³<br>
    ⚠️ <b>NH3:</b> {custom_location_data["NH3"]:.1f} ppm<br>
    """

    popup = folium.Popup(popup_html, max_width=300)

    folium.Marker(
        [user_lat, user_lon],
        popup = popup,
        tooltip="Click for Air Quality Data",
        icon=folium.Icon(color="blue", icon="cloud"),
    ).add_to(m)


# Show preloaded GeoJSON
if st.session_state.show_geojson:
    geojson_file_path = "map.geojson"
    air_quality_data = fetch_air_quality_data()


    if os.path.exists(geojson_file_path):
        with open(geojson_file_path, "r", encoding="utf-8") as file:
            geojson_data = json.load(file)

        # Extract properties and dynamically update pop-ups
        def style_function(feature):
            return {
                "color": "black",  # Outline only
                "weight": 2,
                "fillOpacity": 0  # No fill color,
                
            }

        def highlight_function(feature):
            return {
                "weight": 4,
                "color": "blue",
                "fillOpacity": 0.1
            }

        for feature in geojson_data["features"]:
            district_name = feature["properties"].get("district_name", "Unknown")
            air_quality = air_quality_data.get(district_name)
            

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
                🏭 <b>NH3:</b> {air_quality["NH3"]} µg/m³<br>
                """
            
            popup = folium.Popup(popup_html, max_width=300)
            
            folium.GeoJson(
                feature,
                name="Preloaded GeoJSON",
                popup=popup,
                tooltip="Hover for details",
                style_function=style_function,
                highlight_function=highlight_function
            ).add_to(m)

                


            
    else:
        st.error(f"GeoJSON file '{geojson_file_path}' not found.")

with tab1:
    st.title("📍 Interactive Air Quality Map ")
    st_folium(m, width=900, height=600)

with tab2:
    selected_district = st.selectbox("Select a district : " , list (coordinates_mapping))


    # ✅ Button to fetch data
    if st.button("Analyse"):
        
        # ✅ Fetch and display data
        hourly_dataframe =  fetch_district_data(selected_district)

        daily_data = convert_hourly_to_daily(hourly_dataframe,selected_district)

        # Encode district names into numerical values
        label_encoder = LabelEncoder()
        daily_data["district_id"] = label_encoder.fit_transform(daily_data["district"])

        # Normalize numerical features
        features = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone",
                    "non_methane_volatile_organic_compounds",
                    "secondary_inorganic_aerosol", "nitrogen_monoxide", "european_aqi"]

        scaler = MinMaxScaler()
        daily_data[features] = scaler.fit_transform(daily_data[features])


        model = tf.keras.models.load_model("lstm_air_quality_model.h5",  compile=False)
        model.compile(optimizer= "adam", loss=MeanSquaredError(), metrics=["mae"])

        def predict_next_30_days(district_name, model, label_encoder, scaler, seq_length=30):
            """
            Predicts the next 30 days of AQI for a given district using the past 30 days of data.

            Args:
                district_name (str): The name of the district.
                model (Keras Model): The trained LSTM-GRU model.
                label_encoder (LabelEncoder): Encoder for district names.
                scaler (MinMaxScaler): Scaler used for data normalization.
                seq_length (int): Number of past days to use for prediction (default: 30).

            Returns:
                list: Predicted AQI values for the next 30 days.
            """
            # Convert district name to numerical ID
            district_id = label_encoder.transform([district_name])[0]

            # Get the last 90 days of air quality data
            last_30_days = daily_data[daily_data["district_id"] == district_id][features].iloc[-seq_length:].values
            last_30_days = last_30_days.reshape(1, seq_length, len(features))  # Reshape for model input

            # Store predictions
            future_predictions = []

            for _ in range(30):  # Predict for 30 days
                # Predict AQI
                prediction = model.predict([last_30_days, np.array([[district_id]])])

                # Ensure prediction has the correct shape
                predicted_aqi_scaled = np.zeros((1, len(features)))
                predicted_aqi_scaled[0, -1] = prediction[0][0]  # Insert prediction into last column

                # Convert back to original scale
                predicted_aqi = scaler.inverse_transform(predicted_aqi_scaled)[0, -1]
                future_predictions.append(predicted_aqi)

                # Create a new feature row with predicted AQI
                new_row = np.copy(last_30_days[:, -1, :])  # Copy the last day data
                new_row[:, -1] = predicted_aqi_scaled[0, -1]  # Replace AQI column with predicted value

                # Reshape and append to last_90_days
                new_row = new_row.reshape(1, 1, len(features))  # Ensure shape is (1, 1, num_features)
                last_30_days = np.concatenate((last_30_days[:, 1:, :], new_row), axis=1)  # Slide window forward

            return future_predictions

        predicted_30_days = predict_next_30_days(selected_district, model, label_encoder, scaler)
        
        # Generate the days for x-axis
        days = np.arange(1, 31)

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(days, predicted_30_days, marker='o', linestyle='-', color='b', label='Predicted AQI')

        # Customize the graph
        ax.set_xlabel("Days")
        ax.set_ylabel("Predicted AQI")
        ax.set_title("Predicted AQI for the Next 30 Days")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.set_ylim(min(predicted_30_days) - 5, max(predicted_30_days) + 5)

        # Show the plot in Streamlit
        st.pyplot(fig)

        
        # Calculate Average AQI
        median_aqi = round(np.median(predicted_30_days))
        aqi_comment = generate_aqi_comment(median_aqi)
        aqi_color = get_aqi_color(median_aqi)


        st.markdown("""
        ### 🌍 European AQI Scale 
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 1; text-align: center; padding: 10px; background-color: green; color: white; ">
                0 - 25
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background-color: lightgreen; color: black; ">
                26 - 50
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background-color: yellow; color: black; ">
                51 - 75
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background-color: orange; color: black; ">
                76 - 100
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background-color: red; color: white; ">
                101+
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display Median AQI with Color Code
        st.subheader("📊 Median AQI Level")
        st.markdown(f"<h2 style='text-align: center;'>{aqi_color} {int(median_aqi)}</h2>", unsafe_allow_html=True)

        # Display AQI Comment
        st.subheader("💬 Air Quality Analysis")
        st.write(aqi_comment)
            

        

