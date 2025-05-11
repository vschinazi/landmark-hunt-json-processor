import streamlit as st
import pandas as pd
import numpy as np
import json
import zipfile
from io import BytesIO
import geopandas as gpd
from shapely.geometry import MultiPoint, Point
from pyproj import CRS
import pydeck as pdk

# ------------------------
# Helper Functions
# ------------------------

def safe_extract(obj, field):
    return obj.get(field, None) if isinstance(obj, dict) else None

def read_json_files(uploaded_files):
    json_contents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file) as z:
                for filename in z.namelist():
                    if filename.endswith(".json") and not filename.startswith(("__MACOSX", "._")):
                        with z.open(filename) as f:
                            try:
                                content = f.read().decode("utf-8")
                            except UnicodeDecodeError:
                                f.seek(0)
                                content = f.read().decode("latin-1", errors="ignore")
                            json_contents.append(content)
        elif uploaded_file.name.endswith(".json"):
            try:
                content = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode("latin-1", errors="ignore")
            json_contents.append(content)
    return json_contents

def extract_landmark_data(content):
    data = json.loads(content)
    userID = safe_extract(safe_extract(data, "participantInfo"), "userID")
    landmarks = data.get("landmarks", [])
    records = []
    for lm in landmarks:
        records.append({
            "userID": userID,
            "latitude": safe_extract(lm, "latitude"),
            "longitude": safe_extract(lm, "longitude"),
            "timestamp": safe_extract(lm, "timestamp")
        })
    return pd.DataFrame(records)

def calculate_convex_hull_area(landmarks_df):
    results = []
    for user in landmarks_df['userID'].unique():
        user_data = landmarks_df[landmarks_df['userID'] == user]
        coords = list(zip(user_data['longitude'], user_data['latitude']))
        coords = [(lon, lat) for lon, lat in coords if pd.notna(lon) and pd.notna(lat)]
        if len(coords) >= 3:
            mean_lon = np.mean([lon for lon, _ in coords])
            mean_lat = np.mean([lat for _, lat in coords])
            utm_zone = int((mean_lon + 180) / 6) + 1
            utm_crs = f"EPSG:{32600 + utm_zone}" if mean_lat >= 0 else f"EPSG:{32700 + utm_zone}"

            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*coords)), crs="EPSG:4326")
            gdf = gdf.to_crs(utm_crs)
            hull = gdf.geometry.unary_union.convex_hull  # Use unary_union for safety

            area_km2 = hull.area / 1e6  # Convert m² to km²
        else:
            area_km2 = np.nan
        results.append({
            "userID": user,
            "convex_hull_area_km2": area_km2,
            "num_landmarks": len(coords)
        })
    return pd.DataFrame(results)


def extract_trial_data(content, source_filter=None, task_filter=None):
    data = json.loads(content)
    userID = safe_extract(safe_extract(data, "participantInfo"), "userID")
    sessionID = safe_extract(data, "sessionID")
    daily_results = data.get("dailyResults", [])
    records = []

    for entry in daily_results:
        results_by_source = safe_extract(entry, "resultsBySource")
        if results_by_source:
            for source_entry in results_by_source:
                if not isinstance(source_entry, dict):
                    continue
                for task_type in ["pointing", "distance", "mapping"]:
                    if task_filter and task_type != task_filter:
                        continue
                    trials = safe_extract(source_entry, task_type)
                    if trials:
                        for trial in trials:
                            if source_filter and safe_extract(trial, "taskSource") != source_filter:
                                continue
                            tau = safe_extract(trial, "kendallTau")
                            accuracy = 50 + 50 * tau if tau is not None else None
                            record = {
                                "userID": userID,
                                "sessionID": sessionID,
                                "taskSource": safe_extract(trial, "taskSource"),
                                "taskType": task_type,
                                "error": safe_extract(trial, "error"),
                                "kendallTau": tau,
                                "distance_accuracy": accuracy,
                                "r2": safe_extract(trial, "r2") or safe_extract(trial, "rSquared"),
                                "timestamp": safe_extract(trial, "timestamp")
                            }
                            records.append(record)
    return pd.DataFrame(records)

# ------------------------
# Streamlit UI
# ------------------------

st.title("📖 Landmark Hunter: Explorer")

# Navigation to Converter
st.markdown("[📥 Go to Converter](https://landmark-hunt-json-converter.streamlit.app/)")


uploaded_files = st.file_uploader(
    "Upload JSON files or ZIP archives", 
    type=["json", "zip"], 
    accept_multiple_files=True
)

if uploaded_files:
    json_files = read_json_files(uploaded_files)

    # Section 1: Participant & Session Overview
    with st.expander("📊 Participant & Session Overview"):
        session_data = []
        for content in json_files:
            data = json.loads(content)
            p_info = safe_extract(data, "participantInfo")
            userID = safe_extract(p_info, "userID")
            gender = safe_extract(p_info, "gender")
            age = safe_extract(p_info, "age")
            sessionID = safe_extract(data, "sessionID")
            total_landmarks = len(data.get("landmarks", []))
            deleted_landmarks = data.get("deletedLandmarks", 0)
            session_data.append({
                "userID": userID, "gender": gender, "age": age,
                "sessionID": sessionID, "total_landmarks": total_landmarks,
                "deleted_landmarks": deleted_landmarks
            })
        df_session = pd.DataFrame(session_data)
        st.dataframe(df_session)
        st.download_button("📥 Download Overview", df_session.to_csv(index=False), "session_overview.csv")

    # Section 2: Landmark Coordinates
    with st.expander("📍 Landmark Coordinates"):
        all_landmarks = pd.concat([extract_landmark_data(c) for c in json_files], ignore_index=True)
        st.dataframe(all_landmarks)
        st.download_button("📥 Download Landmarks", all_landmarks.to_csv(index=False), "landmark_coordinates.csv")
        if st.checkbox("🗺️ Show Landmarks on Map"):
            st.map(all_landmarks.rename(columns={'latitude': 'lat', 'longitude': 'lon'}))

    # Section 3: Task-Level Trial Data
    with st.expander("🎯 Task-Level Trial Data"):
        source_filter = st.selectbox("Filter by Task Source", ["All", "assessment", "manual", "reminder"])
        task_filter = st.selectbox("Filter by Task Type", ["All", "pointing", "distance", "mapping"])

        source_f = None if source_filter == "All" else source_filter
        task_f = None if task_filter == "All" else task_filter

        all_trials = pd.concat(
            [extract_trial_data(c, source_f, task_f) for c in json_files], 
            ignore_index=True
        )
        st.dataframe(all_trials)
        st.download_button("📥 Download Trials", all_trials.to_csv(index=False), "trial_data.csv")

    # Section 4: Spatial Footprint (Convex Hull Area)
    with st.expander("🗺️ Spatial Footprint (Convex Hull Area)"):
        all_landmarks = pd.concat([extract_landmark_data(c) for c in json_files], ignore_index=True)
        convex_hull_df = calculate_convex_hull_area(all_landmarks)
        st.dataframe(convex_hull_df)
        st.download_button("📥 Download Convex Hull Data", convex_hull_df.to_csv(index=False), "convex_hull_areas.csv")

        if st.checkbox("🗺️ Show Convex Hulls with Landmarks (Interactive Map)"):
            landmark_layer = pdk.Layer(
                "ScatterplotLayer",
                data=all_landmarks.rename(columns={"latitude": "lat", "longitude": "lon"}),
                get_position='[lon, lat]',
                get_color='[0, 128, 255, 160]',
                get_radius=50,
            )

            polygons = []
            for user in all_landmarks['userID'].unique():
                user_data = all_landmarks[all_landmarks['userID'] == user]
                coords = list(zip(user_data['longitude'], user_data['latitude']))
                coords = [(lon, lat) for lon, lat in coords if pd.notna(lon) and pd.notna(lat)]
                if len(coords) >= 3:
                    hull = MultiPoint(coords).convex_hull
                    if hull.geom_type == 'Polygon':
                        polygons.append({
                            'userID': user,
                            'polygon': [list(hull.exterior.coords)]
                        })

            if polygons:
                polygon_layer = pdk.Layer(
                    "PolygonLayer",
                    data=pd.DataFrame(polygons),
                    get_polygon='polygon',
                    get_fill_color='[255, 0, 0, 80]',
                    stroked=True,
                    get_line_color='[200, 30, 0, 160]',
                    line_width_min_pixels=1,
                )
                layers = [landmark_layer, polygon_layer]
            else:
                layers = [landmark_layer]

            view_state = pdk.ViewState(
                latitude=all_landmarks['latitude'].mean(),
                longitude=all_landmarks['longitude'].mean(),
                zoom=12,
                pitch=0,
            )

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=layers,
            ))

    # Navigation back to Converter
    st.markdown("[📥 Go to Converter](https://landmark-hunt-json-converter.streamlit.app/)")

else:
    st.info("📂 Please upload JSON or ZIP files to explore data.")
