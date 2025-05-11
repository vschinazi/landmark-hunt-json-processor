import streamlit as st
import pandas as pd
import json
import os
from io import BytesIO
from datetime import datetime
import numpy as np
from shapely.geometry import MultiPoint
import geopandas as gpd
from pyproj import CRS
import zipfile

def safe_extract(obj, field):
    return obj.get(field, None)

def calculate_convex_hull_area(landmarks):
    coords = [(safe_extract(l, "longitude"), safe_extract(l, "latitude")) for l in landmarks]
    coords = [(lon, lat) for lon, lat in coords if lon is not None and lat is not None]

    if len(coords) >= 3:
        mean_lon = np.mean([lon for lon, _ in coords])
        mean_lat = np.mean([lat for _, lat in coords])
        utm_zone = int((mean_lon + 180) / 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone}" if mean_lat >= 0 else f"EPSG:{32700 + utm_zone}"

        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*coords)), crs="EPSG:4326")
        gdf = gdf.to_crs(utm_crs)
        hull = gdf.geometry.union_all().convex_hull
        return hull.area / 1e6
    return None

def read_json_files(uploaded_files):
    json_contents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file) as z:
                for filename in z.namelist():
                    if filename.endswith(".json") and "__MACOSX" not in filename:
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

def process_uploaded_files(uploaded_files, source_filter):
    json_files = read_json_files(uploaded_files)
    results = []
    for content in json_files:
        try:
            df = process_file(content, source_filter)
            results.append(df)
        except json.JSONDecodeError:
            st.warning("⚠️ Skipped a file due to invalid JSON format.")
            continue
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def process_file(content, source_filter):
    data = json.loads(content)
    p_info = data.get("participantInfo", {})
    userID = safe_extract(p_info, "userID")
    gender = safe_extract(p_info, "gender")
    age = safe_extract(p_info, "age")
    sessionID = safe_extract(data, "sessionID")

    landmarks = data.get("landmarks", [])
    total_landmarks = len(landmarks)
    deleted_landmarks = data.get("deletedLandmarks", 0)
    convex_hull_area = calculate_convex_hull_area(landmarks)

    daily_results = data.get("dailyResults", [])
    dates = [safe_extract(day, "date") for day in daily_results if safe_extract(day, "date") is not None]
    unique_dates = sorted(set(pd.to_datetime(dates)))
    total_days_used = len(unique_dates)
    first_use_date = min(unique_dates).strftime("%Y-%m-%d") if unique_dates else None
    last_use_date = max(unique_dates).strftime("%Y-%m-%d") if unique_dates else None
    total_duration_days = (max(unique_dates) - min(unique_dates)).days if unique_dates else None
    engagement_ratio = total_days_used / total_landmarks if total_landmarks > 0 else None

    days = sorted(pd.to_datetime(dates).unique())
    gaps = [(days[i+1] - days[i]).days for i in range(len(days)-1)] if len(days) > 1 else []
    longest_gap_days = max(gaps) if gaps else None
    streak = 1
    longest_streak_days = 1
    for i in range(1, len(days)):
        if (days[i] - days[i-1]).days == 1:
            streak += 1
            longest_streak_days = max(longest_streak_days, streak)
        else:
            streak = 1

    trial_data = {"pointing": [], "distance": [], "mapping": [], "kendallTau": []}
    daily_metrics = {}
    weekly_metrics = {}

    ordered_tasks = ["pointing", "distance", "kendallTau", "mapping"]

    for day_idx, entry in enumerate(daily_results, start=1):
        results_by_source = safe_extract(entry, "resultsBySource")
        current_week = (day_idx - 1) // 7 + 1
        day_data = {"pointing": [], "distance": [], "mapping": [], "kendallTau": []}

        if results_by_source:
            for source_entry in results_by_source:
                if not isinstance(source_entry, dict):
                    continue
                for task_type in ["pointing", "distance", "mapping"]:
                    trials = safe_extract(source_entry, task_type)
                    if trials:
                        for trial in trials:
                            if safe_extract(trial, "taskSource") != source_filter:
                                continue
                            if task_type == "pointing":
                                error = safe_extract(trial, "error")
                                if error is not None:
                                    trial_data["pointing"].append(error)
                                    day_data["pointing"].append(error)
                            elif task_type == "distance":
                                tau = safe_extract(trial, "kendallTau")
                                if tau is not None:
                                    accuracy = max(0, min(100, 50 + 50 * tau))
                                    trial_data["distance"].append(accuracy)
                                    trial_data["kendallTau"].append(tau)
                                    day_data["distance"].append(accuracy)
                                    day_data["kendallTau"].append(tau)
                            elif task_type == "mapping":
                                r2 = safe_extract(trial, "r2") or safe_extract(trial, "rSquared")
                                if r2 is not None:
                                    trial_data["mapping"].append(r2)
                                    day_data["mapping"].append(r2)

        for task in ordered_tasks:
            if day_data[task]:
                label = (
                    "Pointing_Error" if task == "pointing" else
                    "Distance_Accuracy" if task == "distance" else
                    "Kendall_Tau" if task == "kendallTau" else
                    "Mapping_R2"
                )
                daily_metrics[f"Day{day_idx}_{source_filter}_{label}"] = np.mean(day_data[task])

        for key in ordered_tasks:
            if day_data[key]:
                week_key = f"Week{current_week}_{source_filter}_{key}"
                weekly_metrics.setdefault(week_key, []).extend(day_data[key])

    week_avg_metrics = {}
    for task in ordered_tasks:
        matching_keys = [k for k in weekly_metrics if k.endswith(task)]
        for week_key in sorted(matching_keys):
            values = weekly_metrics[week_key]
            week_prefix = "_".join(week_key.split("_")[:2])
            label = (
                "Pointing_Error" if task == "pointing" else
                "Distance_Accuracy" if task == "distance" else
                "Kendall_Tau" if task == "kendallTau" else
                "Mapping_R2"
            )
            week_avg_metrics[f"{week_prefix}_{label}"] = np.mean(values)

    result = {
        "userID": userID,
        "gender": gender,
        "age": age,
        "sessionID": sessionID,
        "total_landmarks": total_landmarks,
        "deleted_landmarks": deleted_landmarks,
        "convex_hull_area": convex_hull_area,
        "total_days_used": total_days_used,
        "first_use_date": first_use_date,
        "last_use_date": last_use_date,
        "total_duration_days": total_duration_days,
        "engagement_ratio": engagement_ratio,
        "longest_gap_days": longest_gap_days,
        "longest_streak_days": longest_streak_days
    }

    # Ordered Overall Metrics
    for task in ordered_tasks:
        label = (
            "Pointing_Error" if task == "pointing" else
            "Distance_Accuracy" if task == "distance" else
            "Kendall_Tau" if task == "kendallTau" else
            "Mapping_R2"
        )
        result[f"Overall_{source_filter}_{label}"] = (
            np.mean(trial_data[task]) if trial_data[task] else None
        )

    result.update(daily_metrics)
    result.update(week_avg_metrics)

    return pd.DataFrame([result])

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Processed Data')
    return output.getvalue()

# Streamlit App Interface
st.title("Landmark Hunt JSON Converter")

uploaded_files = st.file_uploader(
    "Upload JSON or ZIP files", 
    type=["json", "zip"], 
    accept_multiple_files=True
)

source_filter = st.selectbox(
    "Select Data Source", 
    ["assessment", "manual", "reminder"]
)

if st.button("Process Data"):
    if uploaded_files:
        df = process_uploaded_files(uploaded_files, source_filter)
        st.success("✅ Processing Completed!")
        st.dataframe(df)

        excel_data = convert_df_to_excel(df)
        st.download_button(
            label="📥 Download as Excel",
            data=excel_data,
            file_name=f"{source_filter}_Details.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("⚠️ Please upload at least one JSON or ZIP file.")
