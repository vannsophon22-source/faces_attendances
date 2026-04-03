import streamlit as st
import pandas as pd
import time 
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- 1. PAGE CONFIG & AUTO-REFRESH ---
st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st.title("Real-Time Attendance Monitoring")

# Refresh every 2 seconds to show new scans automatically
count = st_autorefresh(interval=2000, key="attendance_refresh")

# --- 2. GET CURRENT DATE ---
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
file_path = f"Attendance/Attendance_{date}.csv"

# --- 3. DATA DISPLAY LOGIC ---
st.subheader(f"Log for Today: {date}")

if os.path.exists(file_path):
    try:
        # Read the CSV
        df = pd.read_csv(file_path)
        
        # Display metrics (Total students scanned today)
        total_students = len(df['NAME'].unique())
        st.metric(label="Total Students Present", value=total_students)
        
        # Show the data table
        # We use use_container_width=True to make it look professional
        st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    st.info("Waiting for the first attendance scan of the day...")
    st.warning(f"File not found: {file_path}")

# --- 4. OPTIONAL: FIZZBUZZ LOGIC (Keeping your original code) ---
with st.expander("System Heartbeat (FizzBuzz)"):
    if count == 0:
        st.write("System Initializing...")
    elif count % 3 == 0 and count % 5 == 0:
        st.write("✨ FizzBuzz")
    elif count % 3 == 0:
        st.write("🔥 Fizz")
    elif count % 5 == 0:
        st.write("⚡ Buzz")
    else:
        st.write(f"Refresh Cycle: {count}")