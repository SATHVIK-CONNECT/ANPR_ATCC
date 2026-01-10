import streamlit as st
import cv2
import tempfile
import pandas as pd
import plotly.express as px
from traffic_engine import TrafficLightManager
import os
from collections import Counter



# --- CONFIGURATION ---
TRAFFIC_MODEL_PATH = "model1/best.pt"
VIOLATION_MODEL_PATH = "model2/best2.pt" 

st.set_page_config(page_title="AI Traffic Command Center", layout="wide")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("Control Panel")
    
    # --- MODE SELECTION ---
    st.markdown("### Select Functionality")
    system_mode = st.radio(
        "Choose Operation Mode:",
        ("Traffic Analysis", "Traffic Violation Detection"),
        index=0
    )
    st.markdown("---")
    
    st.subheader("System Status")
    status_indicator = st.empty()
    status_indicator.info("Ready to Start")
    
    # Simulation Buttons
    if 'run_simulation' not in st.session_state: st.session_state['run_simulation'] = False
    def start_logic(): st.session_state['run_simulation'] = True
    def stop_logic(): st.session_state['run_simulation'] = False

    col1, col2 = st.columns(2)
    with col1: st.button("START", on_click=start_logic, type="primary", use_container_width=True)
    with col2: st.button("STOP", on_click=stop_logic, type="secondary", use_container_width=True)

    # Settings
    st.markdown("---")
    st.subheader("Settings")
    
    if system_mode == "Traffic Analysis":
        conf_traffic = st.slider("Vehicle Detection Confidence", 0.1, 1.0, 0.35)
    else:
        conf_helmet = st.slider("No Helmet Confidence", 0.1, 1.0, 0.80)
        conf_mobile = st.slider("Mobile Use Confidence", 0.1, 1.0, 0.50)

# --- LOAD ENGINE ---
@st.cache_resource
def load_engine():
    return TrafficLightManager(TRAFFIC_MODEL_PATH, VIOLATION_MODEL_PATH)

traffic_system = load_engine()

st.title(f"{system_mode} Dashboard")

uploaded_file = st.file_uploader(" Upload CCTV Footage", type=['mp4', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # --- DYNAMIC UI LAYOUT ---
    if system_mode == "Traffic Analysis":
        # Metric Row
        col_sig, col_veh, col_alert = st.columns(3)
        with col_sig: 
            st.markdown("**Signal Status**")
            signal_metric = st.empty()
        with col_veh: 
            st.markdown("**Cumulative Density**")
            vehicle_metric = st.empty()
        with col_alert: 
            st.markdown("**Status**")
            alert_metric = st.empty()
        
        # Main Layout: Video on Left, Graphs on Right
        col_video, col_graphs = st.columns([1.5, 1])
        with col_video:
            video_placeholder = st.empty()
        with col_graphs:
            st.subheader("Live Analytics")
            tab1, tab2 = st.tabs(["Vehicle Class Distribution", "Lane Density"])
            with tab1:
                class_chart_placeholder = st.empty()
            with tab2:
                lane_chart_placeholder = st.empty()
        
    else:
        # Violation Layout
        col_tot, col_hel, col_mob = st.columns(3)
        with col_tot: 
            st.markdown("**Total Violations**")
            viol_metric = st.empty()
        with col_hel: 
            st.markdown("**No Helmet**")
            helmet_metric = st.empty()
        with col_mob: 
            st.markdown("**Mobile Usage**")
            mobile_metric = st.empty()
            
        video_placeholder = st.empty()
        st.subheader("Live Violation Evidence")
        evidence_gallery = st.empty()

    # --- MAIN LOOP ---
    if st.session_state['run_simulation']:
        traffic_system.reset()
        
        # Update settings
        if system_mode == "Traffic Analysis":
            traffic_system.model.conf = conf_traffic
            mode_key = "Analysis"
        else:
            traffic_system.violation_engine.helmet_conf = conf_helmet
            traffic_system.violation_engine.mobile_conf = conf_mobile
            mode_key = "Violation"

        frame_count = 0
        while cap.isOpened() and st.session_state['run_simulation']:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            frame = cv2.resize(frame, (1280, 720))
            
            # PROCESS FRAME
            processed_frame, lane_data, active_lane, green_time, congestion, \
            wait_times, v_counts, total_v, viol_stats, recent_violations = \
            traffic_system.process_frame(frame, frame_count, mode=mode_key)
            
            # DISPLAY VIDEO
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # --- UPDATE GRAPHS & METRICS ---
            if system_mode == "Traffic Analysis":
                colors = {k: '🔴' for k in lane_data.keys()}
                if active_lane in colors: colors[active_lane] = '🟢'
                signal_metric.markdown(f"<h2>{colors.get(active_lane,'🟢')} {active_lane} : {int(green_time)}s</h2>", unsafe_allow_html=True)
                vehicle_metric.metric("Total Vehicles", total_v)
                
                if congestion: alert_metric.error(congestion)
                else: alert_metric.success("Flow Normal")
                
                if lane_data:
                    df_lane = pd.DataFrame({'Lane': list(lane_data.keys()), 'Density': list(lane_data.values())})
                    fig_lane = px.bar(df_lane, x='Lane', y='Density', color='Lane', title="Current Lane Load")
                    lane_chart_placeholder.plotly_chart(fig_lane, use_container_width=True, key=f"lane_{frame_count}")

                # We fetch the live dictionary of unique vehicles from the engine
                unique_classes = traffic_system.unique_vehicle_classes
                if unique_classes:
                    counts = Counter(unique_classes.values())
                    df_class = pd.DataFrame({'Vehicle Type': list(counts.keys()), 'Count': list(counts.values())})
                    
                    # Create a colorful Bar Chart
                    fig_class = px.bar(
                        df_class, 
                        x='Count', 
                        y='Vehicle Type', 
                        orientation='h', # Horizontal bars look better for lists
                        color='Vehicle Type', 
                        title="Total Vehicles by Class"
                    )
                    class_chart_placeholder.plotly_chart(fig_class, use_container_width=True, key=f"class_{frame_count}")
            
            else:
                # Violation Updates
                viol_metric.metric("Total", viol_stats["Total Violations"])
                helmet_metric.metric("No Helmet", viol_stats["No Helmet"])
                mobile_metric.metric("Mobile", viol_stats["Mobile Usage"])
                
                # Gallery Update
                if recent_violations:
                    with evidence_gallery.container():
                        cols = st.columns(4)
                        recents = list(recent_violations)[-4:]
                        for i, item in enumerate(recents):
                            cap_text = f"{item['plate']}\n{item['violations'][0]}"
                            cols[i].image(item['image'], caption=cap_text, use_container_width=True)

    cap.release()