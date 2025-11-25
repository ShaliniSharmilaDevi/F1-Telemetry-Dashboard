"""
F1 Telemetry Dashboard — MAXIMUM POWER
Production-grade Streamlit dashboard for comparing Formula 1 telemetry.
"""
import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fastf1 import plotting
from pathlib import Path
from datetime import datetime
import io
from streamlit import session_state
# --------------------------------------------------------------------
# Helper functions for event/session validation
# --------------------------------------------------------------------

def get_valid_events(year: int):
    #Return all event names available for a given year.
    try:
        schedule = fastf1.get_event_schedule(year)
        events = schedule["EventName"].tolist()
        return events
    except Exception:
        return []
def get_valid_sessions(year: int, gp_name: str):
    #  Return session types that actually contain usable telemetry.
    #  Filters out sessions that exist but contain no lap data.
    valid_sessions = []
    session_types = ["FP1", "FP2", "FP3", "Q", "R"]

    for stype in session_types:
        try:
            s = fastf1.get_session(year, gp_name, stype)
            s.load(laps=True, telemetry=True)

            if s.laps is not None and not s.laps.empty:
                valid_sessions.append(stype)
        except:
            pass  # ignore broken sessions

    return valid_sessions
# --------------------------------------------------------------------
# App configuration and FastF1 cache setup
# --------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="F1 Telemetry — MAX POWER", initial_sidebar_state="expanded")
plotting.setup_mpl()

DEFAULT_CACHE = Path.home() / ".fastf1_cache"
DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(DEFAULT_CACHE))

METRICS = ["Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]

# Map UI session names → FastF1 session codes
SESSION_TYPE_MAP = {
    "Race": "R",
    "Qualifying": "Q",
    "FP1": "FP1",
    "FP2": "FP2",
    "FP3": "FP3",
}

# --------------------------------------------------------------------
# Safe session loading helpers
# --------------------------------------------------------------------

def safe_get_session(year: int, gp: str, session_type_ui: str):
    #Load a FastF1 session with strict validation:
    #Validates event metadata
    #Ensures laps/telemetry exist
    #Converts UI names to FastF1 codes
        gp = gp.strip()
        session_type = SESSION_TYPE_MAP.get(session_type_ui, session_type_ui)

        try:
            session = fastf1.get_session(year, gp, session_type)

            # Validate event metadata before loading
            if session.event is None or session.event.get("EventName", None) is None:
                return None, f"Invalid Grand Prix name or unsupported event for {year}. FastF1 cannot find '{gp}'."

            session.load(laps=True, telemetry=True)

            if session.laps is None or session.laps.empty:
                return None, f"No laps available. Telemetry missing for {gp} {year} {session_type_ui}."

            return session, None

        except Exception as exc:
            return None, f"Error loading session: {exc}"

# --------------------------------------------------------------------
# Lap + telemetry utilities
# --------------------------------------------------------------------
def list_drivers_from_laps(laps: pd.DataFrame) -> list:
    """Return sorted unique list of drivers from laps dataframe."""
    if laps is None or laps.empty:
        return []
    return sorted(laps['Driver'].unique().tolist())


def get_driver_laps(session, driver_code: str) -> pd.DataFrame:
    """Return all laps for a given driver safely."""
    try:
        laps = session.laps.pick_driver(driver_code)
        if laps is None:
            return pd.DataFrame()
        return laps
    except Exception:
        return pd.DataFrame()


def fastest_lap_for_driver(laps_df: pd.DataFrame):
    """Return a driver's fastest lap safely."""
    try:
        if laps_df.empty:
            return None
        return laps_df.pick_fastest()
    except Exception:
        return None


def get_telemetry_safe(lap) -> pd.DataFrame:
    """Retrieve lap telemetry with column safety + fallback Distance."""
    try:
        tel = lap.get_telemetry()
        if tel is None or tel.empty:
            return None
        if 'Distance' not in tel.columns:
            tel = tel.reset_index(drop=True)
            tel['Distance'] = np.arange(len(tel))
        return tel
    except Exception:
        return None


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply centered rolling mean smoothing."""
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1, center=True).mean()


def driver_color_map(drivers: list[str]) -> dict:
    """Return FastF1 driver color mapping with safe fallback."""
    colors = {}
    for d in drivers:
        try:
            colors[d] = plotting.get_driver_color(d)
        except Exception:
            colors[d] = None
    return colors


# --------------------------------------------------------------------
# Sidebar UI: Session selection and telemetry settings
# --------------------------------------------------------------------

st.sidebar.title("🏁 F1 Telemetry — Controls")

# Theme toggle
theme_choice = st.sidebar.radio("Theme", options=["Dark", "Light"], index=0)
if theme_choice == "Light":
    st.markdown("<style>body{background-color: #f7f7f7; color:#111}</style>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("1) Load Session")

# 1. YEAR SELECTION
year = st.sidebar.selectbox("Year", options=list(range(2024, 2018, -1)), index=0)

# 2. VALID GP LIST
events = get_valid_events(year)
if not events:
    st.sidebar.error("No events found for this year.")
    st.stop()

gp_name = st.sidebar.selectbox("Grand Prix", options=events)

# 3. VALID SESSIONS WITH TELEMETRY
valid_sessions = get_valid_sessions(year, gp_name)
if not valid_sessions:
    st.sidebar.error("No valid telemetry sessions for this GP.")
    st.stop()

session_name_map = {
    "R": "Race",
    "Q": "Qualifying",
    "FP1": "FP1",
    "FP2": "FP2",
    "FP3": "FP3",
}

session_code = st.sidebar.selectbox(
    "Session Type (telemetry available)",
    options=valid_sessions,
    format_func=lambda x: session_name_map.get(x, x)
)

# Required variable for downstream code
session_type_ui = session_code

# LOAD BUTTON
cache_button = st.sidebar.button("Load / Refresh Session", key="load_refresh_session_btn")


# ADVANCED OPTIONS
st.sidebar.markdown("---")
st.sidebar.header("Advanced Settings")

telemetry_smoothing = st.sidebar.slider(
    "Telemetry smoothing window (samples)",
    min_value=1, max_value=51, value=5, step=2
)

allow_multiple_laps = st.sidebar.checkbox(
    "Allow multi-lap compare (choose laps instead of fastest)",
    value=False
)

export_csv = st.sidebar.checkbox(
    "Show export CSV button",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.header("2) Analysis Setup")

# These will be defined AFTER session loads, so just placeholder
# They will be overwritten in main area once session exists
selected_drivers = []
driver1_lap = 0
driver2_lap = 0

st.sidebar.markdown("Analysis options will appear after loading a session.")


# ---------------------------
# App state
# ---------------------------
if 'ff_session_key' not in st.session_state:
    st.session_state['ff_session_key'] = None
if 'ff_session' not in st.session_state:
    st.session_state['ff_session'] = None
if 'laps_df' not in st.session_state:
    st.session_state['laps_df'] = pd.DataFrame()
session_type_ui= session_state
session_key = f"{year}_{gp_name}_{session_type_ui}"

# Load session when button pressed or parameters changed
# Selected session type
session_type_ui = session_code   # <-- use validated session code

# Create unique session key
session_key = f"{year}_{gp_name}_{session_type_ui}"

# Load button
cache_button2 = st.sidebar.button("Load / Refresh Session", key="load_refresh_session_btn_2")


if cache_button2 or st.session_state['ff_session_key'] != session_key:
    st.session_state['ff_session_key'] = session_key
    with st.spinner("Loading session — this may take a few seconds (fastf1 cache helps)..."):
        session, err = safe_get_session(year, gp_name, session_type_ui)

    if err:
        st.session_state['ff_session'] = None
        st.session_state['laps_df'] = pd.DataFrame()
        st.error(err)
    else:
        st.session_state['ff_session'] = session
        st.session_state['laps_df'] = session.laps.reset_index(drop=True)

        ### FIX: Guard against weird/partial event metadata
        ev = getattr(session, "event", None)
        ev_name = None
        ev_year = None
        if ev is not None:
            try:
                ev_name = ev.get("EventName", None)
                ev_year = ev.get("Year", None)
            except Exception:
                ev_name = None
                ev_year = None

        if ev_name is None:
            ev_name = f"{gp_name} GP"
        if ev_year is None:
            ev_year = year

        st.success(f"Session loaded: {ev_name} {ev_year} — {session.name}")

# ---------------------------
# Early return if no session loaded
# ---------------------------
session = st.session_state['ff_session']
laps_df = st.session_state['laps_df']

if session is None or laps_df.empty:
    st.title("Formula 1 Telemetry — MAX POWER")
    st.info("Please select a valid session in the sidebar and click 'Load / Refresh Session'.")
    st.stop()

# ---------------------------
# Main UI layout
# ---------------------------
# Get stored session/laps
session = st.session_state['ff_session']
laps_df = st.session_state['laps_df']

# HARD STOP if session is None
if session is None:
    st.title("F1 Telemetry Dashboard")
    st.error("Session failed to load. Check Year / GP Name / Session Type.")
    st.stop()

# HARD STOP if laps are empty
if laps_df is None or laps_df.empty:
    st.title("F1 Telemetry Dashboard")
    st.error("Session loaded but has no laps.")
    st.stop()

st.title("F1 Telemetry Dashboard")

# SAFE event metadata extraction
ev = getattr(session, "event", None)
header_name = ""
header_year = ""

if ev is not None:
    try:
        header_name = ev.get("EventName", None)
        header_year = ev.get("Year", None)
    except Exception:
        header_name = None
        header_year = None

if not header_name:
    header_name = f"{gp_name} GP"
if not header_year:
    header_year = year

# SAFE session name
session_name_safe = getattr(session, "name", "Unknown Session")

# FINAL SAFE SUBHEADER
st.subheader(f"{header_name} ({header_year}) — {session_name_safe}")


# Driver selection UI
drivers = list_drivers_from_laps(laps_df)
if not drivers:
    st.error("No drivers available in session laps. Try a different session.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("2) Analysis Setup")
selected_drivers = st.sidebar.multiselect("Select 1 or 2 Drivers (2 for comparison)", options=drivers, default=drivers[:2])

if len(selected_drivers) == 0:
    st.warning("Select at least one driver to analyze.")
    st.stop()

# Lap pickers
if allow_multiple_laps:
    st.sidebar.markdown("Pick specific lap numbers (optional)")
    driver1_lap = st.sidebar.number_input(f"{selected_drivers[0]} lap number (leave 0 for fastest)", min_value=0, value=0)
    driver2_lap = 0
    if len(selected_drivers) > 1:
        driver2_lap = st.sidebar.number_input(f"{selected_drivers[1]} lap number (leave 0 for fastest)", min_value=0, value=0)
else:
    driver1_lap = 0
    driver2_lap = 0

selected_metric = st.sidebar.selectbox("Metric to plot", options=METRICS, index=0)
compare_multiple_metrics = st.sidebar.multiselect(
    "Also show these metrics",
    options=[m for m in METRICS if m != selected_metric],
    default=[]
)

# ---------------------------
# Data retrieval and validation
# ---------------------------

def prepare_telemetry_for_driver(session, driver_code, lap_no=0):
    laps = get_driver_laps(session, driver_code)
    if laps.empty:
        return None, f"No laps for driver {driver_code} in this session."

    lap = None
    if lap_no and lap_no > 0:
        try:
            lap_match = laps.loc[laps['LapNumber'] == int(lap_no)]
            if not lap_match.empty:
                lap = lap_match.iloc[0]
        except Exception:
            lap = None

    if lap is None:
        lap = fastest_lap_for_driver(laps)
        if lap is None:
            return None, f"Could not determine fastest lap for driver {driver_code}."

    tel = get_telemetry_safe(lap)
    if tel is None or tel.empty:
        return None, f"Telemetry not available for driver {driver_code} on selected lap."

    if telemetry_smoothing and telemetry_smoothing > 1:
        for m in [c for c in tel.columns if isinstance(c, str) and c in METRICS]:
            tel[m] = smooth_series(tel[m], window=telemetry_smoothing)

    tel['Driver'] = driver_code
    return tel, None

telemetry_frames = []
errors = []
for i, d in enumerate(selected_drivers[:2]):
    lap_no = driver1_lap if i == 0 else driver2_lap
    tel, err = prepare_telemetry_for_driver(session, d, lap_no)
    if err:
        errors.append(err)
    else:
        telemetry_frames.append(tel)

if errors:
    for e in errors:
        st.warning(e)

# -----------------------------------------
# SAFE: Build telemetry_combined
# -----------------------------------------

# If no telemetry loaded for ANY selected driver → stop here
if telemetry_frames is None or len(telemetry_frames) == 0:
    telemetry_combined = pd.DataFrame()
    st.error("No telemetry found for the selected drivers. This session may not have telemetry or the drivers have no valid laps.")
    st.stop()
else:
# Safe concat
    try:
       telemetry_combined = pd.concat(telemetry_frames, ignore_index=True)
    except Exception as e:
        telemetry_combined = pd.DataFrame()
        st.error(f"Failed to combine telemetry: {e}")
        st.stop()
if telemetry_combined.empty:
    st.error("No telemetry available for the selected drivers/session.")
    st.stop()
# -----------------------------------------
# From this point on, telemetry_combined EXISTS
# -----------------------------------------

# Ensure Distance exists
if 'Distance' not in telemetry_combined.columns:
    telemetry_combined['Distance'] = np.arange(len(telemetry_combined))

#-----------------------------------------
# GUARANTEE telemetry_combined is valid
# -----------------------------------------

# 1. Hard stop if empty
if telemetry_combined is None or telemetry_combined.empty:
    st.error("Telemetry is empty. No usable data for drivers/session.")
    st.stop()

# 2. Hard stop if required columns missing
required_columns = {"Driver", "Distance"}

missing_cols = required_columns - set(telemetry_combined.columns)
if missing_cols:
    st.error(f"Telemetry missing required columns: {missing_cols}. Cannot process this session.")
    st.stop()

# 3. Hard stop if Driver column exists but is empty
if telemetry_combined["Driver"].isna().all():
    st.error("Driver column contains no valid driver entries. Telemetry cannot be processed.")
    st.stop()

min_dist = telemetry_combined.groupby('Driver')['Distance'].min().max()
max_dist = telemetry_combined.groupby('Driver')['Distance'].max().min()
if min_dist >= max_dist:
    common_dist = np.linspace(0, 1, num=200)
else:
    common_dist = np.linspace(min_dist, max_dist, num=800)

interp_frames = []
for drv in telemetry_combined['Driver'].unique():
    df_drv = telemetry_combined[telemetry_combined['Driver'] == drv].sort_values('Distance')
    if df_drv.empty:
        continue
    df_interp = pd.DataFrame({'Distance': common_dist})
    for col in df_drv.columns:
        if col in ['Distance', 'Driver', 'Time']:
            continue
        try:
            df_interp[col] = np.interp(common_dist, df_drv['Distance'].values, df_drv[col].values)
        except Exception:
            df_interp[col] = np.nan
    df_interp['Driver'] = drv
    interp_frames.append(df_interp)

if interp_frames:
    telemetry_for_plot = pd.concat(interp_frames, ignore_index=True)
else:
    telemetry_for_plot = telemetry_combined.copy()

# ---------------------------
# Plotting
# ---------------------------

colors = driver_color_map(telemetry_for_plot['Driver'].unique())

st.markdown("## Telemetry Comparison")

fig_main = px.line(
    telemetry_for_plot,
    x='Distance',
    y=selected_metric,
    color='Driver',
    title=f"{selected_metric} over Lap Distance",
    color_discrete_map=colors
)
fig_main.update_layout(template="plotly_dark" if theme_choice == "Dark" else "plotly_white")
fig_main.update_traces(mode='lines')
fig_main.update_layout(hovermode='x unified', margin=dict(l=20, r=20, t=50, b=20))

small_figs = []
for m in compare_multiple_metrics:
    fig = px.line(
        telemetry_for_plot,
        x='Distance',
        y=m,
        color='Driver',
        color_discrete_map=colors,
        title=m
    )
    fig.update_layout(template="plotly_dark" if theme_choice == "Dark" else "plotly_white")
    fig.update_traces(mode='lines')
    small_figs.append(fig)

col_main, col_side = st.columns([3, 1])
with col_main:
    st.plotly_chart(fig_main, use_container_width=True)
    for f in small_figs:
        st.plotly_chart(f, use_container_width=True)

with col_side:
    st.markdown("### Session & Lap Info")
    if ev is not None:
        try:
            ev_country = ev.get("Country", "Unknown")
            ev_date = ev.get("Date", None)
            if hasattr(ev_date, "strftime"):
                ev_date_str = ev_date.strftime("%Y-%m-%d")
            else:
                ev_date_str = "N/A"
        except Exception:
            ev_country = "Unknown"
            ev_date_str = "N/A"
    else:
        ev_country = "Unknown"
        ev_date_str = "N/A"

    st.write(f"**Event:** {header_name} ({ev_country})")
    st.write(f"**Date:** {ev_date_str}")
    st.write(f"**Session:** {session.name}")

    st.markdown("---")
    st.markdown("### Drivers Selected")
    for d in telemetry_for_plot['Driver'].unique():
        st.markdown(f"- {d}")

    if export_csv:
        csv_buffer = io.StringIO()
        telemetry_for_plot.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        st.download_button(
            label="Export telemetry CSV",
            data=csv_bytes,
            file_name=f"telemetry_{gp_name}_{session.name}.csv",
            mime='text/csv'
        )

# ---------------------------
# Sector split visualization (simple approximation)
# ---------------------------
st.markdown("## Sector / Segment Analysis (approx)")

dist_min = telemetry_for_plot['Distance'].min()
dist_max = telemetry_for_plot['Distance'].max()
if pd.notna(dist_min) and pd.notna(dist_max) and dist_max > dist_min:
    sectors = [
        dist_min,
        dist_min + (dist_max - dist_min) / 3,
        dist_min + 2 * (dist_max - dist_min) / 3,
        dist_max
    ]
    sector_labels = ['S1', 'S2', 'S3']

    sector_summaries = []
    for drv in telemetry_for_plot['Driver'].unique():
        df_drv = telemetry_for_plot[telemetry_for_plot['Driver'] == drv]
        row = {'Driver': drv}
        for i in range(3):
            seg = df_drv[(df_drv['Distance'] >= sectors[i]) & (df_drv['Distance'] <= sectors[i+1])]
            if seg.empty:
                row[f'{sector_labels[i]}_avg_speed'] = np.nan
                row[f'{sector_labels[i]}_max_speed'] = np.nan
                row[f'{sector_labels[i]}_throttle'] = np.nan
            else:
                row[f'{sector_labels[i]}_avg_speed'] = seg['Speed'].mean() if 'Speed' in seg.columns else np.nan
                row[f'{sector_labels[i]}_max_speed'] = seg['Speed'].max() if 'Speed' in seg.columns else np.nan
                row[f'{sector_labels[i]}_throttle'] = seg['Throttle'].mean() if 'Throttle' in seg.columns else np.nan
        sector_summaries.append(row)

    df_sector = pd.DataFrame(sector_summaries)
    st.dataframe(df_sector.set_index('Driver'))
else:
    st.info("Not enough Distance info to perform sector analysis.")

# ---------------------------
# Raw telemetry preview
# ---------------------------
st.markdown("## Raw Telemetry Preview (sample)")
st.dataframe(telemetry_for_plot.head(200), use_container_width=True)

st.markdown("---")
st.markdown(
    "**Notes:**\n"
    "- Some sessions can have incomplete telemetry.\n"
    "- Use the smoothing slider to remove jitter for line comparisons.\n"
    "- Telemetry is interpolated onto a common distance base for comparison."
)
