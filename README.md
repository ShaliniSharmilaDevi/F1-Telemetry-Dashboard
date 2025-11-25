F1 Telemetry Dashboard — MAXIMUM POWER

A stable, production-grade Streamlit dashboard for analyzing and comparing Formula 1 telemetry using FastF1.
It loads sessions safely, extracts fastest or selected laps, interpolates telemetry onto a common distance axis, and visualizes key metrics for one or two drivers.

---

🚀 **Features**

* Safe session loading with full validation
* Automatic detection of available events and sessions
* Fastest-lap or manual lap selection
* Telemetry smoothing option
* Metric comparison (Speed, Throttle, Brake, RPM, Gear, DRS)
* Driver color mapping based on FastF1
* Interpolation to shared distance axis for accurate overlays
* Multi-metric plotting in interactive Plotly charts
* CSV export
* Sector analysis (approximate)
* Raw telemetry preview

---
🛠 **Tech Stack**


streamlit>=1.32.0

fastf1>=3.3.6

pandas>=2.1.0

numpy>=1.25.0

plotly>=5.18.0


---

📦 **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/F1-Telemetry-Dashboard.git
cd F1-Telemetry-Dashboard
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. FastF1 cache is handled automatically at:

```
~/.fastf1_cache
```

---

▶️ **Run the Dashboard**

```bash
streamlit run dashboard.py
```

The app will open in your browser.

---

## 📊 **How It Works**

1. Select **Year → Grand Prix → Session**
2. App validates event & telemetry availability
3. Retrieves driver list from laps
4. Loads fastest lap or chosen lap
5. Extracts telemetry and applies smoothing (optional)
6. Interpolates telemetry onto a common distance grid
7. Visualizes metrics with Plotly
8. Provides exportable CSV and sector summary

---

📁 **Project Structure**

```
F1-Telemetry-Dashboard/
│── dashboard.py
│── README.md
│── requirements.txt
│── images/
│     └── preview.png (optional)
```

⚠️ Notes

* Some F1 weekends have missing or partial telemetry
* Sessions with incomplete laps may not be comparable
* Interpolation ensures accurate overlays but may introduce artifacts on low-sample telemetry

---

📜 **License**

MIT License — free to use and modify.

