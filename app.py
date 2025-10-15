# WDM Bootcamp – Diffusion (MSD) Analyzer
# Build: streamlit run app.py

import re
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="MSD → Diffusion Coefficient", layout="wide")

# -----------------------------
# 0) URL registry (your links)
# -----------------------------
URLS = {
    "Au": {
        3000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Au/Au-3000K.log",
        4000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Au/Au-4000K.log",
        5000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Au/Au-5000K.log",
        6000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Au/Au-6000K.log",
        7000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Au/Au-7000K.log",
        8000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Au/Au-8000K.log",
        9000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Au/Au-9000K.log",
        10000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Au/Au-10000K.log",
    },
    "Cu": {
        3000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Cu/Cu-3000K.log",
        4000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Cu/Cu-4000K.log",
        5000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Cu/Cu-5000K.log",
        6000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Cu/Cu-6000K.log",
        7000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Cu/Cu-7000K.log",
        8000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Cu/Cu-8000K.log",
        9000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Cu/Cu-9000K.log",
        10000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Cu/Cu-10000K.log",
    },
    "Pt": {
        3000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Pt/Pt-3000K.log",
        4000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Pt/Pt-4000K.log",
        5000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Pt/Pt-5000K.log",
        6000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Pt/Pt-6000K.log",
        7000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Pt/Pt-7000K.log",
        8000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Pt/Pt-8000K.log",
        9000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Pt/Pt-9000K.log",
        10000: "https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Pt/Pt-10000K.log",
    },
}

DIMENSION = 3  # 3D liquids -> MSD = 2*d*D*t so D = slope / (2*3) = slope/6

# -----------------------------------
# 1) Helpers: download & robust parse
# -----------------------------------
@st.cache_data(show_spinner=False)
def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def lammps_log_to_df(text: str) -> pd.DataFrame:
    """
    Parse a LAMMPS log that contains a preamble and then a table beginning at the line
    whose first token is 'Step'. Keep rows until the table ends.
    """
    lines = text.splitlines()
    # find the table header line that starts with 'Step'
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Step"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find 'Step' header in log.")

    # header columns
    header = re.split(r"\s+", lines[start_idx].strip())
    # data block: subsequent lines until a blank or a non-numeric row shows up
    data_rows = []
    for ln in lines[start_idx + 1:]:
        s = ln.strip()
        if not s:
            break
        # keep only rows that begin with a number (Step is integer)
        first = s.split()[0]
        if re.fullmatch(r"[-+]?\d+", first) or re.fullmatch(r"\d+\.\d*", first):
            data_rows.append(s)
        else:
            # stop when table ends
            break

    df = pd.read_csv(io.StringIO("\n".join([lines[start_idx]] + data_rows)),
                     delim_whitespace=True, engine="python")
    # Flexible column detection for Time and MSD
    time_col = None
    for cand in ["Time", "time", "t"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        # sometimes second column is time
        time_col = df.columns[1]

    msd_col = None
    for c in df.columns:
        if re.search(r"msd", c, re.IGNORECASE):
            msd_col = c
            break
    if msd_col is None:
        # common LAMMPS compute name
        for c in df.columns:
            if "c_Msd" in c or "c_msd" in c:
                msd_col = c
                break
    if msd_col is None:
        raise ValueError(f"Could not find MSD column in columns: {list(df.columns)}")

    # restrict to physical data region mentioned in slides: t >= 5 ps
    if df[time_col].dtype.kind in "ifu":
        df = df[df[time_col] >= 5].copy()

    # standardize names
    df = df.rename(columns={time_col: "Time", msd_col: "MSD"}).reset_index(drop=True)
    return df[["Time", "MSD"]]

def fit_diffusion(df: pd.DataFrame) -> dict:
    """
    Fit MSD vs Time linearly: MSD = m*t + b -> D = m / (2*d)
    Returns dict with slope, intercept, D, R2.
    """
    x = df["Time"].to_numpy(dtype=float)
    y = df["MSD"].to_numpy(dtype=float)
    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs[0], coeffs[1]
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    D = m / (2 * DIMENSION)
    return {"slope": m, "intercept": b, "D": D, "R2": r2}

# --------------------------
# 2) Sidebar controls (UI)
# --------------------------
st.sidebar.header("Controls")

elements = list(URLS.keys())
choice_elements = st.sidebar.multiselect("Elements", elements, default=elements)

# collect all available temperatures for selected elements
temps_available = sorted({T for el in choice_elements for T in URLS[el].keys()})
choice_temps = st.sidebar.multiselect("Temperatures (K)", temps_available, default=temps_available)

show_points = st.sidebar.checkbox("Show raw points on MSD plot", value=True)
show_fit = st.sidebar.checkbox("Show linear fit on MSD plot", value=True)

st.title("Temperature Dependence of Diffusion (Cu, Au, Pt)")
st.caption("Parses LAMMPS logs → plots MSD(t) → fits D = slope/(2·d) → plots D vs T. Assumes 3D (d=3).")

# --------------------------
# 3) Main analyses
# --------------------------
msd_plots = []
records = []  # per (element, T) fit results

for el in choice_elements:
    for T in sorted(URLS[el].keys()):
        if T not in choice_temps:
            continue
        url = URLS[el][T]
        try:
            text = fetch_text(url)
            df = lammps_log_to_df(text)
            # Fit
            fit = fit_diffusion(df)

            # Plot MSD vs Time
            fig = px.scatter(df, x="Time", y="MSD", title=f"{el} @ {T} K — MSD vs Time",
                             opacity=1.0 if show_points else 0.0)
            if show_fit:
                xline = np.linspace(df["Time"].min(), df["Time"].max(), 200)
                yline = fit["slope"] * xline + fit["intercept"]
                fig.add_scatter(x=xline, y=yline, mode="lines", name=f"Fit (R²={fit['R2']:.3f})")
            fig.update_layout(height=420, xaxis_title="Time (ps)", yaxis_title="MSD (units²)")
            msd_plots.append(fig)

            # Store result
            records.append({"Element": el, "Temperature (K)": T,
                            "Slope (MSD/time)": fit["slope"],
                            "Intercept": fit["intercept"],
                            "R²": fit["R2"],
                            "D (slope/6)": fit["D"]})
        except Exception as e:
            st.error(f"{el} {T}K: {e}")

# Show MSD figures
if msd_plots:
    st.subheader("MSD vs Time (per temperature)")
    for fig in msd_plots:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one element and temperature in the sidebar.")

# Table + D vs T
if records:
    st.subheader("Fit Results and Diffusion Coefficients")
    results_df = pd.DataFrame.from_records(records).sort_values(["Element", "Temperature (K)"])
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Diffusion Coefficient vs Temperature")
    figD = px.line(results_df, x="Temperature (K)", y="D (slope/6)", color="Element",
                   markers=True, title="D vs Temperature (assuming 3D: D = slope / 6)")
    figD.update_layout(height=500, xaxis_title="Temperature (K)", yaxis_title="D (units² / ps)")
    st.plotly_chart(figD, use_container_width=True)

    with st.expander("Notes / Interpretation Guide"):
        st.markdown(
            """
            **How to interpret:**
            - For normal diffusion in 3D liquids, MSD grows linearly with time. The slope is `2·d·D = 6D`.
            - So `D = slope / 6`. Larger D ⇒ faster atomic motion (weaker binding / higher T).
            - Expect **D to increase with temperature** within the liquid regime.
            - Across elements, heavier atoms and stronger bonding (e.g., Pt) typically diffuse more slowly (smaller D) than lighter/weaker-bonded (e.g., Cu).
            
            **Caveats:**
            - Use only the **linear MSD region** (this app already drops the equilibration before t = 5 ps per instructions).
            - If `R²` is low, the data might not be strictly linear (ballistic at very short times or noise at very long times).
            - Units: keep consistent with your simulation (Time likely ps; MSD in length² of your LAMMPS units).
            """
        )
else:
    st.stop()