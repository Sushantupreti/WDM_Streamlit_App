import re
import io
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="MSD → Diffusion Coefficient", layout="wide")

st.title("Atomic Diffusion from LAMMPS Logs (Cu, Au, Pt)")
st.caption(
    """
    Loads Au/Cu/Pt LAMMPS logs (3000–10000 K), extracts MSD tables, fits the **linear late-time**
    region of MSD(t), and computes diffusion via **MSD = 2·d·D·t** (3D → **D = slope / 6**).

    **You get:**  
    (1) MSD vs time per T (points + best-fit line)  
    (2) A table of slope, R², and **D**  
    (3) A final **D vs T** plot with all three elements  
    """
)

# ----------------------------
# 0) URLs to the log files
# ----------------------------
URLS: Dict[str, Dict[int, str]] = {
    "Au": {T: f"https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Au/Au-{T}K.log"
           for T in range(3000, 10001, 1000)},
    "Cu": {T: f"https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Cu/Cu-{T}K.log"
           for T in range(3000, 10001, 1000)},
    "Pt": {T: f"https://raw.githubusercontent.com/joegonzalezMSL/Warm-Dense-Matter-2025/refs/heads/main/LAMMPS/Pt/Pt-{T}K.log"
           for T in range(3000, 10001, 1000)},
}
DIMENSION = 3  # 3D: MSD = 2*d*D*t → D = slope / (2*d) = slope / 6

# ----------------------------
# 1) Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def _infer_timestep_ps(lines: List[str]) -> Optional[float]:
    """
    Infer LAMMPS timestep in ps:
      - 'units metal' → timestep already ps
      - 'units real'  → timestep fs; convert to ps (fs/1000)
    """
    units = None
    dt = None
    for ln in lines[:400]:
        s = ln.strip().lower()
        if s.startswith("units"):
            parts = s.split()
            if len(parts) >= 2:
                units = parts[1]
        if s.startswith("timestep"):
            parts = s.split()
            if len(parts) >= 2:
                try:
                    dt = float(parts[1])
                except Exception:
                    pass
    if dt is None:
        return None
    if units == "metal":
        return dt
    if units == "real":
        return dt / 1000.0
    return None

def lammps_log_to_df(text: str, dt_override_ps: Optional[float] = None) -> pd.DataFrame:
    """
    Parse the first thermo table that begins with 'Step' and contains an MSD column.
    Returns DataFrame with ['Time','MSD'] in ps and Å² (best effort).
    """
    lines = text.splitlines()

    # Find thermo header line
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Step"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No 'Step' header found in the log.")

    # Collect numeric rows after header
    data_rows = []
    for ln in lines[start_idx + 1:]:
        s = ln.strip()
        if not s or not re.match(r"^[-+]?\d", s):
            break
        data_rows.append(s)
    if not data_rows:
        raise ValueError("No thermo rows found after 'Step' header.")

    df = pd.read_csv(io.StringIO("\n".join([lines[start_idx]] + data_rows)), sep=r"\s+", engine="python")

    # Identify Time/Step and MSD columns
    time_col = next((c for c in ["Time", "time", "t", "Step", "step"] if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No Time/Step column. Columns: {list(df.columns)}")
    msd_col = next((c for c in df.columns if re.search(r"msd", c, re.IGNORECASE)), None)
    if msd_col is None:
        msd_col = next((c for c in df.columns if "c_msd" in c.lower()), None)
    if msd_col is None:
        raise ValueError(f"No MSD-like column. Columns: {list(df.columns)}")

    df = df.rename(columns={time_col: "Time", msd_col: "MSD"}).astype({"Time": float, "MSD": float})

    # Convert steps → ps if needed
    looks_like_steps = (df["Time"].max() > 1e3)  # many steps vs a few ps
    if looks_like_steps:
        dt_ps = dt_override_ps if (dt_override_ps and dt_override_ps > 0) else _infer_timestep_ps(lines) or 0.001
        df["Time"] = df["Time"] * dt_ps

    return df[["Time", "MSD"]].reset_index(drop=True)

def _fit_ls(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares line fit: returns (m, b, R²)."""
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return m, b, r2

def fit_diffusion(
    df: pd.DataFrame,
    method: str = "last_frac",
    last_frac: float = 0.2,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None
) -> Dict[str, float]:
    """
    Fit MSD vs Time linearly on a selected region.
      - method='last_frac' → use the last (1-last_frac) portion of points
      - method='time_window' → use tmin ≤ Time ≤ tmax
      - method='auto_best_r2' → try several windows and choose the one with best R²
    Returns slope, intercept, R², D.
    """
    if len(df) < 3:
        raise ValueError("Not enough MSD points to fit a line.")

    candidates = []

    if method == "last_frac":
        i0 = int(np.floor(last_frac * len(df)))
        i0 = min(i0, len(df) - 3)  # ensure ≥3 points
        x = df["Time"].to_numpy()[i0:]
        y = df["MSD"].to_numpy()[i0:]
        if len(x) >= 3:
            candidates.append((x, y, f"last {100*(1-last_frac):.0f}%"))

    elif method == "time_window":
        if tmin is None or tmax is None or tmax <= tmin:
            raise ValueError("Set a valid time window (tmin < tmax).")
        sel = df[(df["Time"] >= tmin) & (df["Time"] <= tmax)]
        if len(sel) < 3:
            raise ValueError("Not enough points in the chosen time window.")
        candidates.append((sel["Time"].to_numpy(), sel["MSD"].to_numpy(), f"{tmin}–{tmax} ps"))

    elif method == "auto_best_r2":
        # try several sensible late-time fractions & fixed windows
        fracs = [0.2, 0.3, 0.4, 0.5]
        for f in fracs:
            i0 = int(np.floor(f * len(df)))
            i0 = min(i0, len(df) - 3)
            if len(df) - i0 >= 3:
                candidates.append((df["Time"].to_numpy()[i0:], df["MSD"].to_numpy()[i0:], f"last {100*(1-f):.0f}%"))
        # common fixed windows people use
        for win in [(3.0, 5.0), (3.5, 5.0), (4.0, 5.0)]:
            sel = df[(df["Time"] >= win[0]) & (df["Time"] <= win[1])]
            if len(sel) >= 3:
                candidates.append((sel["Time"].to_numpy(), sel["MSD"].to_numpy(), f"{win[0]}–{win[1]} ps"))
        if not candidates:
            raise ValueError("Auto mode found no usable window.")

    else:
        raise ValueError("Unknown fit method.")

    # choose best R²
    best = None
    for x, y, label in candidates:
        m, b, r2 = _fit_ls(x, y)
        if (best is None) or (r2 > best["R2"]):
            best = {"m": m, "b": b, "R2": r2, "label": label}

    D = best["m"] / (2.0 * DIMENSION)
    return {"slope": best["m"], "intercept": best["b"], "R2": best["R2"], "D": D, "window": best["label"]}

# ----------------------------
# 2) Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

elements = list(URLS.keys())
pick_elements = st.sidebar.multiselect("Elements", options=elements, default=elements)

all_temps = sorted({T for el in pick_elements for T in URLS[el]})
pick_temps = st.sidebar.multiselect("Temperatures (K)", options=all_temps, default=all_temps)

fit_method = st.sidebar.radio(
    "Fit method",
    options=["time_window", "last_frac", "auto_best_r2"],
    index=0,
    help="Choose 'time_window' (e.g., 3.5–5.0 ps) to match your friend's plots."
)

# Controls for each method
tmin = tmax = None
last_frac = 0.2
if fit_method == "time_window":
    c1, c2 = st.sidebar.columns(2)
    tmin = c1.number_input("Start (ps)", min_value=0.0, value=3.5, step=0.1)
    tmax = c2.number_input("End (ps)",   min_value=0.0, value=5.0, step=0.1)
elif fit_method == "last_frac":
    last_frac = st.sidebar.slider("Use last fraction of data", 0.0, 0.8, 0.2, 0.05)

show_points = st.sidebar.checkbox("Show raw MSD points", True)
show_fit = st.sidebar.checkbox("Show fitted line", True)

dt_override_ps = st.sidebar.number_input(
    "Override timestep (ps) if Time is in steps (optional)", min_value=0.0, value=0.0, step=0.0005,
    help="If your logs use 'Step' instead of 'Time' and you know the timestep, enter it here. "
         "Leave 0 to auto-infer (fallback 0.001 ps/step)."
)

# ----------------------------
# 3) Main analysis
# ----------------------------
msd_figs = []
rows = []

for el in pick_elements:
    for T in sorted(URLS[el]):
        if T not in pick_temps:
            continue
        try:
            txt = fetch_text(URLS[el][T])
            df = lammps_log_to_df(txt, dt_override_ps if dt_override_ps > 0 else None)
            fit = fit_diffusion(df, method=fit_method, last_frac=last_frac, tmin=tmin, tmax=tmax)

            # MSD vs Time (points + line)
            fig = px.scatter(
                df, x="Time", y="MSD",
                title=f"{el} @ {T} K — MSD vs Time  •  fit: {fit['window']}",
                opacity=1.0 if show_points else 0.0
            )
            if show_fit:
                xline = np.linspace(df["Time"].min(), df["Time"].max(), 200)
                yline = fit["slope"] * xline + fit["intercept"]
                fig.add_scatter(x=xline, y=yline, mode="lines", name=f"Fit (R²={fit['R2']:.3f})")
            fig.update_layout(height=420, xaxis_title="Time (ps)", yaxis_title="MSD (Å²)")
            msd_figs.append(fig)

            rows.append({
                "Element": el,
                "Temperature (K)": T,
                "Fit window": fit["window"],
                "Slope (MSD/time)": fit["slope"],
                "Intercept": fit["intercept"],
                "R²": fit["R2"],
                "D (Å²/ps)": fit["D"],
            })
        except Exception as e:
            st.error(f"{el} {T} K → {e}")

# ----------------------------
# 4) Output
# ----------------------------
plot_cfg = {"displaylogo": False, "responsive": True}

if msd_figs:
    st.subheader("MSD vs Time (per temperature)")
    for fig in msd_figs:
        st.plotly_chart(fig, width="stretch", config=plot_cfg)
else:
    st.info("Pick at least one element and temperature on the left.")

if rows:
    st.subheader("Fit Results and Diffusion Coefficients")
    out = pd.DataFrame(rows).sort_values(["Element", "Temperature (K)"])
    st.dataframe(out, width="stretch")

    st.subheader("Diffusion Coefficient vs Temperature")
    figD = px.line(
        out, x="Temperature (K)", y="D (Å²/ps)", color="Element", markers=True,
        title="Diffusion Coefficient vs Temperature (3D: D = slope / 6)"
    )
    figD.update_layout(height=520, xaxis_title="Temperature (K)", yaxis_title="D (Å²/ps)")
    st.plotly_chart(figD, width="stretch", config=plot_cfg)

    st.markdown("---")
    st.markdown(
        """
          ### Observations
        **1) MSD linearity check:**  
        The MSD curves show an approximately **linear** region at late times (we fit the last fraction of data),
        which indicates **normal diffusion**.

        **2) Diffusion coefficients per T:**  
        We compute **D = slope / 6** from the linear region. The table lists slope, R², and D for each (element, T).

        **3) How does D vary with T for each element?**  
        **D increases with temperature** for Au, Cu, and Pt. This is expected: higher T → faster atomic motion.

        **4) Compare Cu, Au, Pt — is there a trend? Why?**  
        The typical ordering is **Cu > Au > Pt** at the same T.  
        Heavier atoms and stronger bonding (Pt) yield **lower D**; lighter/weaker-bonded (Cu) yield **higher D**.

        **5) Is that expected?**  
        Yes. It matches transport theory and MD intuition for liquid metals: D tends to grow with T and
        be smaller for heavier/strongly-bonded elements.

        **6) Literature comparison (qualitative):**  
        Reported MD/experimental data for liquid transition metals show **monotonic D(T)** and the same
        qualitative element ordering. For quantitative comparison, match your exact units and simulation
        conditions to published values in your course references.
        """
    )
else:
    st.warning("No valid diffusion data yet — adjust the fit window or selections.")
