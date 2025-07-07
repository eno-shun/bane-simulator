import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

# --- App Configuration ---
st.set_page_config(page_title="Mass-Spring System", layout="centered")

# --- Initialize Session State ---
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "running" not in st.session_state:
    st.session_state.running = False
if "solution" not in st.session_state:
    st.session_state.solution = None

# --- Simulation Parameters ---
N = 5
ω0 = 1.0
T = 20
dt = 0.01
t_eval = np.arange(0, T, dt)

# --- UI: Controls ---
st.title("🧷 Horizontal Spring-Mass System (Interactive)")
st.markdown("Masses move left and right, connected by springs.")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start"):
        st.session_state.running = True
with col2:
    if st.button("⏸ Stop"):
        st.session_state.running = False
with col3:
    if st.button("🔁 Reset"):
        st.session_state.running = False
        st.session_state.frame = 0

# --- UI: Parameters ---
v0_user = st.slider("Initial velocity of mass 1", 0.0, 5.0, 1.0, 0.1)
speed = st.selectbox("Playback speed", [0.25, 0.5, 1.0, 2.0, 4.0], index=2, format_func=lambda x: f"{x}×")

# --- Recompute solution if v0 changed or not stored ---
if st.session_state.solution is None or st.session_state.solution["v0"] != v0_user:
    # Define A matrix
    A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    # Initial condition
    x0 = np.zeros(N)
    v0 = np.zeros(N)
    v0[0] = v0_user
    y0 = np.concatenate([x0, v0])
    # ODE
    def dxdt(t, y):
        x = y[:N]
        v = y[N:]
        dx = v
        dv = -ω0**2 * A @ x
        return np.concatenate([dx, dv])
    sol = solve_ivp(dxdt, [0, T], y0, t_eval=t_eval)
    st.session_state.solution = {"x_all": sol.y[:N], "v0": v0_user}

# --- Draw current frame ---
x_all = st.session_state.solution["x_all"]
frame = st.session_state.frame
displacement = x_all[:, frame]
x_positions = np.arange(N) + displacement
y_fixed = np.zeros(N)

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(x_positions, y_fixed, 'o-', lw=3, markersize=12, color="royalblue")
ax.set_xlim(-1, N)
ax.set_ylim(-1, 1)
ax.set_xlabel("X Position")
ax.set_ylabel("Y = 0")
ax.set_title(f"t = {t_eval[frame]:.2f} s")
st.pyplot(fig)

# --- Playback control ---
if st.session_state.running:
    st.session_state.frame += 1
    if st.session_state.frame >= len(t_eval):
        st.session_state.running = False
    else:
        time.sleep(0.01 / speed)
        st.experimental_rerun()

