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

# --- UI Header ---
st.title("🧷 Horizontal Spring-Mass System (Interactive)")
st.markdown("5 masses connected by springs, vibrating horizontally.")

# --- Control Panel ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start"):
        st.session_state.running = True
with col2:
    if st.button("⏸ Stop"):
        st.session_state.running = False
with col3:
    if st.button("🔁 Reset"):
        st.session_state.frame = 0
        st.session_state.running = False

# --- Simulation Parameters ---
N = 5
ω0 = 1.0
T = 20
dt = 0.01
t_eval = np.arange(0, T, dt)

# --- Initial Velocity Selection ---
v0_user = st.slider("Initial velocity on mass 1", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# --- Playback speed ---
speed = st.selectbox("Playback speed", options=[0.25, 0.5, 1.0, 2.0, 4.0], index=2, format_func=lambda x: f"{x}×")

# --- Coupling Matrix A ---
A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)

# --- Initial Conditions ---
x0 = np.zeros(N)
v0 = np.zeros(N)
v0[0] = v0_user
y0 = np.concatenate([x0, v0])

# --- Differential Equation ---
def dxdt(t, y):
    x = y[:N]
    v = y[N:]
    dx = v
    dv = -ω0**2 * A @ x
    return np.concatenate([dx, dv])

# --- Solve ODE Once (no recalculation each frame) ---
sol = solve_ivp(dxdt, [0, T], y0, t_eval=t_eval)
x_all = sol.y[:N]

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(6, 2.5))
plot_placeholder = st.empty()
y_fixed = np.zeros(N)

# --- Draw Current Frame ---
def draw(frame):
    displacement = x_all[:, frame]
    x_positions = np.arange(N) + displacement
    ax.clear()
    ax.plot(x_positions, y_fixed, 'o-', lw=3, markersize=12, color="royalblue")
    ax.set_xlim(-1, N)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y = 0")
    ax.set_title(f"t = {t_eval[frame]:.2f} s")
    plot_placeholder.pyplot(fig)

# --- Main Playback Loop ---
if st.session_state.running:
    draw(st.session_state.frame)
    st.session_state.frame += 1
    if st.session_state.frame >= len(t_eval):
        st.session_state.running = False
    time.sleep(0.01 / speed)
else:
    draw(st.session_state.frame)
