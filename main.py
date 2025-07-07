import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

# --- Simulation Parameters ---
N = 5
ω0 = 1.0
T = 20
dt = 0.01
t_eval = np.arange(0, T, dt)

# --- Coupling Matrix A ---
A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)

# --- Initial Conditions ---
x0 = np.zeros(N)
v0 = np.zeros(N)
v0[0] = 1.0  # only first mass has velocity
y0 = np.concatenate([x0, v0])

# --- Differential Equation ---
def dxdt(t, y):
    x = y[:N]
    v = y[N:]
    dx = v
    dv = -ω0**2 * A @ x
    return np.concatenate([dx, dv])

# --- Solve the ODE ---
sol = solve_ivp(dxdt, [0, T], y0, t_eval=t_eval)
x_all = sol.y[:N]  # shape: (N, time steps)

# --- Streamlit UI ---
st.set_page_config(page_title="Mass-Spring System", layout="centered")
st.title("🧷 Masses Connected by Springs")
st.markdown("A chain of 5 masses connected by springs. Initial velocity on the first mass only.")

# --- Animation loop ---
fig, ax = plt.subplots(figsize=(6, 2.5))
line, = ax.plot([], [], 'o-', lw=3, markersize=12, color="royalblue")
ax.set_xlim(-1, N)
ax.set_ylim(-2, 2)
ax.set_xlabel("Mass Index")
ax.set_ylabel("Displacement")
ax.set_title("Spring-Mass System Over Time")

# --- Placeholder for animation ---
plot_placeholder = st.empty()

# --- Run animation ---
for frame in range(len(t_eval)):
    x_pos = np.arange(N)  # fixed horizontal positions
    y_disp = x_all[:, frame]  # vertical displacement
    ax.clear()
    ax.plot(x_pos, y_disp, 'o-', lw=3, markersize=12, color="royalblue")
    ax.set_xlim(-1, N)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Mass Index")
    ax.set_ylabel("Displacement")
    ax.set_title(f"t = {t_eval[frame]:.2f} s")
    plot_placeholder.pyplot(fig)
    time.sleep(0.01)  # controls playback speed

