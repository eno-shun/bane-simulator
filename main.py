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
v0[0] = 1.0
y0 = np.concatenate([x0, v0])

# --- Differential Equation ---
def dxdt(t, y):
    x = y[:N]
    v = y[N:]
    dx = v
    dv = -ω0**2 * A @ x
    return np.concatenate([dx, dv])

# --- Solve ODE ---
sol = solve_ivp(dxdt, [0, T], y0, t_eval=t_eval)
x_all = sol.y[:N]  # shape (N, time steps)

# --- Streamlit UI ---
st.set_page_config(page_title="Spring-Mass Chain", layout="centered")
st.title("🧷 Horizontal Mass-Spring System")
st.markdown("5 masses connected by springs, vibrating horizontally.")

# --- Setup Plot Placeholder ---
fig, ax = plt.subplots(figsize=(6, 2.5))
plot_placeholder = st.empty()

# --- Fixed vertical position for all balls ---
y_fixed = np.zeros(N)

# --- Animation Loop ---
for frame in range(len(t_eval)):
    displacement = x_all[:, frame]
    x_positions = np.arange(N) + displacement  # ボールの位置 = 初期位置 + 変位
    ax.clear()
    ax.plot(x_positions, y_fixed, 'o-', lw=3, markersize=12, color="royalblue")
    ax.set_xlim(-1, N)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Fixed Y = 0")
    ax.set_title(f"t = {t_eval[frame]:.2f} s")
    plot_placeholder.pyplot(fig)
    time.sleep(0.01)


