import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N = 5
ω0 = 1.0
T = 20
dt = 0.01
A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)

x0 = np.zeros(N)
v0 = np.zeros(N)
v0[0] = 1.0
y0 = np.concatenate([x0, v0])

def dxdt(t, y):
    x = y[:N]
    v = y[N:]
    dx = v
    dv = -ω0**2 * A @ x
    return np.concatenate([dx, dv])

t_eval = np.arange(0, T, dt)
sol = solve_ivp(dxdt, [0, T], y0, t_eval=t_eval)
x_all = sol.y[:N]

st.title("🌸 ばねでつながれた質点たち")
st.markdown("N=5 の質点系の時間発展を視覚化しています。")

frame = st.slider("時刻 t", 0, len(t_eval)-1, 0, step=1)

fig, ax = plt.subplots(figsize=(6, 3))
ax.set_xlim(-1, N)
ax.set_ylim(-2, 2)
ax.set_xlabel("位置")
ax.set_ylabel("変位")
ax.set_title(f"t = {t_eval[frame]:.2f} s")

positions = np.arange(N)
displacements = x_all[:, frame]
ax.plot(positions, displacements, "o-", lw=2)

st.pyplot(fig)
