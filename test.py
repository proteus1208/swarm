import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def landingScore(disToEnd, state, max_dis=0.4):
    # --- proximity term (0 → 1) ---
    if disToEnd > max_dis:
        pro = 0.0
    else:
        x = 1 - disToEnd / max_dis
        pro = x**2

    vel = state[10:13]
    speed = np.linalg.norm(vel)
    vertical_speed = abs(vel[2])
    horizontal_speed = np.linalg.norm(vel[:2])

    # --- motion penalties (all in [0,1]) ---
    speed_factor = np.exp(- (speed / 2.0) ** 2)
    vertical_factor = np.exp(- (vertical_speed / 1.0) ** 2)
    horizontal_factor = np.exp(- (horizontal_speed / 1.0) ** 2)

    motion = speed_factor * vertical_factor * horizontal_factor

    # --- raw reward ---
    reward = pro * motion

    # --- define perfect condition explicitly ---
    perfect = (
        disToEnd < 0.01 and
        speed < 0.05 and
        horizontal_speed < 0.05 and
        vertical_speed < 0.05
    )

    if perfect:
        reward = 1.0
    else:
        # optional shaping boost (still ≤ 1)
        reward = min(1.0, reward)

    return reward, pro


# Fake state (constant velocity for visualization)
state = np.zeros(20)
state[10:13] = np.array([0.5, 0.2, -0.3])

distances = np.linspace(0, 1.0, 200)

pro_vals = []
reward_vals = []

for d in distances:
    r, p = landingScore(d, state)
    pro_vals.append(p)
    reward_vals.append(r)

# Plot
plt.figure()
plt.plot(distances, pro_vals, label="pro")
plt.plot(distances, reward_vals, label="landing reward")

plt.xlabel("Distance to Goal")
plt.ylabel("Value")
plt.title("Landing Behavior")
plt.legend()

plt.show()