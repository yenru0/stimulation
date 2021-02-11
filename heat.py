import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("Qt5Agg")

from function.etc import PerlinNoiseFactory

## matplotlib
fig, ax = plt.subplots()
line, = plt.plot([], [], 'ro')


## domain
# space
L = np.pi
N = 111
X, dX = np.linspace(0, L, N, retstep=True)
ax.set_xlim(0, L)
ax.set_ylim(-2, 2)


T = 40

Y = np.zeros(N)
Y_past = Y.copy()
Y_future = Y

CFL = 1
c = 1
dt = CFL * dX/c
total_t = np.arange(0, T, dt)


pf = PerlinNoiseFactory(1, 4)
Y = np.cos(X)**2 # np.array([pf(i*0.5) for i in X])
Y_future = Y.copy()



# loop
def animate(frame):
    global Y, Y_past, Y_future
    Y[0], Y[-1] = 0, 0
    t = frame + dt
    Y_past = Y.copy(); Y = Y_future.copy()

    for i in range(1, N-2):
        Y_future[i] = 2*Y[i] - Y_past[i] + CFL**2 * (Y[i+1] - 2*Y[i] + Y[i-1])
    line.set_data(X, Y_future)
    return line


anim = animation.FuncAnimation(fig, animate, frames=total_t, interval=20)
plt.show()

"""
fig = plt.figure()
fig.set_dpi(100)
ax1 = fig.add_subplot(1, 1, 1)

# Diffusion constant
# 확산 상수
k = 2

# Scaling factor (for visualisation purposes)
# 스케일링 인수 (시각화 목적)
scale = 5

# Length of the rod (0,L) on the x axis
# 봉의 길이
L = np.pi

# Initial contitions u(0,t) = u(L,t) = 0. Temperature at x=0 and x=L is fixed
# 초기 조건, x=0과 x=L 에서의 온도는 고정된다.
x0 = np.linspace(0, L + 1, 10000)
t0 = 0
temp0 = 5  # Temperature of the rod at rest (before heating) # 정지 상태의 봉의 온도 (가열 전)

# Increment
dt = 0.01 # delta t


# Heat function
def u(x, t):
    return temp0 + scale * np.exp(-k * t) * np.cos(x)


# Gradient of u
def grad_u(x, t):
    # du/dx              #du/dt
    return scale * np.array([np.exp(-k * t) * np.cos(x), -k * np.exp(-k * t) * np.sin(x)])


a = []
t = []

for i in range(500):
    value = u(x0, t0) + grad_u(x0, t0)[1] * dt
    t.append(t0)
    t0 = t0 + dt
    a.append(value)

k = 0


def animate(i):  # The plot shows the temperature evolving with time
    global k  # at each point x in the rod
    x = a[k]  # The ends of the rod are kept at temperature temp0
    k += 1  # The rod is heated in one spot, then it cools down
    ax1.clear()
    plt.plot(x0, x, color='red', label='Temperature at each x')
    plt.plot(0, 0, color='red', label='Elapsed time ' + str(round(t[k], 2)))
    plt.grid(True)
    plt.ylim([temp0 - 2, 2.5 * scale])
    plt.xlim([0, L])
    plt.title('Heat equation')
    plt.legend()


anim = animation.FuncAnimation(fig, animate, frames=360, interval=20)
plt.show()
"""