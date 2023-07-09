import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Lorenz - parametry
sigma, beta, rho = 10, 2.336, 28
u0, v0, w0 = 0, 1, 1.05

width, height, dpi = 1000, 750, 100


tmax, n = 100, 10000

def lorenz(t, X, sigma, beta, rho):
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp



# Scipy
soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho),
                 dense_output=True)

t = np.linspace(0, tmax, n)
x, y, z = soln.sol(t)


# matplotlib 3d
fig = plt.figure(facecolor='k', figsize=(width/dpi, height/dpi))
ax = fig.gca(projection='3d')
ax.set_facecolor('k')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)



# kolor
s = 10
cmap = plt.cm.winter
for i in range(0,n-s,s):
    ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/n), alpha=0.4)

#usunięcie kratek
ax.set_axis_off()


plt.show()