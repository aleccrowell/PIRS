import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import math
import scipy

low_x = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
low_y = [2, 0, -2, 2, 0, -2, 2, 0, -2, 2, 0, -2]

lin_x = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
lin_y = [-32, -30, -28, -12, -10, -8, 12, 10, 8, 32, 30, 28]

circ_x = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
circ_y = [-7, -5, -3, 7, 5, 3, -7, -5, -3, 7, 5, 3]


def calc_intervals(x,y,alpha=0.5):
    dof = len(np.unique(x))
    regr = linear_model.LinearRegression()
    _ = regr.fit(np.array(x)[:, np.newaxis], np.array(y))
    rsq = np.sum((regr.predict(np.array(x)[:, np.newaxis]) - np.array(y)) ** 2)
    regr_error = math.sqrt(rsq/(dof-2))
    xsq = np.sum((np.array(x) - np.mean(np.array(x))) ** 2)
    upper = regr.predict(np.arange(min(x), (max(x)+.1), 0.1)[:, np.newaxis]) + scipy.stats.t.ppf(1.0-alpha/2., dof) * regr_error*np.sqrt(1+1/dof+(((np.arange(min(x), (max(x)+.1), 0.1)-np.mean(x))**2)/xsq))
    lower = regr.predict(np.arange(min(x), (max(x)+.1), 0.1)[:, np.newaxis]) - scipy.stats.t.ppf(1.0-alpha/2., dof) * regr_error*np.sqrt(1+1/dof+(((np.arange(min(x), (max(x)+.1), 0.1)-np.mean(x))**2)/xsq))
    return upper, lower


xs = np.arange(0, 3.1, 0.1)

fig = plt.figure()

ax1 = fig.add_subplot(1, 3, 1, aspect=.05)
ax1.scatter(low_x, low_y,color='w', s=10, marker = 'o', edgecolors='k')
u, l = calc_intervals(low_x, low_y)
ax1.plot(xs, u, color='k')
ax1.plot(xs, l, color='k')
ax1.arrow(xs[2], u[2], 0, -u[2], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[2], l[2], 0, -l[2], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[12], u[12], 0, -u[12], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[12], l[12], 0, -l[12], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[8], u[8], 0, -u[8], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[8], l[8], 0, -l[8], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[18], u[18], 0, -u[18], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[18], l[18], 0, -l[18], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[22], u[22], 0, -u[22], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[22], l[22], 0, -l[22], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[28], u[28], 0, -u[28], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.arrow(xs[28], l[28], 0, -l[28], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax1.plot([0, 3], [0, 0], color='k', linestyle=':')
ax1.set_xlim([-0.1, 3.1])
ax1.set_ylim([-35.0, 35.0])
ax1.axis('off')
ax1.set_title('Low PIRS')


ax2 = fig.add_subplot(1, 3, 2, aspect=.05)
ax2.scatter(lin_x, lin_y, color='w', s=10, marker = 'o', edgecolors='k')
u, l = calc_intervals(lin_x, lin_y)
ax2.plot(xs, u, color='k')
ax2.plot(xs, l, color='k')
ax2.arrow(xs[2], u[2], 0, -u[2], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[2], l[2], 0, -l[2], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[12], u[12], 0, -u[12], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[12], l[12], 0, -l[12], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[8], l[8], 0, -l[8], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[8], u[8], 0, -u[8], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[18], u[18], 0, -u[18], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[18], l[18], 0, -l[18], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[22], u[22], 0, -u[22], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[22], l[22], 0, -l[22], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[28], u[28], 0, -u[28], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.arrow(xs[28], l[28], 0, -l[28], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax2.plot([0, 3], [0, 0], color='k', linestyle=':')
ax2.set_xlim([-0.1, 3.1])
ax2.set_ylim([-35.0, 35.0])
ax2.axis('off')
ax2.set_title('High PIRS')


ax3 = fig.add_subplot(1, 3, 3, aspect=.05)
ax3.scatter(circ_x, circ_y, color='w', s=10, marker = 'o', edgecolors='k')
u, l = calc_intervals(circ_x, circ_y)
ax3.plot(xs, u, color='k')
ax3.plot(xs, l, color='k')
ax3.arrow(xs[2], u[2], 0, -u[2], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[2], l[2], 0, -l[2], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[12], u[12], 0, -u[12], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[12], l[12], 0, -l[12], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[8], u[8], 0, -u[8], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[8], l[8], 0, -l[8], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[18], u[18], 0, -u[18], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[18], l[18], 0, -l[18], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[22], u[22], 0, -u[22], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[22], l[22], 0, -l[22], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[28], u[28], 0, -u[28], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.arrow(xs[28], l[28], 0, -l[28], head_width=0.05, head_length=1.5, length_includes_head=True, color='k')
ax3.plot([0, 3], [0, 0], color='k', linestyle=':')
ax3.set_xlim([-0.1, 3.1])
ax3.set_ylim([-35.0, 35.0])
ax3.axis('off')
ax3.set_title('High PIRS')


plt.savefig('diagram.pdf')
