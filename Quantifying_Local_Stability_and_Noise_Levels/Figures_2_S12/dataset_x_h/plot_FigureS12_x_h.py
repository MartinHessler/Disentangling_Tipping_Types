import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patches import Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.locator_params(axis='x', nbins=5)
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
from matplotlib.animation import FuncAnimation

data_to_plot = np.load('ts_x_h.npy')
markov_slope = np.load('default_save_slopes.npy')
markov_noise = np.load('default_save_noise.npy')

start_time = 0.
end_time = 2000.
samples = 30000

window_size = 2000
window_shift = 100
loop_range = np.arange(0, samples - window_size, window_shift)
dt = (end_time - start_time) / samples
r_array = -1 * np.linspace(5, 15, samples)

analytical_drift_slopes_to_plot = gaussian_filter(1 - 3 * data_to_plot**2, 100)

time = np.arange(start_time, end_time, dt)

min_noise = 0.05
max_noise = 1.
sigma_array = np.array([max_noise])
sigma_array = np.zeros(samples)
sigma_array[:int(samples/2.)] = np.linspace(min_noise,max_noise, int(samples/2.))
sigma_array[int(samples/2.):] = np.linspace(max_noise,min_noise, int(samples/2.))

fig = plt.figure(figsize=(4, 8.27))
outer = gridspec.GridSpec(3, 1, wspace=0., hspace=0.)
for i in range(3):
	if i == 0:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, data_to_plot, lw = 3)
		ax.text(60,2.6, 'd', fontsize = 40, verticalalignment='bottom', horizontalalignment='left')
		ax.set_title(r'$x_{h,g}$', fontsize = 25)
		ax.set_xlim(time[0], time[-1])
		ax.set_xticks([])
		fig.add_subplot(ax)
	elif i == 1:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, np.zeros(time.size), ls = ':', lw = 3, color = 'r')
		ax.text(60, -5, 'e', fontsize = 40, verticalalignment='bottom', horizontalalignment='left')
		ax.plot(time, analytical_drift_slopes_to_plot, 'g', lw = 3)
		ax.fill_between(time[int(window_size/2.) + loop_range], markov_slope[1,:], markov_slope[2,:], color = 'orange', alpha = 0.6, label = r'CB $\epsilon$')
		ax.fill_between(time[int(window_size/2.) + loop_range],markov_slope[3,:], markov_slope[4,:], color = 'orange', alpha = 0.4)
		ax.plot(time[int(window_size/2.) + loop_range], markov_slope[0,:], lw = 3, color = 'b', label = r'$\hat{\zeta}$')
		ax.set_xticks([])
		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.get_yaxis().get_offset_text().set_visible(False)
		ax.set_yticks([-10,0])
		x_pos_powertext, y_pos_powertext = 0.01, 0.1
		power_text_factor = 0.85
		text_size = 25
		ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10$', transform=ax.transAxes, fontsize = power_text_factor*text_size)
		ax.set_xlim(time[0], time[-1])
		ax.set_ylim(-20,2)
		fig.add_subplot(ax)
	elif i == 2:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, sigma_array, color =  'g', lw = 3, label = r'$\sigma$')
		
		ax.plot(time[int(window_size/2.) + loop_range], markov_noise[0,:], color =  'b', lw = 3, label = r'$\hat{\sigma}$')
		ax.fill_between(time[int(window_size/2.) + loop_range],markov_noise[1,:], markov_noise[2,:], color = 'orange', alpha = 0.6,label = r'CB $\epsilon$')
		ax.fill_between(time[int(window_size/2.) + loop_range],markov_noise[3,:], markov_noise[4,:], color = 'orange', alpha = 0.4)
		ax.text(60,0.65, 'f', fontsize = 40, verticalalignment='bottom', horizontalalignment='left')
		ax.set_xlabel(r'Time $t$', fontsize = 25)
		ax.set_xlim(time[0], time[-1])
		ax.set_yticks([0.3,0.7])
		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.get_yaxis().get_offset_text().set_visible(False)
		x_pos_powertext, y_pos_powertext = 0.01, 0.1
		power_text_factor = 0.85
		text_size = 25
		power_text = ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-1}$', transform=ax.transAxes, fontsize = power_text_factor*text_size, bbox = dict(facecolor='white', alpha=0.75, linewidth=0, boxstyle="round", pad =0.075))
		fig.add_subplot(ax)


plt.subplots_adjust(
	left = 0.15,
	bottom = 0.114,
	right  = 0.98,
	top = 0.902,
	wspace = 0.2,
	hspace = 0.2
	)
plt.savefig('FigureS12_def.pdf')
plt.show()
