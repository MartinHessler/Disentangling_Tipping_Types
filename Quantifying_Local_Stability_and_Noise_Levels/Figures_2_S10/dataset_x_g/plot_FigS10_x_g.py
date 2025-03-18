import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter
from matplotlib.animation import FuncAnimation

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.locator_params(axis='x', nbins=5)
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

label_size = 18
label_pad = 22

data_to_plot = np.load('ts_x_g.npy')
markov_slope = np.load('default_save_slopes.npy')
markov_noise = np.load('default_save_noise.npy')


start_time = 0.
end_time = 2000.
samples = 40000

window_size = 2000
window_shift = 100
loop_range = np.arange(0, samples - window_size, window_shift)
dt = (end_time - start_time) / samples

r_array = -1 * np.ones(samples)
analytical_drift_slope_to_plot = gaussian_filter(- r_array - 3 * data_to_plot**2,100)

time = np.arange(start_time, end_time, dt)

min_noise = 0.05
max_noise = 0.35

sigma_array = np.zeros(samples)
sigma_array[:10000] = np.ones(10000) * min_noise
sigma_array[10000:35000] = np.linspace(min_noise, max_noise, 25000)
sigma_array[35000:] = np.ones(5000) * max_noise

x_pos_powertext, y_pos_powertext = 0.01, 0.1
text_size = label_size
power_text_factor = 0.85
for i in range(time.size):
	if data_to_plot[i] <= -0:
		critical_transition = i
		break
print(time[critical_transition])

fig_label_xpos, fig_label_ypos, = 0.025, 0.7

fig = plt.figure(figsize=(6, 8.27))
outer = gridspec.GridSpec(3, 1, wspace=0., hspace=0.)
for i in range(3):
	if i == 0:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, data_to_plot, lw = 3)
		ax.text(fig_label_xpos, fig_label_ypos, 'a', fontsize = 40, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)
		ax.axvline(time[critical_transition], ls = ':', lw = 3, c = 'r')
		ax.set_title(r'$X_{\rm g}$', fontsize = label_size)
		ax.set_xlim(time[0], time[-1])
		ax.set_ylim(-1.8, 2.2)
		ax.set_xticks([])
		ax.set_ylabel('Raw data', fontsize = label_size, labelpad=label_pad)
		fig.add_subplot(ax)
	elif i == 1:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, np.zeros(time.size), ls = ':', lw = 3, color = 'r')
		ax.text(fig_label_xpos, fig_label_ypos, 'b', fontsize = 40, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)
		ax.plot(time, analytical_drift_slope_to_plot, 'g', lw = 3, label = r'Ground truth $\zeta$')
		ax.plot(time[int(window_size/2.) + loop_range], markov_slope[0,:], lw = 3, color = 'b', label = r'Estimate $\hat{\zeta}$')
		ax.axvline(time[critical_transition], ls = ':', lw = 3, c = 'r')
		ax.axvspan(time[critical_transition], time[critical_transition + window_size], alpha=0.7, color='grey')
		ax.fill_between(time[int(window_size/2.) + loop_range], markov_slope[1,:], markov_slope[2,:], color = 'orange', alpha = 0.6)
		ax.fill_between(time[int(window_size/2.) + loop_range],markov_slope[3,:], markov_slope[4,:], color = 'orange', alpha = 0.4)
		ax.set_ylabel('Drift slope', fontsize = label_size)
		handles, labels = ax.get_legend_handles_labels()
		order = [0,1]
		ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize = 21, loc = 'center', 
			bbox_to_anchor = (0.35, 0.75), ncol = 1, columnspacing=0.2, handletextpad = 0.8, handlelength = 0.5, 
			frameon = False)
		ax.set_xticks([])
		ax.set_xlim(time[0], time[-1])
		ax.set_ylim(-3.2,3.2)
		ax.set_yticks([-2,0,2])
		fig.add_subplot(ax)
	elif i == 2:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, sigma_array, color =  'g', lw = 3, label = r'Ground truth $\sigma$')
		ax.axvline(time[critical_transition], ls = ':', lw = 3, c = 'r')
		ax.plot(time[int(window_size/2.) + loop_range], markov_noise[0,:], color =  'b', lw = 3, label = r'Estimate $\hat{\sigma}$')
		ax.fill_between(time[int(window_size/2.) + loop_range],markov_noise[1,:], markov_noise[2,:], color = 'orange', alpha = 0.6)
		ax.fill_between(time[int(window_size/2.) + loop_range],markov_noise[3,:], markov_noise[4,:], color = 'orange', alpha = 0.4)
		ax.text(fig_label_xpos, fig_label_ypos, 'c', fontsize = 40, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)
		ax.legend(fontsize = 23, loc = 'center', bbox_to_anchor = (0.35, 0.65), handlelength = 0.5, frameon = False)
		ax.set_xlabel(r'time $t$', fontsize = label_size)
		ax.set_xlim(time[0], time[-1])
		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.get_yaxis().get_offset_text().set_visible(False)
		ax.set_yticks([0.1,0.2,0.3])
		ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-1}$', transform=ax.transAxes, fontsize = power_text_factor*text_size)
		ax.set_ylabel('Noise level', fontsize = label_size, labelpad= label_pad)
		fig.add_subplot(ax)

plt.subplots_adjust(
	left = 0.15,
	bottom = 0.114,
	right  = 0.98,
	top = 0.902,
	wspace = 0.2,
	hspace = 0.2
	)
plt.savefig('FigureS10_abc.pdf')
plt.show()