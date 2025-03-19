import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'
plt.locator_params(axis='x', nbins=5)
plt.rcParams['xtick.labelsize'] = 35
plt.rcParams['ytick.labelsize'] = 35
from matplotlib.animation import FuncAnimation

label_size = 30
legend_size = 22.5
fig_label_xpos, fig_label_ypos, = 0.025, 0.725

data_to_plot = np.load('ts_x_hg_corr.npy')
realized_noise = np.load('corr_noise_contribution.npy')
markov_slope = np.load('BLE/default_save_slopes.npy')
markov_noise = np.load('BLE/default_save_noise.npy')

#### Load non-Markov Estimates ####

nonmarkov_slope = np.load('NBLE/default_save_slopes.npy')
nonmarkov_noise = np.load('NBLE/default_save_noise.npy')

start_time = 0.
end_time = 2000.
samples = 30000

window_size = 2000
window_shift = 100
loop_range = np.arange(0, samples - window_size, window_shift)
dt = (end_time - start_time) / samples
time = np.arange(start_time, end_time, dt) 

r_array = -1 * np.linspace(5, 15, samples)
analytical_drift_slopes_to_plot = gaussian_filter(- r_array + realized_noise - 3 * data_to_plot**2, 200)

min_noise = 0.05
max_noise = 1.5
sigma_array = np.array([max_noise])
sigma_array = np.zeros(samples)
sigma_array[:int(samples/2.)] = np.linspace(min_noise,max_noise, int(samples/2.))
sigma_array[int(samples/2.):] = np.linspace(max_noise,min_noise, int(samples/2.))

psi_noise = sigma_array * data_to_plot * dt
psi_noise = gaussian_filter(psi_noise, 150)

fig = plt.figure(figsize=(4, 8.27))
outer = gridspec.GridSpec(3, 1, wspace=0., hspace=0.)
for i in range(3):
	if i == 0:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, data_to_plot, lw = 3)
		ax.text(fig_label_xpos, fig_label_ypos, 'd', fontsize = 40, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)
		ax.set_title(r'$x_{h,g}^{\rm corr}$',fontsize = label_size)
		ax.set_xlim(time[0], time[-1])
		ax.set_xticks([])
		# ax.set_yticks([])
		fig.add_subplot(ax)
	elif i == 1:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, np.zeros(time.size), ls = ':', lw = 3, color = 'r')
		ax.text(fig_label_xpos, fig_label_ypos, 'e', fontsize = 40, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)
		ax.plot(time[window_size + loop_range], markov_slope[0,:], lw = 3, color = 'b')
		ax.plot(time, analytical_drift_slopes_to_plot, 'g', lw = 3)
		ax.fill_between(time[window_size + loop_range], markov_slope[1,:], markov_slope[2,:], color = 'orange', alpha = 0.6, label = r'CI $\epsilon$')
		ax.fill_between(time[window_size::window_shift],markov_slope[3,:], markov_slope[4,:], color = 'orange', alpha = 0.4)
		ax.plot(time[window_size + loop_range], nonmarkov_slope[0,:], lw = 3, color = 'r')
		ax.fill_between(time[window_size + loop_range], nonmarkov_slope[1,:],nonmarkov_slope[2,:], color = 'green', alpha = 0.6, label = r'CI $\epsilon$')
		ax.fill_between(time[window_size::window_shift],nonmarkov_slope[3,:],nonmarkov_slope[4,:], color = 'green', alpha = 0.4)
		ax.set_xticks([])
		ax.set_yticks([-20,0])
		x_pos_powertext, y_pos_powertext = 0.01, 0.1
		power_text_factor = 0.85
		text_size = 25
		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.get_yaxis().get_offset_text().set_visible(False)
		ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10$', transform=ax.transAxes, fontsize = power_text_factor*text_size)
		# ax.set_ylabel(r'$\hat{\zeta}$',fontsize = label_size, labelpad = 1.)
		ax.set_xlim(time[0], time[-1])
		ax.set_ylim(-30,2)
		fig.add_subplot(ax)
	elif i == 2:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, sigma_array, ls = ':', color =  'g', lw = 3, label = r'$\sigma_{y}$')
		ax.plot(time, psi_noise, color = 'g', lw = 3)
		ax.plot(time[window_size::window_shift], markov_noise[1,:], color =  'b', lw = 3,)
		ax.fill_between(time[window_size::window_shift],markov_noise[1,:], markov_noise[1,:], color = 'orange', alpha = 0.6,)
		ax.fill_between(time[window_size::window_shift],markov_noise[1,:], markov_noise[1,:], color = 'orange', alpha = 0.4)
		ax.plot(time[window_size + loop_range], nonmarkov_noise[0,:], lw = 3, color = 'r')
		ax.fill_between(time[window_size + loop_range], nonmarkov_noise[1,:],nonmarkov_noise[2,:], color = 'green', alpha = 0.6)#, label = r'CI $\epsilon$')
		ax.fill_between(time[window_size::window_shift],nonmarkov_noise[3,:],nonmarkov_noise[4,:], color = 'green', alpha = 0.4)
		ax.text(fig_label_xpos, fig_label_ypos, 'f', fontsize = 40, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)
		ax.set_xlabel(r'Time $t$',fontsize = label_size)
		ax.set_xlim(time[0], time[-1])
		ax.legend(fontsize = 21,loc = 'center left', bbox_to_anchor = (0.075, 0.75
			), ncol = 1, columnspacing=0.3, handletextpad = 0.05, handlelength = 0.5, frameon = False)
		fig.add_subplot(ax)


plt.subplots_adjust(
	left = 0.15,
	bottom = 0.114,
	right  = 0.98,
	top = 0.902,
	wspace = 0.2,
	hspace = 0.2
	)
plt.savefig('Figure2_jkl.pdf')
plt.show()
