import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation

dummy_white_line = mlines.Line2D([], [], color='white')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'
plt.locator_params(axis='x', nbins=5)
plt.rcParams['xtick.labelsize'] = 35
plt.rcParams['ytick.labelsize'] = 35

label_size = 30
legend_size = 18
fig_label_xpos, fig_label_ypos, = 0.025, 0.725

data_to_plot = np.load('ts_x_h_corr.npy')

markov_slope = np.load('BLE/default_save_slopes.npy')
markov_noise = np.load('BLE/default_save_noise.npy')
nonmarkov_slope= np.load('NBLE/default_save_slopes.npy')
nonmarkov_noise= np.load('NBLE/default_save_noise.npy')

print(markov_slope.shape)

start_time = 0.
end_time = 2000.
samples = 30000

window_size = 2000
window_shift = 100
loop_range = np.arange(0, samples - window_size, window_shift)
dt = (end_time - start_time) / samples
r_array = np.linspace(-15, 5, samples)

for i in range(r_array.size):
	if r_array[i] >= 0:
		critical_transition = i
		break

for i in range(data_to_plot.size):
	if data_to_plot[i] < 0.943:
		early_critical_transition = i
		break

c = 0.75
d = np.sqrt(c)
q = 0.5
psi_noise = np.ones(data_to_plot.size) * d * q * dt

analytical_drift_slopes_to_plot = gaussian_filter(1 - 3 * data_to_plot**2, 100)

time = np.arange(start_time, end_time, dt)

min_noise = 0.05
max_noise = 1.

fig = plt.figure(figsize=(4, 8.27))
outer = gridspec.GridSpec(3, 1, wspace=0., hspace=0.)
for i in range(3):
	if i == 0:

		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, data_to_plot, lw = 3)
		ax.text(fig_label_xpos, fig_label_ypos, 'g', fontsize = 40, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)
		ax.set_title(r'$x_{h}^{\rm corr}$', fontsize = label_size, y = 1.03)
		ax.set_xlim(time[0], time[-1])
		ax.set_xticks([])
		ax.set_yticks([0,2])
		ax.axvline(time[early_critical_transition], color  = 'red', ls = ':', lw = 3)
		left_frame_border = 0.05275
		right_frame_border = 0.6333
		top_frame_border = 0.25
		ax.spines["bottom"].set_bounds(0,0)
		frameax = ax.twinx()
		ax.set_zorder(5)
		ax.plot([105,1265], [-0.925,-0.925], lw = 0.75, color = 'k')
		frameax.set_zorder(10)
		frameax.spines["top"].set_bounds(left_frame_border, right_frame_border)
		frameax.spines["top"].set_position(("axes", top_frame_border))
		frameax.spines["top"].set(edgecolor = 'black', visible = True, zorder =10)
		frameax.spines["left"].set_position(("axes", left_frame_border))
		frameax.spines["left"].set_bounds(-0.3,top_frame_border)
		frameax.spines["right"].set_position(("axes", right_frame_border))
		frameax.spines["right"].set_bounds(-0.3,top_frame_border)
		frameax.spines["bottom"].set_bounds(right_frame_border,1)
		frameax.set_xticks([])
		frameax.set_yticks([])
		fig.add_axes(ax)
		fig.add_axes(frameax)
	elif i == 1:
		inner = gridspec.GridSpecFromSubplotSpec(2, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.1)

		ax1 = plt.Subplot(fig, inner[0])
		ax1.set_yticks([-1,0])
		ax2 = plt.Subplot(fig, inner[1])
		ax1.plot(time, np.zeros(time.size), ls = ':', lw = 3, color = 'r')
		ax1.text(fig_label_xpos, fig_label_ypos - 0.3, 'h', fontsize = 40, verticalalignment='bottom', horizontalalignment='left', transform=ax1.transAxes)
		ax1.axvline(time[early_critical_transition], color  = 'red', ls = ':', lw = 3)
		ax1.plot(time[int(window_size/2.) + loop_range], markov_slope[0,:], lw = 3, color = 'b')
		ax1.plot(time, analytical_drift_slopes_to_plot, 'g', lw = 3)
		ax1.plot(time[int(window_size/2.) + loop_range], nonmarkov_slope[0,:], color = 'r', lw = 3, label = r'Estimate $\hat{\zeta}_{\rm NBLE}$')
		ax1.fill_between(time[int(window_size/2.) + loop_range], markov_slope[1,:], markov_slope[2,:], color = 'orange', alpha = 0.6)
		ax1.fill_between(time[int(window_size/2.) + loop_range],markov_slope[3,:], markov_slope[4,:], color = 'orange', alpha = 0.4)
		ax1.fill_between(time[int(window_size/2.) + loop_range], nonmarkov_slope[1,:],nonmarkov_slope[2,:], color = 'green', alpha = 0.6)
		ax1.fill_between(time[int(window_size/2.) + loop_range],nonmarkov_slope[3,:],nonmarkov_slope[4,:], color = 'green', alpha = 0.4)
		ax1.set_xticks([])
		ax1.set_xlim(time[0], time[-1])
		ax1.set_ylim(-2.1,0.3)
		ax2.axvline(time[early_critical_transition], color  = 'red', ls = ':', lw = 3)
		ax2.plot(time, analytical_drift_slopes_to_plot, 'g', lw = 3)
		ax2.plot(time[int(window_size/2.) + loop_range], nonmarkov_slope[0,:], color = 'r', lw = 3)
		ax2.fill_between(time[int(window_size/2.) + loop_range], nonmarkov_slope[1,:],nonmarkov_slope[2,:], color = 'green', alpha = 0.6)
		ax2.fill_between(time[int(window_size/2.) + loop_range],nonmarkov_slope[3,:],nonmarkov_slope[4,:], color = 'green', alpha = 0.4)
		ax2.set_xticks([])
		ax2.set_xlim(time[0], time[-1])
		ax2.set_ylim(-21,-2.2)
		ax2.set_yticks([-20,-10])
		x_pos_powertext, y_pos_powertext = 0.01, -0.8
		power_text_factor = 0.85
		text_size = 25
		ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax2.get_yaxis().get_offset_text().set_visible(False)
		ax2.text(x_pos_powertext,y_pos_powertext, r'$\times 10$', transform=ax.transAxes, fontsize = power_text_factor*text_size, bbox = dict(facecolor='white', alpha=0.5, linewidth=0, boxstyle="round", pad =0.075))
		ax1.spines['bottom'].set_visible(False)
		ax2.spines['top'].set_visible(False)
		ax1.xaxis.tick_top()
		ax1.tick_params(labeltop=False)
		ax2.xaxis.tick_bottom()
		d = .5 
		kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
		              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
		ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
		ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


		ax1handles, ax1labels = ax1.get_legend_handles_labels()

		print(ax1labels)
		order = [0]
		ax1.legend([ax1handles[idx] for idx in order],[ax1labels[idx] for idx in order], columnspacing=0.2, loc = 'center', bbox_to_anchor=(0.35,1.2),
		          fancybox=False, shadow=False, ncol=1, fontsize = legend_size, handletextpad = 0.5, handlelength = 0.5, frameon = False,
		          labelspacing = 0.05)
		ax1.spines["top"].set_visible(False)
		bottom_line_ycoord = 0.29
		ax1.plot([0,105], [bottom_line_ycoord,bottom_line_ycoord], lw = 0.75, color = 'k')
		ax1.plot([1265,2000], [bottom_line_ycoord,bottom_line_ycoord], lw = 0.75, color = 'k')
		ax1.set_zorder(20)
		fig.add_axes(ax1)
		fig.add_axes(ax2)
	elif i == 2:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, psi_noise, color =  'g', lw = 3, label = r'Ground truth $\Psi$')
		ax.axvline(time[early_critical_transition], color  = 'red', ls = ':', lw = 3)
		ax.plot(time[int(window_size/2.) + loop_range], markov_noise[0,:], color =  'blue', lw = 3)
		ax.fill_between(time[int(window_size/2.) + loop_range],markov_noise[1,:],markov_noise[2,:], color = 'orange', alpha = 0.6)
		ax.fill_between(time[int(window_size/2.) + loop_range],markov_noise[3,:],markov_noise[4,:], color = 'orange', alpha = 0.4)
		ax.plot(time[int(window_size/2.) + loop_range], nonmarkov_noise[0,:], color =  'r', label = r'Estimate $\hat{\Psi}$', lw = 3)
		ax.fill_between(time[int(window_size/2.) + loop_range],nonmarkov_noise[1,:],nonmarkov_noise[2,:], color = 'green', alpha = 0.6)
		ax.fill_between(time[int(window_size/2.) + loop_range],nonmarkov_noise[3,:],nonmarkov_noise[4,:], color = 'green', alpha = 0.4)
		ax.text(fig_label_xpos, fig_label_ypos, 'i', fontsize = 40, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)
		ax.set_xlabel(r'Time $t$', fontsize = label_size)
		ax.set_xlim(time[0], time[-1])
		ax.set_yticks([0.03,0.04])
		x_pos_powertext, y_pos_powertext = 0.01, 0.2
		power_text_factor = 0.85
		text_size = 25
		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.get_yaxis().get_offset_text().set_visible(False)
		ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-2}$', transform=ax.transAxes, fontsize = power_text_factor*text_size)
		handles, labels = ax.get_legend_handles_labels()
		handles.append(dummy_white_line)

		labels.append('')
	

		print(labels)
		order = [0,1]
		ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'center', bbox_to_anchor=(0.37,0.75),
		          fancybox=False, shadow=False, ncol=1, fontsize = legend_size, handletextpad = 0.8, handlelength = 0.3, frameon = False,
		          labelspacing = 0.05)
		fig.add_axes(ax)

plt.subplots_adjust(
	left = 0.15,
	bottom = 0.114,
	right  = 0.98,
	top = 0.902,
	wspace = 0.2,
	hspace = 0.2
	)
plt.savefig('FigureS10_ghi.pdf')
plt.show()
