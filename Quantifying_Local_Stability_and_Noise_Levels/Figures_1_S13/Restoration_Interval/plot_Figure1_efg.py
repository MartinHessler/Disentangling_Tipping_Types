import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import matplotlib.ticker as ticker
from datetime import datetime
import matplotlib.transforms

hatch_width =1.5
plt.rcParams['hatch.linewidth'] = hatch_width

label_size = 60
legend_size = 50
tick_size = 60
text_size = 60
line_width = 10
data_line_width = 10
power_text_factor = 1.
fig_label_add = 10

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
from matplotlib.animation import FuncAnimation

data_uncorrected = np.load('restoration_data_with_outliers.npy')

time_reconstructed = np.load('reconstructed_time_step.npy')

start_absolute_hours = 15
start_absolute_minutes = 10
start_absolute_seconds = 45

hours_in_seconds = start_absolute_hours * 3600
minutes_in_seconds = start_absolute_minutes * 60

start_in_seconds = hours_in_seconds + minutes_in_seconds + start_absolute_seconds
time = start_in_seconds + time_reconstructed

hour_stamps = np.array(time/3600., dtype = int)

minute_stamps = np.array((time/60.)%60, dtype = int)

second_stamps = time%60


year = np.ones(second_stamps.size) * 1996
month = np.ones(second_stamps.size) * 8
day = np.ones(second_stamps.size) * 10
time_stamp_frame = pd.DataFrame({'year':year, 'month':month, 'day':day,'hours':hour_stamps, 'minutes':minute_stamps,'seconds':second_stamps})

date_times = pd.to_datetime(time_stamp_frame[['year', 'month', 'day', 'hours', 'minutes', 'seconds']])

time = date_times
time_reconstructed = time 

data_to_plot = np.load('restoration_data.npy')

samples = data_to_plot.size

window_size = 1000
window_shift = 100
loop_range = np.arange(0, samples - window_size, window_shift)

nonmarkov_slope = np.load('NBLE/default_save_slopes.npy')
nonmarkov_noise = np.load('NBLE/default_save_noise.npy')

markov_slope = np.load('BLE/default_save_slopes.npy')
markov_noise = np.load('BLE/default_save_noise.npy')

nonmarkov_slope_uncorrected_data = np.load('NBLE/slopes_without_outlier_correction.npy')
nonmarkov_noise_uncorrected_data = np.load('NBLE/noise_without_outlier_correction.npy')

fig = plt.figure(figsize=(21,17))
outer = gridspec.GridSpec(3, 1, wspace=0., hspace=0.)
x_fig_label, y_fig_label = 1.015, 0.5
x_pos_powertext,y_pos_powertext = 0.025, 0.8

zorder_interval_markers = 0

for i in range(3):
	if i == 0:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])

		ax.plot(time, data_uncorrected, lw = data_line_width, color = 'orange')
		ax.plot(time, data_to_plot, lw = data_line_width)

		ax.text(x_fig_label, y_fig_label, 'e', transform=ax.transAxes, fontsize = text_size+fig_label_add, weight = 'bold')


		ax.set_title('NAWI Restoration, 10th/11th August 1996', fontsize = text_size)
		ax.set_xlim(time.iloc[0], time.iloc[-101])
		ax.set_ylim(-0.25,0.6)
		ax.set_xticks([])
		ax.axvspan(time.iloc[0], datetime.strptime('1996-08-10 15:48:54.95', "%Y-%m-%d %H:%M:%S.%f"), facecolor="none", edgecolor = 'black', hatch = '/', linewidth = hatch_width)
		ax.axvspan(datetime.strptime('1996-08-10 18:47:00.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 21:42:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkorange', zorder = zorder_interval_markers)
		ax.axvspan(datetime.strptime('1996-08-10 21:42:00.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-11 01:00:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkgreen', zorder = zorder_interval_markers)

		ax.set_xlim(time.iloc[0], time.iloc[-101]) # The index -101 is due to the window shift of 100 that excludes the last 99 data points from the analysis.

		ax.set_xticks([])
		date_form = DateFormatter("%H:%M")

		ax.set_yticks([0,0.2, 0.4])
		ax.set_ylabel(r'$\tilde{\omega}_{\rm R}(t)$', fontsize = label_size)
		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)

		ax.get_yaxis().get_offset_text().set_visible(False)
		
		ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-2}$', transform=ax.transAxes, fontsize = power_text_factor*text_size)

		plt.gca().xaxis.set_major_formatter(date_form)
		fig.add_subplot(ax)

	elif i == 1:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.text(x_fig_label, y_fig_label, 'f', transform=ax.transAxes, fontsize = text_size+fig_label_add, weight = 'bold')
		ax.plot(time.iloc[window_size - 1 + loop_range], nonmarkov_slope_uncorrected_data[0,:], color = 'grey', label = r'$\hat{\zeta}$ with outliers', lw = line_width)
		
		ax.plot([time.iloc[0],time.iloc[-1]],[0,0], ls = ':', lw = line_width, color = 'r')

		ax.plot(time.iloc[window_size - 1 + loop_range], nonmarkov_slope[0,:], color = 'r', label = r'$\hat{\zeta}_{\rm NBLE}$', lw = line_width)
		ax.fill_between(time.iloc[window_size - 1 + loop_range], nonmarkov_slope[1,:], nonmarkov_slope[2,:], color = 'green', alpha = 0.6)
		ax.fill_between(time.iloc[window_size - 1 + loop_range], nonmarkov_slope[3,:], nonmarkov_slope[4,:], color = 'green', alpha = 0.4)

		ax.set_xticks([])

		ax.axvspan(datetime.strptime('1996-08-10 18:47:00.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 21:42:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkorange', zorder = zorder_interval_markers)
		ax.axvspan(datetime.strptime('1996-08-10 21:42:00.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-11 01:00:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkgreen', zorder = zorder_interval_markers)
		ax.axvspan(time.iloc[0], datetime.strptime('1996-08-10 15:48:54.95', "%Y-%m-%d %H:%M:%S.%f"), facecolor="none", edgecolor = 'black', hatch = '/', linewidth = hatch_width)

		ax.set_xlim(time.iloc[0], time.iloc[-101])

		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.get_yaxis().get_offset_text().set_visible(False)
		ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-3}$', transform=ax.transAxes, fontsize = power_text_factor*text_size)

		date_form = DateFormatter("%H:%M")
		ax.xaxis.set_major_formatter(date_form)
		plt.gca().xaxis.set_major_formatter(date_form) 

		fig.add_subplot(ax)
	elif i == 2:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])

		ax.plot(time.iloc[window_size - 1 + loop_range], nonmarkov_noise_uncorrected_data[0,:], color = 'grey', label = r'$\hat{\sigma}$ with outliers', lw = line_width, zorder =1)
	
		ax.plot(time.iloc[window_size - 1 + loop_range], nonmarkov_noise[0,:], color =  'r', lw = line_width, label = r'$\hat{\psi}$')
		ax.fill_between(time.iloc[window_size - 1 + loop_range], nonmarkov_noise[1,:], nonmarkov_noise[2,:], color = 'green', alpha = 0.6)
		ax.fill_between(time.iloc[window_size - 1 + loop_range], nonmarkov_noise[3,:], nonmarkov_noise[4,:], color = 'green', alpha = 0.4)

		ax.text(x_fig_label, y_fig_label, 'g', transform=ax.transAxes, fontsize = text_size+fig_label_add, weight = 'bold')

		ax.set_xlabel(r'Time $t$ after outage', fontsize = label_size)

		ax.set_xlim(time.iloc[0], time.iloc[-101])
		ax.set_ylim(0, 0.004)
		ax.set_yticks([0, 0.001,0.002,0.003])

		ax.axvspan(datetime.strptime('1996-08-10 18:47:00.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 21:42:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkorange', zorder = zorder_interval_markers)
		ax.axvspan(datetime.strptime('1996-08-10 21:42:00.01', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-11 01:00:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkgreen', zorder = zorder_interval_markers)
		ax.axvspan(time.iloc[0], datetime.strptime('1996-08-10 15:48:54.95', "%Y-%m-%d %H:%M:%S.%f"), facecolor="none", edgecolor = 'black', hatch = '/', linewidth = hatch_width)

		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.get_yaxis().get_offset_text().set_visible(False)
		ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-3}$', transform=ax.transAxes, fontsize = power_text_factor*text_size)

		date_form = DateFormatter("%H:%M")
		ax.xaxis.set_major_formatter(date_form)

		ax.xaxis.set_major_locator(mdates.HourLocator(byhour=np.array([18,21,0,3,6,9]), interval=1, tz=None))
		ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha="right", rotation_mode = "anchor") 
		plt.gca().xaxis.set_major_formatter(date_form) 

		fig.add_subplot(ax)	


subaxis1 = fig.add_axes([0.675,0.835,0.17,0.1])
subaxis2 = fig.add_axes([0.2941,0.835,0.17,0.1])

sublabelsize = 40
subpowersize = 30
x_pos_subaxispowertext,y_pos_subaxispowertext = 0.213, 2.83
x_pos_subaxispowertext2,y_pos_subaxispowertext2 = 0.68, 2.83

condition_indices = (time > datetime.strptime('1996-08-11 05:36:40', "%Y-%m-%d %H:%M:%S")) & (time <= datetime.strptime('1996-08-11 05:37:30', "%Y-%m-%d %H:%M:%S"))

sub2time = time.iloc[np.arange(condition_indices.size)[condition_indices]]
subaxis2.plot(sub2time, data_uncorrected[condition_indices], linewidth = data_line_width, color = 'C1')
subaxis2.plot(sub2time, data_to_plot[condition_indices], linewidth = data_line_width, color = 'C0')
subaxis2.set_xlim(sub2time.iloc[0], sub2time.iloc[-1])

subaxis2.tick_params(labelsize = sublabelsize)

date_form = DateFormatter("%H:%M:%S")
subaxis2.xaxis.set_major_formatter(date_form)
subaxis2.xaxis.set_major_locator(ticker.LinearLocator(numticks = 2))

subaxis1.set_yticks([0,0.02])

plt.gca().xaxis.set_major_formatter(date_form)

condition_indices = (time > datetime.strptime('1996-08-11 06:25:10', "%Y-%m-%d %H:%M:%S")) & (time <= datetime.strptime('1996-08-11 06:27:18', "%Y-%m-%d %H:%M:%S"))

sub1time = time_reconstructed.iloc[np.arange(condition_indices.size)[condition_indices]]
subaxis1.plot(sub1time, data_uncorrected[condition_indices], linewidth = data_line_width, color = 'C1')
subaxis1.plot(sub1time, data_to_plot[condition_indices], linewidth = data_line_width, color = 'C0')
subaxis1.xaxis.set_major_locator(ticker.LinearLocator(numticks = 2))
subaxis2.set_ylim(-0.005,0.045)

subaxis1.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
subaxis1.get_yaxis().get_offset_text().set_visible(False)
subaxis2.text(x_pos_subaxispowertext,y_pos_subaxispowertext, r'$\times 10^{-2}$', transform=ax.transAxes, fontsize = subpowersize)
subaxis2.set_yticks([0,0.03])
subaxis1.xaxis.set_major_formatter(date_form)
subaxis1.xaxis.set_major_locator(ticker.LinearLocator(numticks = 2))


dx = 10/72.; dy = 0/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
i = 0
for label in subaxis2.xaxis.get_majorticklabels():
	if i == 0:
	    label.set_transform(label.get_transform() + offset)
	if i == 1:
	    label.set_transform(label.get_transform() - offset)
	i += 1

#### patch for right layer of the y power factor ####
subaxis2.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
subaxis2.get_yaxis().get_offset_text().set_visible(False)
subaxis1.text(x_pos_subaxispowertext2,y_pos_subaxispowertext2, r'$\times 10^{-2}$', transform=ax.transAxes, fontsize = subpowersize)



subaxis1.set_xlim(sub1time.iloc[0], sub1time.iloc[-1])

subaxis1.tick_params(labelsize = sublabelsize)

date_form = DateFormatter("%H:%M:%S")

plt.gca().xaxis.set_major_formatter(date_form)


fig.get_axes()[0].set_visible(False)

plt.subplots_adjust(
	left = 0.12,
	bottom = 0.165,
	right  = 0.935,
	top = 0.95,
	wspace = 0.2,
	hspace = 0.2
	)

plt.savefig('Figure1_efg.pdf')
plt.show()
