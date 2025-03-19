import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
from datetime import datetime

label_size = 60
legend_size = 50
tick_size = 60
text_size = 60
line_width = 10
power_text_factor = 1.
fig_label_add = 10

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
from matplotlib.animation import FuncAnimation

data_npz = np.load('BV_data_GKF5.npz')
data_to_plot = data_npz['xx']
time = data_npz['tt']

start_absolute_hours = 15
start_absolute_minutes = 29
start_absolute_seconds = 40

hours_in_seconds = start_absolute_hours * 3600
minutes_in_seconds = start_absolute_minutes * 60

start_in_seconds = hours_in_seconds + minutes_in_seconds + start_absolute_seconds

time = start_in_seconds + time

hour_stamps = np.array(time/3600., dtype = int)
print(hour_stamps)
minute_stamps = np.array((time/60.)%60, dtype = int)
print(minute_stamps)
second_stamps = time%60
print(second_stamps)

year = np.ones(second_stamps.size) * 1996
month = np.ones(second_stamps.size) * 8
day = np.ones(second_stamps.size) * 10
time_stamp_frame = pd.DataFrame({'year':year, 'month':month, 'day':day,'hours':hour_stamps, 'minutes':minute_stamps,'seconds':second_stamps})
print(time_stamp_frame)

date_times = pd.to_datetime(time_stamp_frame[['year', 'month', 'day', 'hours', 'minutes', 'seconds']])
print(date_times)

time = date_times

samples = data_to_plot.size

window_size = 1000
window_shift = 100
loop_range = np.arange(0, samples - window_size, window_shift)

green_line_index = 14863

nonmarkov_slope = np.load('NBLE/default_save_slopes.npy')
nonmarkov_noise = np.load('NBLE/default_save_noise.npy')

fig = plt.figure(figsize=(21,17))
x_fig_label, y_fig_label = 1.015, 0.5
x_pos_powertext,y_pos_powertext = 0.01, 0.8

method_fingerprint_start = '1996-08-10 15:40:09.00'
zorder_interval_markers = 0
zorder_fill_between = 1
interval_color = 'darkred'

outer = gridspec.GridSpec(3, 1, wspace=0., hspace=0.)
for i in range(3):
	if i == 0:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.plot(time, data_to_plot, lw  = line_width)
		ax.text(x_fig_label, y_fig_label, 'a', transform=ax.transAxes, fontsize = text_size+fig_label_add, weight = 'bold')

		ax.set_title('NAWI Pre-Outage Period, 10th August 1996', fontsize = text_size)
		ax.set_xlim(time[0], time[23099])
		ax.set_xticks([])
		ax.axvspan(datetime.strptime(method_fingerprint_start, "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:42:03.139', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)
		ax.axvspan(datetime.strptime('1996-08-10 15:47:36.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:47:52.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)
		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.get_yaxis().get_offset_text().set_visible(False)
		ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-2}$', transform=ax.transAxes, fontsize = power_text_factor*text_size)
		ax.set_ylabel(r'$\tilde{\omega}_{\rm P}(t)$', fontsize = label_size)

		fig.add_subplot(ax)
	elif i == 1:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])
		ax.text(x_fig_label, y_fig_label, 'b', transform=ax.transAxes, fontsize = text_size+fig_label_add, weight = 'bold')

		ax.plot(time.iloc[window_size - 1 + loop_range], nonmarkov_slope[0,:], lw = line_width, color = 'r', label = r'$\hat{\zeta}_{\rm NBLE}$')

		ax.plot([time.iloc[0],time.iloc[-1]],[0,0], ls = ':', lw  = line_width, color = 'r')
		ax.fill_between(time.iloc[window_size - 1 + loop_range], nonmarkov_slope[1,:], nonmarkov_slope[2,:], color = 'green', alpha = 0.6, zorder = zorder_fill_between)
		ax.fill_between(time.iloc[window_size - 1 + loop_range], nonmarkov_slope[3,:], nonmarkov_slope[4,:], color = 'green', alpha = 0.4, zorder = zorder_fill_between)
		ax.set_xticks([])
		ax.set_xlim([time.iloc[0],time.iloc[-1]])
		ax.set_ylim([-6,0.75])
		ax.set_yticks([-4,0])
		ax.axvspan(datetime.strptime(method_fingerprint_start, "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:42:03.139', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)
		ax.axvspan(datetime.strptime('1996-08-10 15:47:36.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:47:52.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)
		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.set_ylabel(r'$\hat{\zeta}_{\rm NBLE}$', fontsize = label_size)
		fig.add_subplot(ax)
	elif i == 2:
		inner = gridspec.GridSpecFromSubplotSpec(1, 1,
				subplot_spec=outer[i], wspace=0., hspace=0.)
		ax = plt.Subplot(fig, inner[0])

		ax.plot(time.iloc[window_size - 1 + loop_range], nonmarkov_noise[0,:], color =  'r', lw  = line_width, label = r'$\hat{\psi}$')
		ax.fill_between(time.iloc[window_size - 1 + loop_range], nonmarkov_noise[1,:], nonmarkov_noise[2,:], color = 'green', alpha = 0.6, zorder = zorder_fill_between)
		ax.fill_between(time.iloc[window_size - 1 + loop_range], nonmarkov_noise[3,:], nonmarkov_noise[4,:], color = 'green', alpha = 0.4, zorder = zorder_fill_between)
	
		ax.text(x_fig_label, y_fig_label, 'c', transform=ax.transAxes, fontsize = text_size+fig_label_add, weight = 'bold')
		ax.axvspan(datetime.strptime(method_fingerprint_start, "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:42:03.139', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)
		ax.axvspan(datetime.strptime('1996-08-10 15:47:36.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:47:52.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)
		ax.set_xlabel(r'Time $t$ before outage', fontsize = label_size)
		ax.set_yticks([0.0,0.01,0.02])
		ax.set_xlim([time.iloc[0],time.iloc[-1]])
		ax.set_ylim(0, 0.022)
		ax.set_yticks([0,0.01])
		ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
		ax.get_yaxis().get_offset_text().set_visible(False)
		ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-2}$', transform=ax.transAxes, fontsize = power_text_factor*text_size)
		date_form = DateFormatter("%H:%M")
		ax.xaxis.set_major_formatter(date_form)
		ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=np.array([33,36,39,42,45,48]), interval=1, tz=None))
		ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha="right", rotation_mode = "anchor")
		ax.set_ylabel(r'Noise $\hat{\psi}$', fontsize=label_size)
		fig.add_subplot(ax)

print('--------------------------------------------')
print('Green line time stamp: ', time[green_line_index])
print('--------------------------------------------')
plt.subplots_adjust(
	left = 0.12,
	bottom = 0.165,
	right  = 0.935,
	top = 0.95,
	wspace = 0.2,
	hspace = 0.2
	)
plt.savefig('Figure1_bcd.pdf')
plt.show()
