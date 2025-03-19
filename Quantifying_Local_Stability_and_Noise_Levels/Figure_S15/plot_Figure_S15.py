import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.cbook import get_sample_data
from datetime import datetime

hatch_width =1.5
plt.rcParams['hatch.linewidth'] = hatch_width
label_size = 30
legend_size = 30
tick_size = 25
text_size = 50
line_width = 3
pdf_line_width = 2.
power_text_factor = 0.7
fig_label_add = 10

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size


x_fig_label, y_fig_label = 0.03, 0.82
x_pos_powertext,y_pos_powertext = 0.72, 0.86
drift1_y_pos_powertext = 0.86


drift_slope_axs = [None] * 5
noise_level_axs = [None] * 5


fig = plt.figure(constrained_layout=False, figsize = (21.,17.))
gs1 = fig.add_gridspec(nrows=2, ncols=3, left=0.07, right=0.99,
                        wspace=0.2, hspace = 0., bottom=0.57, top = 0.95)
drift_slope_axs[0] = fig.add_subplot(gs1[0,0])
drift_slope_axs[1] = fig.add_subplot(gs1[0,1])
noise_level_axs[0] = fig.add_subplot(gs1[1,0])
noise_level_axs[1] = fig.add_subplot(gs1[1,1])

gs2 = fig.add_gridspec(nrows=2, ncols=3, left=0.07, right=0.99,
                        wspace=0.2, hspace = 0.,bottom=0.05, top = 0.43)
drift_slope_axs[2] = fig.add_subplot(gs2[0,0])
drift_slope_axs[3] = fig.add_subplot(gs2[0,1])
drift_slope_axs[4] = fig.add_subplot(gs2[0,2])
noise_level_axs[2] = fig.add_subplot(gs2[1,0])
noise_level_axs[3] = fig.add_subplot(gs2[1,1])
noise_level_axs[4] = fig.add_subplot(gs2[1,2])

drift_slope_axs[0].xaxis.set_visible(False)
drift_slope_axs[1].xaxis.set_visible(False)
drift_slope_axs[2].xaxis.set_visible(False)
drift_slope_axs[3].xaxis.set_visible(False)
drift_slope_axs[4].xaxis.set_visible(False)

drift_slope_axs[0].set_title('Pre-Outage Interval\n (Baseline: 1000 samples)', fontsize = label_size)
drift_slope_axs[1].set_title('Restoration Interval\n (Baseline: 1000 samples)', fontsize = label_size)
drift_slope_axs[2].set_title('(i) Pre-Outage Conditions\n (Baseline: 8000 samples)', fontsize = label_size)
drift_slope_axs[3].set_title('(ii) Keeler-Allston THIF\n (Baseline: 8000 samples)', fontsize = label_size)
drift_slope_axs[4].set_title('(iii) McNary Loss\n (Baseline: 8000 samples)', fontsize = label_size)

num_datasets = 4
window_sizes = [500, 750, 1250, 1500]
color_code = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
window_size_label = ['-50%', '-25%', '+25%', '+50%']
window_shift = 100
pre_outage_drift = [None] * num_datasets 
pre_outage_noise = [None] * num_datasets 


######################### BEGIN PRE-OUTAGE PLOT #############################
##############################################################################

BV_npz = np.load('Pre_outage/window_size500/BV_data_GKF5.npz')
BVtime = BV_npz['tt']
samples = BVtime.size

#### IMPLEMENT REAL PRE-OUTAGE TIME AXIS ####
#############################################

start_absolute_hours = 15
start_absolute_minutes = 29
start_absolute_seconds = 40

hours_in_seconds = start_absolute_hours * 3600
minutes_in_seconds = start_absolute_minutes * 60

start_in_seconds = hours_in_seconds + minutes_in_seconds + start_absolute_seconds

BVtime = start_in_seconds + BVtime

hour_stamps = np.array(BVtime/3600., dtype = int)
print(hour_stamps)
minute_stamps = np.array((BVtime/60.)%60, dtype = int)
print(minute_stamps)
second_stamps = BVtime%60
print(second_stamps)

year = np.ones(second_stamps.size) * 1996
month = np.ones(second_stamps.size) * 8
day = np.ones(second_stamps.size) * 10
time_stamp_frame = pd.DataFrame({'year':year, 'month':month, 'day':day,'hours':hour_stamps, 'minutes':minute_stamps,'seconds':second_stamps})
print(time_stamp_frame)

date_times = pd.to_datetime(time_stamp_frame[['year', 'month', 'day', 'hours', 'minutes', 'seconds']])
print(date_times)

BVtime = date_times

###########################
###########################


file_labels = ['window_size500', 'window_size750', 'window_size1250', 'window_size1500']
for i in range(num_datasets):
    loop_range = np.arange(0, samples - window_sizes[i], window_shift)
    pre_outage_drift[i] = np.load('Pre_Outage/' + file_labels[i] + '/NM_slopes_BV.npy')
    drift_slope_axs[0].plot(BVtime.iloc[window_sizes[i]-1+loop_range], pre_outage_drift[i][0,:], lw = line_width, color = color_code[i])
for i in range(num_datasets):
    loop_range = np.arange(0, samples - window_sizes[i], window_shift)
    pre_outage_noise[i] = np.load('Pre_Outage/' + file_labels[i] + '/NM_noise_BV.npy')
    noise_level_axs[0].plot(BVtime.iloc[window_sizes[i]-1+loop_range],pre_outage_noise[i][0,:], lw = line_width, color = color_code[i])

method_fingerprint_start = '1996-08-10 15:40:09.00'
interval_color = 'darkred'
zorder_interval_markers = 0

drift_slope_axs[0].plot([BVtime.iloc[0], BVtime.iloc[-1]],[0,0], ls = ':', color = 'red', lw = line_width)

drift_slope_axs[0].axvspan(datetime.strptime(method_fingerprint_start, "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:42:03.139', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)
drift_slope_axs[0].axvspan(datetime.strptime('1996-08-10 15:47:36.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:47:52.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)
noise_level_axs[0].axvspan(datetime.strptime(method_fingerprint_start, "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:42:03.139', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)
noise_level_axs[0].axvspan(datetime.strptime('1996-08-10 15:47:36.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 15:47:52.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color=interval_color, zorder = zorder_interval_markers)

date_form = DateFormatter("%H:%M")
drift_slope_axs[0].xaxis.set_major_formatter(date_form)
drift_slope_axs[0].xaxis.set_major_formatter(date_form)
noise_level_axs[0].xaxis.set_major_formatter(date_form)
noise_level_axs[0].xaxis.set_major_formatter(date_form)

noise_level_axs[0].xaxis.set_major_locator(mdates.MinuteLocator(byminute=np.array([33,36,39,42,45,48]), interval=1, tz=None))
noise_level_axs[0].set_xticklabels(noise_level_axs[0].get_xticklabels(), rotation = 30, ha="right", rotation_mode = "anchor")

drift_slope_axs[0].set_xlim(BVtime.iloc[0],BVtime.iloc[-1])
noise_level_axs[0].set_xlim(BVtime.iloc[0],BVtime.iloc[-1])

drift_slope_axs[0].set_ylabel(r'Drift slope $\hat{\zeta}_{\rm NBLE}$', fontsize = label_size)
noise_level_axs[0].set_ylabel(r'Noise level $\hat{\Psi}$', fontsize = label_size)
noise_level_axs[0].set_xlabel(r'Time $t$ before outage', fontsize = label_size)

noise_level_axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
noise_level_axs[0].get_yaxis().get_offset_text().set_visible(False)
noise_level_axs[0].text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-2}$', transform=noise_level_axs[0].transAxes, fontsize = power_text_factor*text_size)

######################### END PRE-OUTAGE PLOT #############################
###########################################################################


######################### BEGIN POST-OUTAGE PLOT #############################
##############################################################################


#### IMPLEMENT REAL POST-OUTAGE TIME AXIS ####
##############################################

time_reconstructed = np.load('Post_Outage/window_size500/reconstructed_time_step.npy')
samples = time_reconstructed.size

start_absolute_hours = 15
start_absolute_minutes = 10
start_absolute_seconds = 45

hours_in_seconds = start_absolute_hours * 3600
minutes_in_seconds = start_absolute_minutes * 60

start_in_seconds = hours_in_seconds + minutes_in_seconds + start_absolute_seconds
time = start_in_seconds + time_reconstructed

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

CDtime = date_times

#####################################
#####################################

post_outage_drift = [None] * num_datasets
post_outage_noise = [None] * num_datasets

for i in range(num_datasets):
    loop_range = np.arange(0, samples - window_sizes[i], window_shift)
    post_outage_drift[i] = np.load('Post_Outage/' + file_labels[i] + '/default_slopes_nonmarkov.npy')
    drift_slope_axs[1].plot(CDtime.iloc[window_sizes[i]-1+loop_range], post_outage_drift[i][0,:], lw = line_width, color = color_code[i])
for i in range(num_datasets):
    loop_range = np.arange(0, samples - window_sizes[i], window_shift)
    post_outage_noise[i] = np.load('Post_Outage/' + file_labels[i] + '/default_noise_nonmarkov.npy')
    noise_level_axs[1].plot(CDtime.iloc[window_sizes[i]-1+loop_range],post_outage_noise[i][0,:], lw = line_width, color = color_code[i])

date_form = DateFormatter("%H:%M")
drift_slope_axs[1].xaxis.set_major_formatter(date_form)
drift_slope_axs[1].xaxis.set_major_formatter(date_form)
noise_level_axs[1].xaxis.set_major_formatter(date_form)
noise_level_axs[1].xaxis.set_major_formatter(date_form)

drift_slope_axs[1].plot([CDtime.iloc[0], CDtime.iloc[-1]],[0,0], ls = ':', color = 'red', lw = line_width)

hatch_width =1.5
zorder_interval_markers = 0
drift_slope_axs[1].axvspan(CDtime.iloc[0], datetime.strptime('1996-08-10 15:48:54.95', "%Y-%m-%d %H:%M:%S.%f"), facecolor="none", edgecolor = 'black', hatch = '/', linewidth = hatch_width)
drift_slope_axs[1].axvspan(datetime.strptime('1996-08-10 18:47:00.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 21:42:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkorange', zorder = zorder_interval_markers)
drift_slope_axs[1].axvspan(datetime.strptime('1996-08-10 21:42:00.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-11 01:00:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkgreen', zorder = zorder_interval_markers)
noise_level_axs[1].axvspan(CDtime.iloc[0], datetime.strptime('1996-08-10 15:48:54.95', "%Y-%m-%d %H:%M:%S.%f"), facecolor="none", edgecolor = 'black', hatch = '/', linewidth = hatch_width)
noise_level_axs[1].axvspan(datetime.strptime('1996-08-10 18:47:00.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-10 21:42:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkorange', zorder = zorder_interval_markers)
noise_level_axs[1].axvspan(datetime.strptime('1996-08-10 21:42:00.00', "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime('1996-08-11 01:00:00.00', "%Y-%m-%d %H:%M:%S.%f"), alpha=0.3, color='darkgreen', zorder = zorder_interval_markers)
noise_level_axs[1].xaxis.set_major_locator(mdates.HourLocator(byhour=np.array([18,21,0,3,6,9]), interval=1, tz=None))# np.array([16,18,20,22,0,2,4,6,8,10])
noise_level_axs[1].set_xticklabels(noise_level_axs[1].get_xticklabels(), rotation = 30, ha="right", rotation_mode = "anchor")

drift_slope_axs[1].set_xlim(CDtime.iloc[0],CDtime.iloc[-1])
noise_level_axs[1].set_xlim(CDtime.iloc[0],CDtime.iloc[-1])

noise_level_axs[1].set_xlabel(r'Time $t$ after outage', fontsize = label_size)

drift_slope_axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
drift_slope_axs[1].get_yaxis().get_offset_text().set_visible(False)
drift_slope_axs[1].text(x_pos_powertext, drift1_y_pos_powertext, r'$\times 10^{-2}$', transform=drift_slope_axs[1].transAxes, fontsize = power_text_factor*text_size)


noise_level_axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
noise_level_axs[1].get_yaxis().get_offset_text().set_visible(False)
noise_level_axs[1].text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-3}$', transform=noise_level_axs[1].transAxes, fontsize = power_text_factor*text_size)


######################### END POST-OUTAGE PLOT #############################
############################################################################


######################### PLOT ALL MODEL SCENARIOS #############################
################################################################################

time = np.linspace(0,2000,4000000)
time = time[0::100]

plot_times = [None] * 3
plot_times[0] = time
plot_times[1] = time
time = np.arange(0,4000,0.05)
plot_times[2] = time

samples = [40000, 40000, 80000]
window_sizes = [4000, 6000, 10000, 12000]
time_labels = [[0, 1000], [0, 1000], [0, 2000]]

drift = [None] * num_datasets
noise = [None] * num_datasets
markov_drift = [None] * num_datasets
markov_noise = [None] * num_datasets

num_cases = 3
case_labels = ['Consumer_Noise_Modelling', 'Keeler_Allston_Modelling', 'McNary_Modelling']
file_labels = ['window_size4000', 'window_size6000', 'window_size10000', 'window_size12000']
axis_label = [2,3,4]

consumer_noise = np.load('Consumer_Noise_Modelling/consumer_noise.npy')
consumer_noise = consumer_noise[::100]

noise_time = np.arange(0,2000,0.05)

for j in range(num_cases):
    if j==2:
        file_labels[j] = 'window_size9000'
        window_sizes[j] = 9000
        num_datasets = 3
        color_code[2] = 'k'
    for i in range(num_datasets):
        loop_range = np.arange(0, samples[j] - window_sizes[i], window_shift)
        drift[i] = np.load('' + case_labels[j] + '/' + file_labels[i] + '/default_save_slopes.npy')
        noise[i] = np.load('' + case_labels[j] + '/' + file_labels[i] + '/default_save_noise.npy')
        if j!=1:
            drift_slope_axs[axis_label[j]].plot(plot_times[j][window_sizes[i]-1+loop_range], drift[i][0,:], lw = line_width, color = color_code[i])
            noise_level_axs[axis_label[j]].plot(plot_times[j][window_sizes[i]-1+loop_range], noise[i][0,:], lw = line_width, color = color_code[i])
        else:
            drift_slope_axs[axis_label[j]].plot(plot_times[j][window_sizes[i]-1+loop_range], drift[i][0,:], lw = line_width, color = color_code[i], label = window_size_label[i])
            noise_level_axs[axis_label[j]].plot(plot_times[j][window_sizes[i]-1+loop_range], noise[i][0,:], lw = line_width, color = color_code[i])
        
        drift_slope_axs[axis_label[j]].set_xlim(plot_times[j][0], plot_times[j][-1])
        noise_level_axs[axis_label[j]].set_xlim(plot_times[j][0], plot_times[j][-1])

        noise_level_axs[axis_label[j]].set_xlabel(r'Time $t$', fontsize = label_size)
        if j==1:
            markov_drift[i] = np.load('' + case_labels[j] + '/' + file_labels[i] + '/Markov/default_save_slopes.npy')
            drift_slope_axs[axis_label[j]].plot(plot_times[j][window_sizes[i]-1+loop_range], markov_drift[i][0,:], ls=(0, (3, 1, 1, 1, 1, 1)), lw = line_width, color = color_code[i], label = window_size_label[i])

            markov_noise[i] = np.load('' + case_labels[j] + '/' + file_labels[i] + '/Markov/default_save_noise.npy')
            noise_level_axs[axis_label[j]].plot(plot_times[j][window_sizes[i]-1+loop_range], markov_noise[i][0,:], ls=(0, (3, 1, 1, 1, 1, 1)), lw = line_width, color = color_code[i])

drift_slope_axs[3].plot([0,1],[-10,-10], lw = line_width, color = 'k', label = '+12.5%')
handles, labels = drift_slope_axs[3].get_legend_handles_labels()
order = [0,2,8,4,6,1,3,5,7]

drift_slope_axs[3].legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                            title='Window Size Variation\n NBLE                  BLE', ncols = 2, loc='lower left', 
                            bbox_to_anchor=(1.15,1.65), handlelength = 1.5,fancybox=True, shadow=True,
                            handletextpad = 0.8, columnspacing = 1.8, fontsize = legend_size, title_fontsize = legend_size)

drift_slope_axs[3].set_ylim(-2.280, -0.176)
drift_slope_axs[2].set_ylabel(r'Drift slope', fontsize = label_size)
noise_level_axs[2].set_ylabel(r'Noise level', fontsize = label_size)

noise_level_axs[2].plot(noise_time[:], consumer_noise[:], ls=':', color = 'g', lw=line_width, label=r'$\sigma_{\bar{C}}$')
noise_level_axs[2].legend(fontsize=legend_size, loc = 'upper center')

init_plateau_size = 1000000
ramp_size  =1000000
end_plateau_size = 2000000
thinning = 100
avoid_window_linetrip_overlap = 800000
keeler_time = np.arange(0,2000,0.05)
keeler_tripping_index = int((init_plateau_size+ramp_size+avoid_window_linetrip_overlap)/thinning)
drift_slope_axs[3].axvspan(keeler_time[int(init_plateau_size/thinning)], keeler_time[int((init_plateau_size+ramp_size)/thinning)], color = 'darkred', alpha = 0.3)
noise_level_axs[3].axvspan(keeler_time[int(init_plateau_size/thinning)], keeler_time[int((init_plateau_size+ramp_size)/thinning)], color = 'darkred', alpha = 0.3)
drift_slope_axs[3].axvline(keeler_time[keeler_tripping_index], ls='--', color = 'gray', lw = line_width)
noise_level_axs[3].axvline(keeler_time[keeler_tripping_index], ls='--', color = 'gray', lw = line_width)

McNary_index = 2000000
damping_index = 7000000
damping_level = 0.07
load_imbalance_interval = 2000000
delay = 1000000
compensation_interval = 100000 ### has to be smaller than load imbalance interval
load_imbalance_factor = 1.
line_cut_delay = 10

overload_shock = 0

mcnary_loss_index = int((McNary_index+load_imbalance_interval)/thinning)
primary_control_init = mcnary_loss_index+int(delay/thinning)
line_trip = mcnary_loss_index + int(line_cut_delay/thinning)
no_damping_index = int(damping_index/thinning)

mcnary_time = np.arange(0,4000,0.05)

drift_slope_axs[4].axvspan(mcnary_time[mcnary_loss_index], mcnary_time[primary_control_init], color = 'darkred', alpha = 0.3)
drift_slope_axs[4].axvline(mcnary_time[line_trip], ls='--', color = 'gray', lw = line_width)
drift_slope_axs[4].axvline(mcnary_time[no_damping_index], ls='--', color = 'r', lw = line_width)
noise_level_axs[4].axvspan(mcnary_time[mcnary_loss_index], mcnary_time[primary_control_init], color = 'darkred', alpha = 0.3)
noise_level_axs[4].axvline(mcnary_time[line_trip], ls='--', color = 'gray', lw = line_width)
noise_level_axs[4].axvline(mcnary_time[no_damping_index], ls='--', color = 'r', lw = line_width)

drift_slope_axs[0].text(x_fig_label, y_fig_label, 'a', transform=drift_slope_axs[0].transAxes, fontsize = text_size, weight = 'bold')
noise_level_axs[0].text(x_fig_label, y_fig_label, 'b', transform=noise_level_axs[0].transAxes, fontsize = text_size, weight = 'bold')

drift_slope_axs[1].text(x_fig_label, y_fig_label, 'c', transform=drift_slope_axs[1].transAxes, fontsize = text_size, weight = 'bold')
noise_level_axs[1].text(x_fig_label, y_fig_label, 'd', transform=noise_level_axs[1].transAxes, fontsize = text_size, weight = 'bold')

drift_slope_axs[2].text(x_fig_label, y_fig_label, 'e', transform=drift_slope_axs[2].transAxes, fontsize = text_size, weight = 'bold')
noise_level_axs[2].text(x_fig_label, y_fig_label, 'f', transform=noise_level_axs[2].transAxes, fontsize = text_size, weight = 'bold')

drift_slope_axs[3].text(x_fig_label, y_fig_label, 'g', transform=drift_slope_axs[3].transAxes, fontsize = text_size, weight = 'bold')
noise_level_axs[3].text(x_fig_label, y_fig_label, 'h', transform=noise_level_axs[3].transAxes, fontsize = text_size, weight = 'bold')

drift_slope_axs[4].text(x_fig_label, y_fig_label, 'i', transform=drift_slope_axs[4].transAxes, fontsize = text_size, weight = 'bold')
noise_level_axs[4].text(x_fig_label, y_fig_label, 'j', transform=noise_level_axs[4].transAxes, fontsize = text_size, weight = 'bold')

drift_slope_axs[4].set_yticks([-4,-2])
noise_level_axs[0].set_yticks([0.01,0.02])
noise_level_axs[4].set_yticks([0.3,0.4])
noise_level_axs[3].set_yticks([0.3,0.6])
noise_level_axs[2].set_yticks([0.3,0.6])

noise_level_axs[2].set_xticks([0,1000])
noise_level_axs[3].set_xticks([0,1000])
noise_level_axs[4].set_xticks([0,2000])

plt.savefig('Figure_S15.pdf')
plt.show()

################################################################################
################################################################################