import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.cbook import get_sample_data

cm_to_inch = 2.54
plt.rcParams["figure.figsize"] = (44/cm_to_inch,31/cm_to_inch)
label_size = 25 
title_size_reduction=2.5
legend_size = 17.5
tick_size = 25
text_size = 22
line_width = 10
pdf_line_width = 2.
power_text_factor = 1.
fig_label_add = 10

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size

def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")

fig = plt.figure(layout="constrained")

gs0 = fig.add_gridspec(7, 6)

data_axs = [None] * 2

stats_axs = [None]*9

data_axs[0] = fig.add_subplot(gs0[0,:3])
data_axs[1] = fig.add_subplot(gs0[0,3:6])

k = 0
arange = np.array([0,2,4,6])
for a in range(3):
    for b in range(3):
        stats_axs[k] = fig.add_subplot(gs0[1+arange[a]:1+arange[a+1],arange[b]:arange[b+1]])
        k += 1

line_width = 3

#### DATASETS ####
##################

BV_npz = np.load('data/BV_data_GKF5.npz')
BVdata = BV_npz['xx']
BVtime = BV_npz['tt']

dt=BVtime[1]-BVtime[0]

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

BV_interval1_start = 0
BV_interval1_end = int(600/dt)

BV_data_interval1 = np.load('BV_stats_1st_interval/BVdata_interval1.npy')
BV_time_intveral1 = np.load('BV_stats_1st_interval/BVtime_interval1.npy')
numberone = plt.imread('number1.png')

BV_interval2_start = int(700/dt)
BV_interval2_end = int(1064/dt)

BV_data_interval2 = np.load('BV_stats_2nd_interval/BVdata_interval2.npy')
BV_time_intveral2 = np.load('BV_stats_2nd_interval/BVtime_interval2.npy')
numbertwo = plt.imread('number2.png')


CDdata = np.load('data/restoration_data.npy')
CDtime = np.load('data/reconstructed_time_step.npy')
numberthree = plt.imread('number3.png')

dt = CDtime[1]-CDtime[0]
CD_data_interval1_start = int(25000/dt)
CD_time_intveral1_end = -1

start_absolute_hours = 15
start_absolute_minutes = 10
start_absolute_seconds = 45

hours_in_seconds = start_absolute_hours * 3600
minutes_in_seconds = start_absolute_minutes * 60

start_in_seconds = hours_in_seconds + minutes_in_seconds + start_absolute_seconds
time = start_in_seconds + CDtime

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
CDtime = time 

CD_data_interval1 = np.load('CD_stats/CDdata_interval1.npy')
CD_time_intveral1 = np.load('CD_stats/CDtime_interval1.npy')

data_axs[0].plot(BVtime,BVdata, lw = line_width)
data_axs[0].plot(BVtime.iloc[BV_interval1_start:BV_interval1_end], BV_data_interval1, lw = line_width)
data_axs[0].plot(BVtime.iloc[BV_interval2_start:BV_interval2_end], BV_data_interval2, lw = line_width, color = 'tab:orange')
data_axs[0].set_title(r'NAWI Pre-Outage $\tilde{\omega}_{\rm P}(t)$, 10th August 1996', fontsize = label_size-title_size_reduction, loc = 'right')
data_axs[0].set_ylabel(r'Bus volt. freq. $\tilde{\omega}$    ', fontsize = label_size, loc = 'center')
date_form = DateFormatter("%H:%M")
data_axs[0].xaxis.set_major_formatter(date_form)
data_axs[0].xaxis.set_major_locator(mdates.MinuteLocator(byminute=np.array([33,39,45]), tz=None))
data_axs[0].set_xlabel(r'Time $t$ before outage', fontsize = label_size)
data_axs[0].set_xlim([BVtime.iloc[0],BVtime.iloc[-1]])
data_axs[0].set_ylim(-0.06,0.09)
data_axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)

num_ax = fig.add_axes([0.13, 0.914, 0.038, 0.038], anchor='NE', zorder=10)
num_ax.imshow(numberone)
num_ax.axis('off')

num_ax2 = fig.add_axes([0.345, 0.914, 0.038, 0.038], anchor='NE', zorder=10)
num_ax2.imshow(numbertwo)
num_ax2.axis('off')

data_axs[1].set_title(r'NAWI Restoration $\tilde{\omega}_{\rm R}(t)$, 10th August 1996', fontsize = label_size-title_size_reduction, loc = 'right')
data_axs[1].plot(CDtime,CDdata, lw = line_width)
data_axs[1].plot(CDtime.iloc[CD_data_interval1_start:CD_time_intveral1_end], CD_data_interval1, lw = line_width)
data_axs[1].set_xlabel(r'Time $t$ after outage', fontsize = label_size)
data_axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
date_form = DateFormatter("%H:%M")
data_axs[1].xaxis.set_major_formatter(date_form)
data_axs[1].xaxis.set_major_locator(mdates.HourLocator(byhour=np.array([18,0,6]), tz=None))
data_axs[1].set_xlim([CDtime.iloc[0],CDtime.iloc[-1]])

num_ax3 = fig.add_axes([0.8, 0.914, 0.038, 0.038], anchor='NE', zorder=10)
num_ax3.imshow(numberthree)
num_ax3.axis('off')


label = ['data', 'BLE', 'NBLE']
quantity = ['acf', 'kde_fit', 'kde_fit_incr']
directory = ['BV_stats_1st_interval', 'BV_stats_2nd_interval', 'CD_stats']

col_titles = [r'Autocorrelation AC$(\tau)$', r'Raw Data PDF $p(\tilde{\omega})$', r'Increment PDF $p(\tilde{\omega}_{n+1}-\tilde{\omega}_{n})$']
xlabels = [r'Time lag $\tau$ $[\Delta t]$', r'Bus voltage frequency $\tilde{\omega}$', r'Frequency increments $\tilde{\omega}_{n+1}-\tilde{\omega}_{n}$']
AC_lag_limits = np.array([60,60,60])
kde_grid = np.linspace(-0.04,0.04, 10000)

k = 0
for row in range(3):
    direct = directory[row]

    for col in range(3):
        for i in range(3):
            loaded_quantity = np.load(direct+'/'+ quantity[col] + '_' + label[i] + '.npy')
            if col==0:
                if i==0:
                    for j in range(loaded_quantity.size):
                        if loaded_quantity[j] <= -0.03:
                            zero_crossing = j
                            break
                    if row < 2:
                        raw_AC = loaded_quantity[:zero_crossing]
                        stats_axs[k].axvspan(zero_crossing,AC_lag_limits[row], color = 'gray', alpha = 0.3)
                    else:
                        raw_AC = loaded_quantity
                    if row == 0:
                        stats_axs[k].plot(loaded_quantity, lw = line_width, marker = 'o', label = 'Raw ' + label[i])
                    else:
                        stats_axs[k].plot(loaded_quantity, lw = line_width, marker = 'o')
                    stats_axs[k].plot([0,100], [0,0], color ='gray', lw = line_width)
                else:
                    SSE = np.sum((loaded_quantity[:raw_AC.size] - raw_AC)**2)
                    SSE = np.round(SSE, 2)
                    stats_axs[k].plot(loaded_quantity,lw = line_width, ls = '--', label = label[i] + ' (SSE: ' + str(SSE) + ')')
                if row==1:
                    stats_axs[k].legend(fontsize = legend_size, frameon = False, borderpad = 0.1, handlelength = 0.4, handletextpad=0.2)
                elif row==0:
                    stats_axs[k].legend(fontsize = legend_size, frameon = True, borderpad = 0.1, handlelength = 0.4, handletextpad=0.2, loc = 'upper left', bbox_to_anchor=[0.16,1])
                else:
                    stats_axs[k].legend(fontsize = legend_size, ncol=2, frameon = False, borderpad = 0.1, handlelength = 0.5, loc = 'lower center', columnspacing=0.5, handletextpad=0.2)
            else:
                if i == 0:
                    stats_axs[k].plot(kde_grid,loaded_quantity, lw = pdf_line_width)
                else:
                    stats_axs[k].plot(kde_grid,loaded_quantity, lw = pdf_line_width, ls='--')
        if row==0:
            stats_axs[k].set_title(col_titles[col], fontsize = label_size)
        if col==0:
            stats_axs[k].set_yticks([0,1])
            stats_axs[k].set_ylim(-0.47,1)
            stats_axs[k].set_xlim(0,AC_lag_limits[row])
        elif col>=1:
            stats_axs[k].set_xlim(-0.04,0.04)
            stats_axs[k].set_yticks([0,50])
            if col==2:
                stats_axs[k].yaxis.set_visible(False)
        if row<2:
            stats_axs[k].xaxis.set_visible(False)
        if row==2:
            stats_axs[k].set_xlabel(xlabels[col], fontsize = label_size)
        k+=1

num_ax12 = fig.add_axes([-0.01, 0.635, 0.038, 0.038], anchor='NE', zorder=10)
num_ax12.imshow(numberone)
num_ax12.axis('off')

num_ax22 = fig.add_axes([-0.01, 0.41, 0.038, 0.038], anchor='NE', zorder=10)
num_ax22.imshow(numbertwo)
num_ax22.axis('off')

num_ax32 = fig.add_axes([-0.01, 0.165, 0.038, 0.038], anchor='NE', zorder=10)
num_ax32.imshow(numberthree)
num_ax32.axis('off')

plt.savefig('Figure_S12.pdf', dpi = 200)
plt.show()