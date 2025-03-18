from mpl_toolkits.axes_grid1 import axes_grid
from matplotlib.legend import Legend
import numpy as np
import pylab
import time
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
from mpl_toolkits.axes_grid1 import axes_grid
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from scipy.ndimage.filters import gaussian_filter

prepared_data = [None] * 12

prepared_data[0] = np.load('BN_tip_example/BN_tipping_transition_cut.npy')
prepared_data[1] = np.load('BN_tip_example/BN_II_tipping_transition_cut.npy')

prepared_data[2] = np.load('BN_tip_exampleII/detrended_B_tip_fold_cutted.npy')
prepared_data[3] = np.load('BN_tip_exampleII/detrended_BN_II_tip_fold_cutted.npy')

prepared_data[4] = np.load('BN_tip_example/BN_tip_slopes.npy')
prepared_data[5] = np.load('BN_tip_example/BN_tip_noise.npy')
prepared_data[6] = np.load('BN_tip_example/BN_tip_II_slopes.npy')
prepared_data[7] = np.load('BN_tip_example/BN_tip_II_noise.npy')

prepared_data[8] = np.load('BN_tip_exampleII/B_tip_slopes.npy')
prepared_data[9] = np.load('BN_tip_exampleII/B_tip_noise.npy')
prepared_data[10] = np.load('BN_tip_exampleII/BN_II_fold_tip_slopes.npy')
prepared_data[11] = np.load('BN_tip_exampleII/BN_II_fold_tip_noise.npy')


statistical_measures = [None] * 8
statistical_measures[0] = np.load('BN_tip_example/AR1_BN_tip_I.npy')
statistical_measures[1] = np.load('BN_tip_example/std_BN_tip_I.npy')
statistical_measures[2] = np.load('BN_tip_example/AR1_BN_tip_II.npy')
statistical_measures[3] = np.load('BN_tip_example/std_BN_tip_II.npy')

statistical_measures[4] = np.load('BN_tip_exampleII/AR1_B_tip_fold.npy')
statistical_measures[5] = np.load('BN_tip_exampleII/std_B_tip_fold.npy')
statistical_measures[6] = np.load('BN_tip_exampleII/AR1_BN_fold_tip_II.npy')
statistical_measures[7] = np.load('BN_tip_exampleII/std_BN_fold_tip_II.npy')

window_size = 2000
window_shift = 150

start_time = 0.
end_time = 2000.
samples = 30000

dt = (end_time - start_time) / samples
time = np.linspace(start_time, end_time, samples)

#### Compute OUE Estimates ####
###############################

statistical_measures[0] = np.log(statistical_measures[0])/dt
statistical_measures[2] = np.log(statistical_measures[2])/dt
statistical_measures[4] = np.log(statistical_measures[4])/dt
statistical_measures[6] = np.log(statistical_measures[6])/dt

statistical_measures[1] = np.sqrt(-2. * statistical_measures[1]**2 * statistical_measures[0])
statistical_measures[3] = np.sqrt(-2. * statistical_measures[3]**2 * statistical_measures[2])
statistical_measures[5] = np.sqrt(-2. * statistical_measures[5]**2 * statistical_measures[4])
statistical_measures[7] = np.sqrt(-2. * statistical_measures[7]**2 * statistical_measures[6])

###############################
###############################

true_slopes = [None] * 4

r_arrayI = np.linspace(-10., 2, samples)
true_slopes[0] = r_arrayI  

r_arrayII = np.linspace(-28, 5, samples)
true_slopes[1] = r_arrayII 

true_slopes[2] = np.load('BN_tip_exampleII/analytical_BN_tip_exampleII_drift_slope.npy')
true_slopes[3] = np.load('BN_tip_exampleII/analytical_BN_tip_exampleII_drift_slope.npy')


burn_in = 500
r_array = np.linspace(10.2, -2, samples + burn_in)
for i in range(samples):
    if r_array[i] <= 0:
        transition_index = i
        break


BN_noise_level = [None] * 2

BN_noise_level[0] = np.load('BN_tip_exampleII/baseline_noise.npy')
BN_noise_level[1] = np.linspace(0.7,0.9, samples)

prepared_time = [None] * 4

pitchfork_data_size = prepared_data[0].size
pitchfork_time = time[:transition_index-burn_in]

loop_range = np.arange(0, pitchfork_data_size - window_size, window_shift)

prepared_time[0] = pitchfork_time
prepared_time[1] = pitchfork_time[window_size + loop_range]

data_size = prepared_data[2].size

loop_range = np.arange(0, data_size - window_size, window_shift)

r_array = np.linspace(-15, 5, samples)

for i in range(samples):
    if r_array[i] >= -0.5:
        transition_index = i
        break

time = np.linspace(start_time, end_time, samples)

fold_time = time[:transition_index]
loop_range = np.arange(0, data_size - window_size, window_shift)

 
prepared_time[2] = fold_time
prepared_time[3] = fold_time[window_size + loop_range]

nrows = 2
ncols = 2
naxes = 2

fig_labels = np.array(['a','b','c','d'])

f = plt.figure(figsize=(6.5, 5))
for i in range(4):
    axs = axes_grid.Grid(f, (nrows, ncols, i+1), (naxes, 1), axes_pad=0)

    if i == 0:
        axs[0].text(0.025,0.77, fig_labels[i], transform=axs[0].transAxes, fontsize = 18, weight='bold')
        axs[0].plot(prepared_time[0], prepared_data[0])
        axs[0].set_xlim(prepared_time[0][0], prepared_time[0][-1])
        axs[0].set_ylabel(r'Data $x$', fontsize = 18, loc = 'top')
        axs[0].set_ylim(-0.6, 0.6)
        axs[0].set_yticks([-0.3,0,0.3])
        axs[0].set_xticks([0,750,1500])
        axs[1].plot(prepared_time[0], true_slopes[0][:prepared_time[0].size], color ='b', ls=':', label = r'Ground truth $\zeta$')
        axs[1].plot(prepared_time[1], prepared_data[4][0,:], color = 'b', label = r'BLE $\hat{\zeta}$')
        axs[1].fill_between(prepared_time[1], prepared_data[4][1,:], prepared_data[4][2,:], alpha = 0.6, color = 'orange')
        axs[1].fill_between(prepared_time[1], prepared_data[4][3,:], prepared_data[4][4,:], alpha = 0.4, color = 'orange')
        axs[1].plot(prepared_time[1], statistical_measures[0], ls = '-.', color = 'purple', lw = 2, label = r'OUE $\hat{\zeta}_{\rm OU}$')
        axs[1].set_ylabel(r'Slopes', fontsize = 18, loc = 'bottom')
        axs[1].set_xlim(prepared_time[0][0], prepared_time[0][-1])
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].legend(handles[:1], labels[:1], frameon = False, labelspacing = 0.1, fontsize = 12, handlelength = 0.5, loc = 'upper left', bbox_to_anchor = [-0.05,1.1])
        leg2 = Legend(axs[1], handles[1:], labels[1:], frameon = False, labelspacing = 0.1, fontsize = 12, handlelength = 0.5, loc = 'lower left', bbox_to_anchor = [0.5,-0.1])
        axs[1].add_artist(leg2)
    if i == 1:
        axs[0].text(0.025,0.77, fig_labels[i], transform=axs[0].transAxes, fontsize = 18, weight='bold')
        axs[0].plot(prepared_time[0], prepared_data[1])
        axs[0].yaxis.set_visible(False)
        axs[0].set_xticks([0,750,1500])
        axs[0].set_ylim(-0.6, 0.6)
        axs[1].yaxis.tick_right()
        axs[0].set_xlim(prepared_time[0][0], prepared_time[0][-1])
        axs[1].set_xlim(prepared_time[0][0], prepared_time[0][-1])
        axs[1].plot(prepared_time[0], true_slopes[1][:prepared_time[0].size], color ='b', ls=':')
        axs[1].plot(prepared_time[1], prepared_data[6][0,:], color = 'b')
        axs[1].fill_between(prepared_time[1], prepared_data[6][1,:], prepared_data[6][2,:], alpha = 0.6, color = 'orange')
        axs[1].fill_between(prepared_time[1], prepared_data[6][3,:], prepared_data[6][4,:], alpha = 0.4, color = 'orange')
        axs[1].plot(prepared_time[1], statistical_measures[2], ls = '-.', color = 'purple', lw = 2)
    if i == 2:
        axs[0].text(0.025,0.77, fig_labels[i], transform=axs[0].transAxes, fontsize = 18, weight='bold')
        axs[0].plot(prepared_time[2], prepared_data[2])
        axs[0].set_ylabel(r'Data $x$', fontsize = 18, loc = 'top')
        axs[1].set_ylabel(r'Slopes', fontsize = 18, loc = 'bottom')
        axs[0].set_xlim(prepared_time[2][0], prepared_time[2][-1])
        axs[0].set_ylim(-1.85,1.85)
        axs[0].set_yticks([-1.3, 0, 1.3])
        axs[0].set_xticks([0,650,1300])
        axs[1].set_ylim(-90,10)
        axs[1].set_yticks([-90,0])
        axs[1].set_xlim(prepared_time[2][0], prepared_time[2][-1])
        axs[1].plot(prepared_time[2], true_slopes[2][:prepared_time[2].size], color ='b', ls=':')
        axs[1].plot(prepared_time[3], prepared_data[8][0,:], color = 'b')
        axs[1].fill_between(prepared_time[3], prepared_data[8][1,:], prepared_data[8][2,:], alpha = 0.6, color = 'orange')
        axs[1].fill_between(prepared_time[3], prepared_data[8][3,:], prepared_data[8][4,:], alpha = 0.4, color = 'orange')
        axs[1].plot(prepared_time[3], statistical_measures[4], ls = '-.', color = 'purple', lw = 2)
    if i == 3:
        axs[0].text(0.025,0.77, fig_labels[i], transform=axs[0].transAxes, fontsize = 18, weight='bold')
        axs[0].plot(prepared_time[2], prepared_data[3])
        axs[0].set_xlim(prepared_time[2][0], prepared_time[2][-1])
        axs[1].set_xlim(prepared_time[2][0], prepared_time[2][-1])
        axs[1].yaxis.tick_right()
        axs[0].set_ylim(-1.85,1.85)
        axs[1].set_yticks([-90,0])
        axs[0].yaxis.set_visible(False)
        axs[0].set_xticks([0,650,1300])
        axs[1].plot(prepared_time[2], true_slopes[3][:prepared_time[2].size], color  = 'b', ls=':')
        axs[1].plot(prepared_time[3], prepared_data[10][0,:], color = 'b')
        axs[1].fill_between(prepared_time[3], prepared_data[10][1,:], prepared_data[10][2,:], alpha = 0.6, color = 'orange')
        axs[1].fill_between(prepared_time[3], prepared_data[10][3,:], prepared_data[10][4,:], alpha = 0.4, color = 'orange')
        axs[1].plot(prepared_time[3], statistical_measures[6], ls = '-.', color = 'purple', lw = 2)
plt.subplots_adjust(left = 0.15, top = 0.98, bottom = 0.15, right = 0.9, wspace = 0.1)
plt.savefig('Figure_S6_abcd.pdf')
plt.show()