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

samples = 30000

pitchfork_drift_slope = [None] * 2
nu_array = np.linspace(-10., 2, samples)
pitchfork_drift_slope[0] = nu_array

nu_arrayII = np.linspace(-28, 5, samples)
pitchfork_drift_slope[1] = nu_arrayII

fold_drift_slope = [None] * 2
fold_drift_slope[0] = np.load('BN_tip_exampleII/analytical_BN_tip_exampleII_drift_slope.npy')
fold_drift_slope[1] = np.load('BN_tip_exampleII/analytical_BN_tip_exampleII_drift_slope.npy')

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

fold_time = time[:transition_index]
loop_range = np.arange(0, data_size - window_size, window_shift)

 
prepared_time[2] = fold_time
prepared_time[3] = fold_time[window_size + loop_range]

nrows = 2
ncols = 2
naxes = 2

time = np.linspace(start_time, end_time, samples)

axes_range = [-0.5, 0.5, -1.75,1.75]
y_axes_range = [0.1,0.3,0.65,0.95]
sublot_text = ['a','b','c','d']
xtick_array = [[0, 750,1500],[0, 650,1300]]

f = plt.figure(figsize=(8,5))
for i in range(4):
	print('i:', i)


	ag = axes_grid.Grid(f, (nrows, ncols, i+1), (naxes, 1), axes_pad=0)
	if i < 2:
		fold_pitchfork = 0
	else:
		fold_pitchfork = 2
	print(6 + 2*(i-2))
	ag2 = ag[0].twinx()
	ag2.set_axes_locator(ag[1].get_axes_locator())
	ag[0].set_xticks([0,500,1000,1500])
	ag[0].text(0.025,0.77, sublot_text[i], transform=ag[0].transAxes, fontsize = 18, weight='bold')
	ag[0].set_ylabel(r'Data $x$', fontsize = 18, labelpad = 0.)
	ag[0].set_ylim(axes_range[0+fold_pitchfork], axes_range[1+fold_pitchfork])
	ag2.set_xlim(prepared_time[0+fold_pitchfork][0], prepared_time[0+fold_pitchfork][-1])
	ag[1].set_ylim(y_axes_range[0+fold_pitchfork],y_axes_range[1+fold_pitchfork])
	ag[1].set_xlim(prepared_time[0+fold_pitchfork][0], prepared_time[0+fold_pitchfork][-1])
	ag[0].plot(prepared_time[0+fold_pitchfork], prepared_data[i])
	ag[1].plot(prepared_time[1+fold_pitchfork], prepared_data[5 + 2*(i)][0,:], color = 'r', lw = 2)
	twinax = ag[0].twinx()
	twinax_AR = ag[0].twinx()
	twinax_AR.set_axes_locator(ag[1].get_axes_locator())
	twinax_AR.spines['right'].set_position(("axes", 1.4))
	twinax_AR.tick_params(axis='y', colors='purple')
	ag[1].tick_params(axis='y', colors='r')
	twinax_AR.yaxis.label.set_color('purple')
	twinax_AR.set_ylabel(r'AR1 $\hat{\rho}_{1}$', fontsize = 18,  loc ='bottom')
	ag[1].yaxis.label.set_color('red')
	if 2*(i) < 8:
		twinax_AR.plot(prepared_time[1+fold_pitchfork], statistical_measures[2*(i)], ls = '-.', color = 'purple', lw = 2)
	ag[1].fill_between(prepared_time[1+fold_pitchfork], prepared_data[5 + 2*(i)][1,:], prepared_data[5 + 2*(i)][2,:], alpha = 0.6, color = 'green')
	ag[1].fill_between(prepared_time[1+fold_pitchfork], prepared_data[5 + 2*(i)][3,:], prepared_data[5 + 2*(i)][4,:], alpha = 0.4, color = 'green')
	ag[1].set_ylabel(r'Noise $\hat{\sigma}$', fontsize = 18, color = 'red', loc = 'bottom', labelpad = 10.5)
	if i < 2:
		ag[1].set_xticks(xtick_array[0])
	else:
		ag[1].set_xticks(xtick_array[1])
	if i == 0:
		ag[0].set_yticks([-0.5, 0., 0.5])
		ag[1].set_yticks([0.1,0.2])
	if i == 0:
		ag2.plot(prepared_time[0+fold_pitchfork], pitchfork_drift_slope[0][:prepared_time[0+fold_pitchfork].size], ls = ':', lw = 2, color = 'blue')
	if i == 1:
		ag2.plot(prepared_time[0+fold_pitchfork], pitchfork_drift_slope[1][:prepared_time[0+fold_pitchfork].size], ls = ':', lw = 2, color = 'blue')
	if i == 2:
		ag2.plot(prepared_time[0+fold_pitchfork], fold_drift_slope[0][:prepared_time[0+fold_pitchfork].size], ls = ':', lw = 2, color = 'blue')
	if i == 3:
		ag2.plot(prepared_time[0+fold_pitchfork], fold_drift_slope[1][:prepared_time[0+fold_pitchfork].size], ls = ':', lw = 2, color = 'blue')
	twinax.set_axes_locator(ag[1].get_axes_locator())
	twinax.set_ylabel(r'STD $\tilde{\sigma}$', fontsize = 18, loc = 'bottom')
	print(twinax)
	twinax.tick_params(axis='y', colors='k')
	ag2.tick_params(axis='y', colors='blue')
	twinax.yaxis.label.set_color('k')
	ag2.yaxis.label.set_color('blue')
	ag2.spines['left'].set_visible(True)
	ag2.tick_params(axis = 'y', colors = 'blue')
	ag2.yaxis.set_label_position('left')
	ag2.set_ylabel(r'Slopes $\hat{\zeta}$', fontsize = 18, loc = 'bottom', color = 'b')
	ag2.yaxis.set_ticks_position('left')
	ag2.spines['left'].set_position(('outward',70))

	ag2.plot(prepared_time[1+fold_pitchfork], prepared_data[4 + 2 * (i)][0,:], color = 'b')
	if 1 + 2*(i-2) < 8:
	 	twinax.plot(prepared_time[1+fold_pitchfork], statistical_measures[1 + 2*(i)], ls = '-.', color = 'k', lw = 2)
	ag2.fill_between(prepared_time[1+fold_pitchfork], prepared_data[4 + 2*(i)][1,:], prepared_data[4 + 2*(i)][2,:], alpha = 0.6, color = 'orange')
	ag2.fill_between(prepared_time[1+fold_pitchfork], prepared_data[4 + 2*(i)][3,:], prepared_data[4 + 2*(i)][4,:], alpha = 0.4, color = 'orange')
	if i < 2:
		
		if i == 0:
			ag[1].plot([0, prepared_time[0][-1]], [0.15,0.15], color = 'red', ls = ':', lw = 2)
		elif i == 1:
			ag[1].plot(prepared_time[0], BN_noise_level[0][:prepared_time[0].size], color = 'r', ls =':', lw = 2, alpha = 1.0)

		ag2.set_ylim(-30,10)
		twinax_AR.set_ylim(-1,1)
		twinax.set_ylim(-0.1,0.1)
		twinax.set_yticks([0,0.1])
	else:
		ag[0].set_yticks([-1.5,0,1.5])
		if i == 2:
			ag[1].set_ylim(0.7,1.1)
			ag[1].set_yticks([0.7,0.9])
			ag[1].plot([0, prepared_time[0][-1]], [0.75,0.75], color = 'red', ls = ':', lw = 2)
		elif i == 3:	
			ag[1].plot(prepared_time[2], BN_noise_level[1][:prepared_time[2].size], color = 'red', ls = ':', lw = 2)
		ag2.set_ylim(-22,10)
		twinax_AR.set_ylim(-1.1,1)
		twinax.set_ylim(-0.1,0.4)
		twinax.set_yticklabels([0,0.2,0.4])
	if i%ncols == 0:
		twinax.get_yaxis().set_visible(False)
		twinax_AR.get_yaxis().set_visible(False)
	if i%ncols == 1:
		ag[0].get_yaxis().set_visible(False)
		ag[1].get_yaxis().set_visible(False)
		ag2.get_yaxis().set_visible(False)
		ag2.spines['left'].set_visible(False)

	if i == 0 or i == 1:
		current_values =ag[0].get_yticks()
		ag[0].set_yticklabels(['{:,.1f}'.format(x) for x in current_values])
		ag[0].set_yticklabels([f'{x}'.replace('-', '\N{MINUS SIGN}') for x in ag[0].get_yticks()])		

	if i == 2 or i == 3:
		ag[1].set_xlabel(r'Time $t$', fontsize = 18)
		current_values =ag[0].get_yticks()
		ag[0].set_yticklabels(['{:,.1f}'.format(x) for x in current_values])
		ag[0].set_yticklabels([f'{x}'.replace('-', '\N{MINUS SIGN}') for x in ag[0].get_yticks()])


plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.2, left = 0.255, bottom = 0.143, right = 0.783, top = 0.97)
plt.savefig('Figure_S1.pdf')
plt.show()
