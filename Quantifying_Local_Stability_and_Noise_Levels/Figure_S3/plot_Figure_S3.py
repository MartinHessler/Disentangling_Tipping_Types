import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.cbook import get_sample_data
import svgutils.transform as sg
import sys 


cm_to_inch = 2.54
cm_width = 44
cm_height = 70
figuresize = (cm_width/cm_to_inch,cm_height/cm_to_inch)
plt.rcParams["figure.figsize"] = figuresize
label_size = 40 #20
legend_size = 20
tick_size = 35
text_size = 45
line_width = 3
pdf_line_width = 2.
power_text_factor = 0.35
fig_label_add = 10

sublabelsize = 20
subpowersize = 20

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size

x_fig_label, y_fig_label = 0.01, 0.85
x_pos_powertext,y_pos_powertext = 0.01, 0.78


def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")

fig = plt.figure(constrained_layout=False,figsize=(30,17))

gs00 = fig.add_gridspec(1, 3, top = 0.96, bottom = 0.72, left = 0.075, right = 0.99)
gs10 = fig.add_gridspec(3, 3, hspace = 0, top = 0.68,  bottom = 0.1, left = 0.075, right = 0.99)

axs00 = [None]*3
axs10 = [None]*9
k = 0
for a in range(3):
    for b in range(3):
        if a==0:
            axs00[k]=fig.add_subplot(gs00[a, b])
        axs10[k]=fig.add_subplot(gs10[a,b])
        k+=1

#### NAWI SCHEME FRAME SETTINGS ####
####################################
axs00[0].axes.set_visible(False)
####################################


#### PARAMETER TUNING EXAMPLE II ####
#####################################

reserve_capacity = np.load('KeelerAllston/reserve_capacity.npy')
keeler_time = np.arange(0,2000,0.05)

init_plateau_size = 1000000
ramp_size  =1000000
end_plateau_size = 2000000
thinning = 100

Keeler_KLevel = 7

reserve_capacity = reserve_capacity[::thinning]

end_plateau_size = 2000000

avoid_window_linetrip_overlap = 800000
keeler_tripping_index = int((init_plateau_size+ramp_size+avoid_window_linetrip_overlap)/thinning)

axs00[1].set_title('(ii) Keeler Allston THIF', fontsize = label_size)
axs00[1].text(x_fig_label, y_fig_label, 'e', transform=axs00[1].transAxes, fontsize = text_size, weight = 'bold')
axs00[1].plot(keeler_time, np.ones(keeler_time.size)*Keeler_KLevel, lw = line_width, label = r' Keeler-Allston $K_{GC}$')
axs00[1].plot(keeler_time, reserve_capacity, lw = line_width, ls = '--', label = r'Compensating grid $K_{\bar{G}\bar{C}}$')
axs00[1].set_ylabel(r'Capacity $K_{ij}$', fontsize = label_size)
axs00[1].axvspan(keeler_time[int(init_plateau_size/thinning)], keeler_time[int((init_plateau_size+ramp_size)/thinning)], color = 'darkred', alpha = 0.3)
axs00[1].axvline(keeler_time[keeler_tripping_index], ls='--', color = 'gray', lw = line_width)
axs00[1].set_xlim(keeler_time[0], keeler_time[-1])
axs00[1].legend(fontsize = legend_size, handlelength = 0.5, handletextpad = 0.3, frameon=True, borderpad = 0.1, loc = 'center left', bbox_to_anchor = (0.55,0.2))

#### PARAMETER TUNING EXAMPLE III ####
######################################

McNary_power = np.load('McNary/McNary_power.npy')
consumer_power = np.load('McNary/consumer_power.npy')
units_power = np.load('McNary/units_power.npy')

McNary_index = 2000000
damping_index = 7000000
damping_level = 0.07
load_imbalance_interval = 2000000
delay = 1000000
compensation_interval = 100000
load_imbalance_factor = 1.
line_cut_delay = 10

overload_shock = 0

McNary_power = McNary_power[::thinning]
consumer_power = consumer_power[::thinning]
units_power = units_power[::thinning]

mcnary_loss_index = int((McNary_index+load_imbalance_interval)/thinning)
primary_control_init = mcnary_loss_index+int(delay/thinning)
line_trip = mcnary_loss_index + int(line_cut_delay/thinning)
no_damping_index = int(damping_index/thinning)

mcnary_time = np.arange(0,4000,0.05)

axs00[2].set_title('(iii) McNary Loss', fontsize = label_size)
axs00[2].text(x_fig_label, y_fig_label, 'i', transform=axs00[2].transAxes, fontsize = text_size, weight = 'bold')
axs00[2].plot(mcnary_time, consumer_power, lw = line_width, label = r'Consumers $P_{\bar{C}}$')
axs00[2].axvline(mcnary_time[line_trip], ls='--', color = 'gray', lw = line_width)
axs00[2].plot(mcnary_time, units_power, lw = line_width, label = r'Generators $P_{\bar {G}}$')
axs00[2].plot(mcnary_time, McNary_power, lw = line_width, label = r'McNary $P_{G}$', ls='--')
axs00[2].set_ylabel(r'Power $P_i$', fontsize = label_size)
axs00[2].axvspan(mcnary_time[mcnary_loss_index], mcnary_time[primary_control_init], color = 'darkred', alpha = 0.3)
axs00[2].axvline(mcnary_time[no_damping_index], ls='--', color = 'r', lw = line_width)
axs00[2].set_xlim(mcnary_time[0], mcnary_time[-1])
axs00[2].legend(fontsize = legend_size, handlelength = 1.5, handletextpad = 0.3, frameon=True, loc = 'center left', bbox_to_anchor = (0.02,0.4), borderpad = 0.1)


#### CONSUMER NOISE ####
########################

freq_data = np.load('consumer_noise/detrended_input_frequencies1.npy')
slopes = np.load('consumer_noise/slopes_consumer_noise_1.npy')
noise = np.load('consumer_noise/noise_consumer_noise_1.npy')
nonmarkov_slopes = np.load('consumer_noise/NonMarkov/slopes_consumer_noise1.npy')
nonmarkov_noise = np.load('consumer_noise/NonMarkov/noise_consumer_noise1.npy')

consumer_noise = np.load('consumer_noise/consumer_noise.npy')
consumer_noise = consumer_noise[::100]

window_size = 8000
window_shift = 100

noise_time = np.arange(0,2000,0.05)


axs10[0].set_title('(i) Fast Scale Disturbance', fontsize=label_size)
axs10[0].text(x_fig_label, y_fig_label, 'b', transform=axs10[0].transAxes, fontsize = text_size, weight = 'bold')
axs10[0].plot(noise_time[:], freq_data[:], lw=line_width)
axs10[0].set_ylabel(r'Frequency $\tilde{\omega}_{i}$', fontsize = label_size)
axs10[0].xaxis.set_visible(False)

#### INSET ####
###############
sublabelsize = 20
x_pos_subaxispowertext,y_pos_subaxispowertext = 0.1,0.395 

inset_ax = axs10[0].inset_axes(
                       [0.1,0.125, 0.35, 0.3],  # [x, y, width, height] w.r.t. axes
                        xlim=[111, 119], ylim=[-0.65,0.65], # sets viewport &amp; tells relation to main axes
                        ) # 111, 120         1793,1808
inset_ax.plot(noise_time[:], freq_data[:], lw  = 1)

# inset_ax.set_yticks([-1,0,1])
inset_ax.yaxis.tick_right()
inset_ax.set_yticks([-0.6,0,0.6])
inset_ax.set_xticks([111, 119])
inset_ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
inset_ax.get_yaxis().get_offset_text().set_visible(False)
inset_ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-1}$', transform=inset_ax.transAxes, fontsize = power_text_factor*text_size)


inset_ax.tick_params(labelsize=sublabelsize)

axs10[0].indicate_inset_zoom(inset_ax, edgecolor = 'k')
#############
#############

axs10[3].text(x_fig_label, y_fig_label, 'c', transform=axs10[3].transAxes, fontsize = text_size, weight = 'bold')
axs10[3].plot(noise_time[window_size::window_shift], slopes[0,:], lw=line_width, color = 'b', label = r' BLE $\hat{\zeta}$')
axs10[3].fill_between(noise_time[window_size::window_shift], slopes[1,:], slopes[2,:] , color = 'orange', alpha  =0.6)
axs10[3].fill_between(noise_time[window_size::window_shift], slopes[3,:], slopes[4,:] , color = 'orange', alpha  =0.4)

axs10[6].text(x_fig_label, y_fig_label, 'd', transform=axs10[6].transAxes, fontsize = text_size, weight = 'bold')
axs10[6].plot(noise_time[window_size::window_shift], noise[0,:], lw=line_width, color = 'b', label = r' BLE $\hat{\sigma}$')
axs10[6].fill_between(noise_time[window_size::window_shift], noise[1,:], noise[2,:] , color = 'orange', alpha  =0.6)
axs10[6].fill_between(noise_time[window_size::window_shift], noise[3,:], noise[4,:] , color = 'orange', alpha  =0.4)
axs10[6].plot(noise_time[:], consumer_noise[:], ls=':', color = 'g', lw=line_width, label=r'$\sigma_{\bar{C}}$')

axs10[3].plot(noise_time[window_size::window_shift], nonmarkov_slopes[0,:], lw=line_width, color = 'r', label = r' NBLE $\hat{\zeta}_{\rm NBLE}$')
axs10[3].fill_between(noise_time[window_size::window_shift], nonmarkov_slopes[1,:], nonmarkov_slopes[2,:] , color = 'green', alpha  =0.6)
axs10[3].fill_between(noise_time[window_size::window_shift], nonmarkov_slopes[3,:], nonmarkov_slopes[4,:] , color = 'green', alpha  =0.4)

axs10[6].plot(noise_time[window_size::window_shift], nonmarkov_noise[0,:], lw=line_width, color = 'r', label = r' NBLE $\hat{\psi}$')
axs10[6].fill_between(noise_time[window_size::window_shift], nonmarkov_noise[1,:], nonmarkov_noise[2,:] , color = 'green', alpha  =0.6)
axs10[6].fill_between(noise_time[window_size::window_shift], nonmarkov_noise[3,:], nonmarkov_noise[4,:] , color = 'green', alpha  =0.4)

axs10[3].set_ylabel('Drift slope', fontsize = label_size)
axs10[3].xaxis.set_visible(False)
axs10[3].legend(fontsize = legend_size, handlelength = 0.5, handletextpad = 0.3, frameon=True, loc = 'center left', bbox_to_anchor = (0.02,0.625), borderpad = 0.1, ncols=2)


axs10[0].set_xlim(noise_time[0], noise_time[-1])
axs10[3].set_xlim(noise_time[0], noise_time[-1])

axs10[6].set_ylabel('Noise level', fontsize = label_size)
axs10[6].set_xlim(noise_time[0], noise_time[-1])
axs10[6].set_xlabel(r'Time $t$', fontsize = label_size)
axs10[6].legend(fontsize = legend_size, handlelength = 0.5, handletextpad = 0.3, frameon=True, loc = 'center left', bbox_to_anchor = (0.2,0.7), borderpad = 0.1)
axs10[6].xaxis.set_visible(True)


#### KEELER ALLSTON LINE THIF ####
##################################

freq_data = np.load('KeelerAllston/detrended_input_frequencies2.npy')
slopes = np.load('KeelerAllston/default_save_slopes.npy')
noise = np.load('KeelerAllston/default_save_noise.npy')
nonmarkov_slopes = np.load('KeelerAllston/NonMarkov/default_save_slopes.npy')
nonmarkov_noise = np.load('KeelerAllston/NonMarkov/default_save_noise.npy')

window_size = 8000
window_shift = 100

keeler_time = np.arange(0,2000,0.05)

axs10[1].text(x_fig_label, y_fig_label, 'f', transform=axs10[1].transAxes, fontsize = text_size, weight = 'bold')
axs10[1].plot(keeler_time[:], freq_data[:], lw=line_width)
axs10[1].axvspan(keeler_time[int(init_plateau_size/thinning)], keeler_time[int((init_plateau_size+ramp_size)/thinning)], color = 'darkred', alpha = 0.3)
axs10[1].axvline(keeler_time[keeler_tripping_index], ls='--', color = 'gray', lw = line_width)
axs10[1].xaxis.set_visible(False)

#### INSET ####
###############
sublabelsize = 20
x_pos_subaxispowertext,y_pos_subaxispowertext = 0.1,0.395

inset_ax = axs10[1].inset_axes(
                       [0.1,0.42, 0.35, 0.3],  # [x, y, width, height] w.r.t. axes
                        xlim=[76,83], ylim=[-0.65,0.65], # sets viewport &amp; tells relation to main axes
                        ) #76,82           205, 221            1216,1233
inset_ax.plot(keeler_time[:], freq_data[:], lw  = 1)

# inset_ax.set_yticks([-1,0,1])
inset_ax.yaxis.tick_right()
inset_ax.set_yticks([-0.6,0,0.6])
inset_ax.set_xticks([76, 83])
inset_ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
inset_ax.get_yaxis().get_offset_text().set_visible(False)
inset_ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-1}$', transform=inset_ax.transAxes, fontsize = power_text_factor*text_size)


inset_ax.tick_params(labelsize=sublabelsize)

axs10[1].indicate_inset_zoom(inset_ax, edgecolor = 'k')
#############
#############

axs10[4].text(x_fig_label, y_fig_label, 'g', transform=axs10[4].transAxes, fontsize = text_size, weight = 'bold')
axs10[4].plot(keeler_time[window_size::window_shift], slopes[0,:], lw=line_width, color = 'b', label = r' BLE $\hat{\zeta}$')
axs10[4].axvspan(keeler_time[int(init_plateau_size/thinning)], keeler_time[int((init_plateau_size+ramp_size)/thinning)], color = 'darkred', alpha = 0.3)
axs10[4].axvline(keeler_time[keeler_tripping_index], ls='--', color = 'gray', lw = line_width)
axs10[4].fill_between(keeler_time[window_size::window_shift], slopes[1,:], slopes[2,:] , color = 'orange', alpha  =0.6)
axs10[4].fill_between(keeler_time[window_size::window_shift], slopes[3,:], slopes[4,:] , color = 'orange', alpha  =0.4)

axs10[7].text(x_fig_label, y_fig_label, 'h', transform=axs10[7].transAxes, fontsize = text_size, weight = 'bold')
axs10[7].plot(keeler_time[window_size::window_shift], noise[0,:], lw=line_width, color = 'b', label = r' BLE $\hat{\sigma}$')
axs10[7].axvspan(keeler_time[int(init_plateau_size/thinning)], keeler_time[int((init_plateau_size+ramp_size)/thinning)], color = 'darkred', alpha = 0.3)
axs10[7].axvline(keeler_time[keeler_tripping_index], ls='--', color = 'gray', lw = line_width)
axs10[7].fill_between(keeler_time[window_size::window_shift], noise[1,:], noise[2,:] , color = 'orange', alpha  =0.6)
axs10[7].fill_between(keeler_time[window_size::window_shift], noise[3,:], noise[4,:] , color = 'orange', alpha  =0.4)

axs10[4].plot(keeler_time[window_size::window_shift], nonmarkov_slopes[0,:], lw=line_width, color = 'r', label = r' NBLE $\hat{\zeta}_{\rm NBLE}$')
axs10[4].fill_between(keeler_time[window_size::window_shift], nonmarkov_slopes[1,:], nonmarkov_slopes[2,:] , color = 'green', alpha  =0.6)
axs10[4].fill_between(keeler_time[window_size::window_shift], nonmarkov_slopes[3,:], nonmarkov_slopes[4,:] , color = 'green', alpha  =0.4)

axs10[7].plot(keeler_time[window_size::window_shift], nonmarkov_noise[0,:], lw=line_width, color = 'r', label = r' NBLE $\hat{\psi}$')
axs10[7].fill_between(keeler_time[window_size::window_shift], nonmarkov_noise[1,:], nonmarkov_noise[2,:] , color = 'green', alpha  =0.6)
axs10[7].fill_between(keeler_time[window_size::window_shift], nonmarkov_noise[3,:], nonmarkov_noise[4,:] , color = 'green', alpha  =0.4)

axs10[4].xaxis.set_visible(False)

axs10[1].set_xlim(keeler_time[0], keeler_time[-1])
axs10[4].set_xlim(keeler_time[0], keeler_time[-1])

axs10[7].set_xlim(keeler_time[0], keeler_time[-1])
axs10[7].set_xlabel(r'Time $t$', fontsize = label_size)
axs10[7].xaxis.set_visible(True)


#### MCNARY LOSS ####
#####################


freq_data = np.load('McNary/detrended_input_frequencies2.npy')
slopes = np.load('McNary/default_save_slopes.npy')
noise = np.load('McNary/default_save_noise.npy')
nonmarkov_slopes = np.load('McNary/NonMarkov/combined_default_slopes_nonmarkov.npy')
nonmarkov_noise = np.load('McNary/NonMarkov/combined_default_noise_nonmarkov.npy')

window_size = 8000
window_shift = 100

mcnary_time = np.arange(0,4000,0.05)

axs10[2].text(x_fig_label, y_fig_label, 'j', transform=axs10[2].transAxes, fontsize = text_size, weight = 'bold')
axs10[2].plot(mcnary_time[:], freq_data[:], lw=line_width)
axs10[2].axvspan(mcnary_time[mcnary_loss_index], mcnary_time[primary_control_init], color = 'darkred', alpha = 0.3)
axs10[2].axvline(mcnary_time[line_trip], ls='--', color = 'gray', lw = line_width)
axs10[2].axvline(mcnary_time[no_damping_index], ls='--', color = 'r', lw = line_width)
axs10[2].xaxis.set_visible(False)

#### INSET ####
###############
sublabelsize = 20
x_pos_subaxispowertext,y_pos_subaxispowertext = 0.1, 0.8 

inset_ax = axs10[2].inset_axes(
                       [0.0625,0.575, 0.35, 0.3],  # [x, y, width, height] w.r.t. axes
                        xlim=[418.5,425], ylim=[-0.65,0.65], # sets viewport &amp; tells relation to main axes
                        ) #    418,424      416,427      1000,1013
inset_ax.plot(mcnary_time[:], freq_data[:], lw  = 1)

inset_ax.set_xticks([419, 425])
inset_ax.xaxis.tick_top()
inset_ax.yaxis.tick_right()
inset_ax.set_yticks([-0.6,0,0.6])
inset_ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
inset_ax.get_yaxis().get_offset_text().set_visible(False)
inset_ax.text(x_pos_powertext,y_pos_powertext, r'$\times 10^{-1}$', transform=inset_ax.transAxes, fontsize = power_text_factor*text_size)

inset_ax.tick_params(labelsize=sublabelsize)

axs10[2].indicate_inset_zoom(inset_ax, edgecolor = 'k')
#############
#############

axs10[5].text(x_fig_label, y_fig_label, 'k', transform=axs10[5].transAxes, fontsize = text_size, weight = 'bold')
axs10[5].plot(mcnary_time[window_size::window_shift], slopes[0,:], lw=line_width, color = 'b', label = r' BLE $\hat{\zeta}$')
axs10[5].fill_between(mcnary_time[window_size::window_shift], slopes[1,:], slopes[2,:] , color = 'orange', alpha  =0.6)
axs10[5].axvspan(mcnary_time[mcnary_loss_index], mcnary_time[primary_control_init], color = 'darkred', alpha = 0.3)
axs10[5].axvline(mcnary_time[line_trip], ls='--', color = 'gray', lw = line_width)
axs10[5].axvline(mcnary_time[no_damping_index], ls='--', color = 'r', lw = line_width)

axs10[8].text(x_fig_label, y_fig_label, 'l', transform=axs10[8].transAxes, fontsize = text_size, weight = 'bold')
axs10[8].plot(mcnary_time[window_size::window_shift], noise[0,:], lw=line_width, color = 'b', label = r' BLE $\hat{\sigma}$')
axs10[8].fill_between(mcnary_time[window_size::window_shift], noise[1,:], noise[2,:] , color = 'orange', alpha  =0.6)
axs10[8].fill_between(mcnary_time[window_size::window_shift], noise[3,:], noise[4,:] , color = 'orange', alpha  =0.4)
axs10[8].axvspan(mcnary_time[mcnary_loss_index], mcnary_time[primary_control_init], color = 'darkred', alpha = 0.3)
axs10[8].axvline(mcnary_time[line_trip], ls='--', color = 'gray', lw = line_width)
axs10[8].axvline(mcnary_time[no_damping_index], ls='--', color = 'r', lw = line_width)

axs10[5].plot(mcnary_time[window_size::window_shift], nonmarkov_slopes[0,:], lw=line_width, color = 'r', label = r' NBLE $\hat{\zeta}_{\rm NBLE}$', zorder = 5)
axs10[5].fill_between(mcnary_time[window_size::window_shift], nonmarkov_slopes[1,:], nonmarkov_slopes[2,:] , color = 'green', alpha  =0.6, zorder = 4)
axs10[5].fill_between(mcnary_time[window_size::window_shift], nonmarkov_slopes[3,:], nonmarkov_slopes[4,:] , color = 'green', alpha  =0.4, zorder = 4)

axs10[8].plot(mcnary_time[window_size::window_shift], nonmarkov_noise[0,:], lw=line_width, color = 'r', label = r' NBLE $\hat{\psi}$', zorder = 5)
axs10[8].fill_between(mcnary_time[window_size::window_shift], nonmarkov_noise[1,:], nonmarkov_noise[2,:] , color = 'green', alpha  =0.6, zorder = 4)
axs10[8].fill_between(mcnary_time[window_size::window_shift], nonmarkov_noise[3,:], nonmarkov_noise[4,:] , color = 'green', alpha  =0.4, zorder = 4)

axs10[5].xaxis.set_visible(False)

axs10[2].set_xlim(mcnary_time[0], mcnary_time[-1])
axs10[5].set_xlim(mcnary_time[0], mcnary_time[-1])

axs10[8].set_xlim(mcnary_time[0], mcnary_time[-1])
axs10[8].set_xlabel(r'Time $t$', fontsize = label_size)
axs10[8].xaxis.set_visible(True)



plt.savefig('Figure_S3.pdf')
plt.show()
