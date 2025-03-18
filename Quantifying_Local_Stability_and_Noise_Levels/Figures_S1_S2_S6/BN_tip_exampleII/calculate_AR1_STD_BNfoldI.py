import numpy as np
import pylab
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from statsmodels.tsa.stattools import acf
import scipy.stats as sts

data = np.load('detrended_B_tip_fold_cutted.npy')
samples = 30000

r_array = np.linspace(-15, 5, samples)
for i in range(samples):
	if r_array[i] >= -0.5:
		transition_index = i
		break

start_time = 0.
end_time = 2000.

dt = (end_time - start_time) / samples
time = np.linspace(start_time, end_time, samples)
time = time[:transition_index]

print(time)

window_size = 2000
window_shift = 150

loop_range = np.arange(0, data.size - window_size, window_shift)
loop_range_size = loop_range.size
show_indicators = True

AR1_array = np.zeros(loop_range_size)
std_array = np.zeros(loop_range_size)

i = 0

for k in range(loop_range_size):
	data_window = np.roll(data, shift = - loop_range[k])[:window_size]
	time_window = np.roll(time, shift = - loop_range[k])[:window_size]
	AR1_array[k] = acf(data_window, adjusted = False, nlags = 1)[1]
	std_array[k] = np.std(data_window, ddof = 0)
	print(r'progress: ' + str(k+1) + r'/' + str(loop_range_size))

np.save('AR1_B_tip_fold.npy', AR1_array)
np.save('std_B_tip_fold.npy', std_array)

if show_indicators:
	fig, axs = plt.subplots(2,1)
	axs[0].plot(time, data)
	axs[1].plot(time[window_size + loop_range], AR1_array)
	axs[1].plot(time[window_size + loop_range], std_array)
	plt.show()