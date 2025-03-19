import numpy as np
from antiCPy.early_warnings import drift_slope as ds # Version 1.0.0 or smaller can be used.
import matplotlib.pyplot as plt

if __name__ == '__main__':
	generator_ID = '2'
	data = np.load('detrended_input_frequencies' + generator_ID + '.npy')
	time = np.arange(0,2000,0.05)

	plt.plot(time,data)
	plt.show()

	window_size = 12000
	window_shift = 100
	nsteps = 35000
	num_processes = 10

	langevin_model = ds.LangevinEstimation(data,time)

	langevin_model.fast_resilience_scan(window_size = window_size, window_shift = window_shift, nsteps = nsteps, 
											num_processes = num_processes, slope_grid = np.linspace(-10, 1, 5000), 
											noise_grid = np.linspace(0,3,5000), ignore_AC_error = False)

