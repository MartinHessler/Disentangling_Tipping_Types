import numpy as np
import time as tt
from antiCPy.early_warnings import drift_slope as ds  # Version 1.0.0 can be used. Older versions work as well.

def main():
	start_time = 0.
	end_time = 2000.
	samples = 30000
	dt = (end_time - start_time) / samples
	time = np.arange(start_time, end_time, dt)

	data = np.load('../ts_x_h_corr.npy')

	# # define parameters for analysis in rolling windows
	window_size = 2000
	window_shift = 100

	nsteps = 10000 # number of MCMC steps to take
	num_processes = 12

	n_slope_grid_points = 1000
	n_noise_grid_points = 10000
	slope_grid = np.linspace(-20,2,n_slope_grid_points)
	noise_grid = np.linspace(0,0.1, n_noise_grid_points)
	
	data_model = ds.LangevinEstimation(data, time)
	alpha = tt.time()
	data_model.fast_resilience_scan(window_size, window_shift, slope_grid, noise_grid, 
										nsteps = nsteps, num_processes = num_processes, 
										detrending_per_window='linear') # For the rest, default options are used.
	beta = tt.time()

	print('Time elpapsed using ' + str(num_processes) + ' CPU cores: ' + str(beta-alpha) + 'seconds.')
	print('Computations completed.')

if __name__ == '__main__':
	main()
