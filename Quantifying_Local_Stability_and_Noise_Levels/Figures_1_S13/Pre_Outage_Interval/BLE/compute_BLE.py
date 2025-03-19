import numpy as np
import time as tt
from antiCPy.early_warnings import drift_slope as ds # Version 1.0.0 can be used. Older versions work as well.

def main():
	# load data
	data_npz = np.load('../BV_data_GKF5.npz')
	data = data_npz['xx']
	time = data_npz['tt']

	samples = data.size
	dt = round(time[1] - time[0],3)

	# define parameters for analysis in rolling windows
	window_size = 1000
	window_shift = 100

	nsteps = 5000 # number of MCMC steps to take

	# Define grids for PDF read-out
	n_slope_samples = 200000
	n_example_grid_points = 2000
	n_slope_grid_points = 2500
	n_noise_grid_points = 5000
	noise_grid = np.linspace(0,0.03,n_noise_grid_points)
	slope_grid = np.linspace(-8, 3,n_slope_grid_points)
	example_grid = np.linspace(-30,30, n_example_grid_points)

	num_processes = 10 # number of parallel workers

	data_model = ds.LangevinEstimation(data, time)

	alpha = tt.time()
	data_model.fast_resilience_scan(window_size, window_shift, slope_grid, noise_grid, nsteps = nsteps, 
									n_slope_samples = n_slope_samples, num_processes = num_processes) # For the rest, default options are used. 
	beta = tt.time()
	
	print('Time elpapsed using ' + str(num_processes) + ' CPU cores: ' + str(beta-alpha) + 'seconds.')
	print('Computations completed.')

if __name__ == '__main__':
	main()
