import numpy as np
from antiCPy.early_warnings import drift_slope as ds # Version 1.0.0 can be used. Older versions work as well.
import matplotlib.pyplot as plt
import time as tt

if __name__ == '__main__':
	generator_ID = '1'
	data = np.load('../detrended_input_frequencies' + generator_ID + '.npy')
	time = np.linspace(0,2000,4000000)
	time = time[0::100]

	plt.plot(time,data)
	plt.show()
	# define parameters for analysis in rolling windows
	window_size = 8000
	window_shift = 100

	nsteps = 20000 # number of MCMC steps to take
	num_processes = 10 # number of parallel workers

	NM_model = ds.NonMarkovEstimation(data,time, activate_time_scale_separation_prior=True, slow_process='X', 
							time_scale_separation_factor=2, max_likelihood_starting_guesses=np.array([0.,1.,0.,0.,0.,0.1]),
							prior_range = np.array([[100.,-100.],[100.,-100.],[100.,-100.],[100.,-100],[100.,0],[200.,0]]))

	n_slope_grid_points = 5000
	n_noise_grid_points = 5000
	noise_grid = np.linspace(0,3,n_noise_grid_points)
	slope_grid = np.linspace(-10, 1,n_slope_grid_points)
	OU_grid = np.linspace(0,5,10000)
	X_coupling_grid =np.linspace(0,20,10000)

	alpha = tt.time()
	NM_model.fast_resilience_scan(window_size, window_shift, slope_grid, noise_grid, OU_grid, X_coupling_grid, 
									nsteps = nsteps, num_processes = num_processes, ignore_AC_error = False, 
									slope_save_name = 'slopes_consumer_noise' + generator_ID + '.npy',
									noise_save_name = 'noise_consumer_noise' + generator_ID + '.npy'
									) # For the rest, default options are used.
	beta = tt.time()

	print('Time elpapsed using ' + str(num_processes) + ' CPU cores: ' + str(beta-alpha) + 'seconds.')
	print('Computations completed.')