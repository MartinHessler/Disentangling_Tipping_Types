import numpy as np
from antiCPy.early_warnings import drift_slope as ds # Version 1.0.0 can be used. Older versions work as well.
import matplotlib.pyplot as plt
import time as tt

if __name__ == '__main__':
	data = np.load('../restoration_data_with_outliers.npy')
	time = np.load('../reconstructed_time_step.npy')


	samples = data.size
	dt = round(time[1] - time[0],3)

	# define parameters for analysis in rolling windows
	window_size = 1000
	window_shift = 100

	nsteps = 200000 # number of MCMC steps to take
	num_processes = 12 # number of parallel workers

	NM_model = ds.NonMarkovEstimation(data, time, activate_time_scale_separation_prior=True, slow_process='X', 
							time_scale_separation_factor=2, max_likelihood_starting_guesses=np.array([0.,1.,0.,0.,0.,0.1]),
							prior_range = np.array([[100.,-100.],[100.,-100.],[100.,-100.],[100.,-100],[100.,0],[200.,0]]))

	n_slope_grid_points = 5000
	n_noise_grid_points = 5000
	noise_grid = np.linspace(0,0.005,n_noise_grid_points)
	slope_grid = np.linspace(-0.05, 0.01,n_slope_grid_points)
	OU_grid = np.linspace(0,190,5000)
	X_coupling_grid =np.linspace(0,0.1,5000)

	alpha = tt.time()
	NM_model.fast_resilience_scan(window_size, window_shift, slope_grid, noise_grid, OU_grid, X_coupling_grid, 
										nsteps = nsteps, print_progress = True, num_processes = num_processes,
										slope_save_name='slopes_without_outlier_correction',
										noise_level_save_name='noise_without_outlier_correction',
										OU_param_save_name='OUparam_without_outlier_correction',
										X_coupling_save_name='Xcoupling_without_outlier_correction') # For the rest, default options are used. 

	beta = tt.time()

	print('Time elpapsed using ' + str(num_processes) + ' CPU cores: ' + str(beta-alpha) + 'seconds.')
	print('Computations completed.')

if __name__ == '__main__':
	main()
