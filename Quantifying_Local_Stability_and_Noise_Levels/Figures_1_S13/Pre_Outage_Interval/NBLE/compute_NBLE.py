import numpy as np
from antiCPy.early_warnings import drift_slope as ds # Version 1.0.0 can be used. Older versions work as well.
import matplotlib.pyplot as plt
import time as tt

if __name__ == '__main__':
	data_npz = np.load('../BV_data_GKF5.npz')
	data = data_npz['xx']
	time = data_npz['tt']

	samples = data.size
	dt = round(time[1] - time[0],3)

	# define parameters for analysis in rolling windows
	window_size = 1000 
	window_shift = 100

	nsteps = 15000 # number of MCMC steps to take
	num_processes = 12 # number of parallel workers


	slope_grid = np.linspace(-10.,1,3000)
	noise_grid = np.linspace(0.,0.04,2000)
	OU_grid = np.linspace(0,1,2000)
	X_coupling_grid =np.linspace(0,0.2,2000)

	NM_model = ds.NonMarkovEstimation(data, time, activate_time_scale_separation_prior=True, slow_process='X', 
							time_scale_separation_factor=2, max_likelihood_starting_guesses=np.array([0.,1.,0.,0.,0.,0.1])) # For the rest, default options are used. 


	alpha = tt.time()
	NM_model.fast_resilience_scan(window_size, window_shift, slope_grid, noise_grid, OU_grid, X_coupling_grid, 
										nsteps = nsteps, print_progress = True, num_processes = num_processes)

	beta = tt.time()

	print('Time elpapsed using ' + str(num_processes) + ' CPU cores: ' + str(beta-alpha) + 'seconds.')
	print('Computations completed.')

if __name__ == '__main__':
	main()
