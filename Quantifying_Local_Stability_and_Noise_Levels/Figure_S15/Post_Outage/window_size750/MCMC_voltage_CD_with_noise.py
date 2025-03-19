import numpy as np
from antiCPy.early_warnings import drift_slope as ds # Version 1.0.0 or smaller can be used.
import matplotlib.pyplot as plt

if __name__ == '__main__':
	data = np.load('restoration_data.npy')
	time = np.load('reconstructed_time_step.npy')


	samples = data.size
	dt = round(time[1] - time[0],3)
	print(data.size)
	print('dt: '+ str(dt))

	plt.plot(time,data)
	plt.show()

	window_size = 750
	window_shift = 100
	nsteps = 30000 
	num_processes = 6

	NM_model = ds.NonMarkovEstimation(data, time, activate_time_scale_separation_prior=True, slow_process='X', 
							time_scale_separation_factor=2, max_likelihood_starting_guesses=np.array([0.,1.,0.,0.,0.,0.1]),
							prior_range = np.array([[100.,-100.],[100.,-100.],[100.,-100.],[100.,-100],[100.,0],[200.,0]]))

	n_slope_grid_points = 5000
	n_noise_grid_points = 5000
	noise_grid = np.linspace(0,0.005,n_noise_grid_points)
	slope_grid = np.linspace(-0.05, 0.01,n_slope_grid_points)
	OU_grid = np.linspace(0,190,5000)
	X_coupling_grid =np.linspace(0,0.1,5000)

	NM_model.fast_resilience_scan(window_size, window_shift, slope_grid, noise_grid, OU_grid, X_coupling_grid, 
										cred_percentiles = np.array([16, 1]), nsteps = nsteps,
										print_progress = True, ignore_AC_error = False, thinning_by = 60, print_details = False,
										slope_save_name = 'default_slopes_nonmarkov', print_AC_tau = True,
										noise_level_save_name = 'default_noise_nonmarkov', save = True, 
										num_processes = num_processes)