import numpy as np
from antiCPy.early_warnings import drift_slope as ds # Version 1.0.0 or smaller can be used.
import matplotlib.pyplot as plt

if __name__ == '__main__':
	data_npz = np.load('BV_data_GKF5.npz')
	data = data_npz['xx']
	time = data_npz['tt']

	samples = data.size
	dt = round(time[1] - time[0],3)
	print(data.size)
	print('dt: '+ str(dt))

	plt.plot(time,data)
	plt.show()

	window_size = 750
	window_shift = 100
	nsteps = 15000
	num_processes = 5

	NM_model = ds.NonMarkovEstimation(data, time, activate_time_scale_separation_prior=True, slow_process='X', 
							time_scale_separation_factor=2, max_likelihood_starting_guesses=np.array([0.,1.,0.,0.,0.,0.1]),
							prior_range = np.array([[50.,-50.],[50.,-50.],[50.,-50.],[50.,-50],[50.,0],[50.,0]]))

	slope_grid = np.linspace(-10.,1,3000)
	noise_grid = np.linspace(0.,0.04,2000)
	OU_grid = np.linspace(0,1,2000)
	X_coupling_grid =np.linspace(0,0.2,2000)

	NM_model.fast_resilience_scan(window_size, window_shift, slope_grid, noise_grid, OU_grid, X_coupling_grid, 
										cred_percentiles = np.array([16, 1]), nsteps = nsteps,
										print_progress = True, ignore_AC_error = False, thinning_by = 60, print_details = False,
										slope_save_name = 'NM_slopes_BV', print_AC_tau = True,
										noise_level_save_name = 'NM_noise_BV', save = True, 
										num_processes = num_processes)
