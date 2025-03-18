import numpy as np
from antiCPy.early_warnings import drift_slope as ds # Version 1.0.0 or smaller can be used.
import matplotlib.pyplot as plt

if __name__ == '__main__':
	generator_ID = '2'
	data = np.load('detrended_input_frequencies' + generator_ID + '.npy')
	time = np.arange(0,4000,0.05)

	plt.plot(time,data)
	plt.show()

	window_size = 9000
	window_shift = 100
	nsteps = 35000
	num_processes = 18

	NM_model = ds.NonMarkovEstimation(data,time, activate_time_scale_separation_prior=True, slow_process='X', 
							time_scale_separation_factor=2, max_likelihood_starting_guesses=np.array([0.,1.,0.,0.,0.,0.1]),
							prior_range = np.array([[100.,-100.],[100.,-100.],[100.,-100.],[100.,-100],[100.,0],[200.,0]]))

	n_slope_grid_points = 5000
	n_noise_grid_points = 5000
	noise_grid = np.linspace(0,1,n_noise_grid_points)
	slope_grid = np.linspace(-15, 0,n_slope_grid_points)
	OU_grid = np.linspace(0,2,5000)
	X_coupling_grid =np.linspace(0,10,5000)

	NM_model.fast_resilience_scan(window_size, window_shift, slope_grid, noise_grid, OU_grid, X_coupling_grid, nsteps = nsteps, 
									num_processes = num_processes, ignore_AC_error = False)

	np.save('slopes_K_increase3_steep' + generator_ID + '.npy', NM_model.slope_storage)
	np.save('noise_K_increase3_steep' + generator_ID + '.npy', NM_model.noise_level_storage)

	plt.plot(time[window_size::window_shift],NM_model.slope_storage[0,:])
	plt.fill_between(time[window_size::window_shift],NM_model.slope_storage[1,:],NM_model.slope_storage[2,:], color = 'orange', alpha = 0.6)
	plt.fill_between(time[window_size::window_shift],NM_model.slope_storage[3,:],NM_model.slope_storage[4,:], color = 'orange', alpha = 0.4)
	plt.savefig('slopes_K_increase3_steep' + generator_ID + '.png')
	plt.show()

	plt.plot(time[window_size::window_shift],NM_model.noise_level_storage[0,:])
	plt.fill_between(time[window_size::window_shift],NM_model.noise_level_storage[1,:],NM_model.noise_level_storage[2,:], color = 'orange', alpha = 0.6)
	plt.fill_between(time[window_size::window_shift],NM_model.noise_level_storage[3,:],NM_model.noise_level_storage[4,:], color = 'orange', alpha = 0.4)
	plt.savefig('noise_level_K_increase3_steep' + generator_ID + '.png')
	plt.show()