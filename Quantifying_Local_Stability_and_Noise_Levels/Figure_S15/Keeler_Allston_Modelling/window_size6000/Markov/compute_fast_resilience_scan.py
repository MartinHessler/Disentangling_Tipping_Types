import numpy as np
from antiCPy.early_warnings import drift_slope as ds # Version 1.0.0 or smaller can be used.
import matplotlib.pyplot as plt

if __name__ == '__main__':
	generator_ID = '2'
	data = np.load('detrended_input_frequencies' + generator_ID + '.npy')
	time = np.arange(0,2000,0.05)

	plt.plot(time,data)
	plt.show()

	window_size = 6000
	window_shift = 100
	nsteps = 35000
	num_processes = 10

	langevin_model = ds.LangevinEstimation(data,time)

	langevin_model.fast_resilience_scan(window_size = window_size, window_shift = window_shift, nsteps = nsteps, 
											num_processes = num_processes, slope_grid = np.linspace(-10, 1, 5000), 
											noise_grid = np.linspace(0,3,5000), ignore_AC_error = False)

	np.save('slopes_K_increase3_steep' + generator_ID + '.npy', langevin_model.slope_storage)
	np.save('noise_K_increase3_steep' + generator_ID + '.npy', langevin_model.noise_level_storage)

	plt.plot(time[window_size::window_shift],langevin_model.slope_storage[0,:])
	plt.fill_between(time[window_size::window_shift],langevin_model.slope_storage[1,:],langevin_model.slope_storage[2,:], color = 'orange', alpha = 0.6)
	plt.fill_between(time[window_size::window_shift],langevin_model.slope_storage[3,:],langevin_model.slope_storage[4,:], color = 'orange', alpha = 0.4)
	plt.savefig('slopes_K_increase3_steep' + generator_ID + '.png')
	plt.show()

	plt.plot(time[window_size::window_shift],langevin_model.noise_level_storage[0,:])
	plt.fill_between(time[window_size::window_shift],langevin_model.noise_level_storage[1,:],langevin_model.noise_level_storage[2,:], color = 'orange', alpha = 0.6)
	plt.fill_between(time[window_size::window_shift],langevin_model.noise_level_storage[3,:],langevin_model.noise_level_storage[4,:], color = 'orange', alpha = 0.4)
	plt.savefig('noise_level_K_increase3_steep' + generator_ID + '.png')
	plt.show()