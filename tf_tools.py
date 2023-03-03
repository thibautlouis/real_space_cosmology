import numpy as np
import pylab as plt
from scipy.interpolate import InterpolatedUnivariateSpline


    
def select_and_interpolate_tf(output_data, fields, redshifts, cosmo_parameters, plot=True):

    tf_select = [
        {field: output_data[i][field] for field in fields + ["k (h/Mpc)"]}
        for i in range(len(redshifts))
    ]

    transfer_fcns = {field: [] for field in fields}
    for i, z in enumerate(redshifts):
    
        k_data = output_data[0]["k (h/Mpc)"] * cosmo_parameters["h"]  # in order to get k [1/Mpc]
        k_data_zero = np.concatenate(([0.0], k_data))
        for field in fields:
            data = tf_select[i][field]
            data_zero = np.concatenate(([data[0]], data))
           
            interpolated_func = InterpolatedUnivariateSpline(k_data_zero, data_zero)
            transfer_fcns[field].append(interpolated_func)
            
            if plot:
                plt.figure()
                plt.semilogx()
                plt.plot(k_data_zero, data_zero, label=field)
                plt.legend()
                plt.savefig(f"plot/tf_{z}_{field}.png")
                plt.clf()
                plt.close()

    return k_data_zero, transfer_fcns


def evolve_with_tf(k, tf, ft_ini, plot_tf = False):
    tf_ravel = tf(k.ravel())
    tf_2D = tf_ravel.reshape(k.shape)
    result = np.fft.fftshift(np.fft.ifft2(tf_2D * ft_ini).real)

    return result
    
    
