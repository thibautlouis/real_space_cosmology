import numpy as np
from classy import Class
import initial_conditions
import plot_utils
import tf_tools
import os
from scipy.interpolate import InterpolatedUnivariateSpline


result_dir = "plot"
os.makedirs(result_dir, exist_ok=True)

redshifts = [10**6, 100000, 10000, 3000, 2000, 1100]

points = 3000 #number of point along each dimension of the box
size = 1000  #size of the box in MPC
sigma = 10

cosmo_parameters = {
    "h": 0.67556,
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "Omega_k": 0,
    "N_ur": 3.046,
    "w0_fld": -1,
    "wa_fld": 0,
    "Omega_Lambda": 0,
    "YHe": 0.25,
    "output": "mTk, vTk",
    "gauge": "newtonian",
    "evolver": "1",
    "P_k_max_h/Mpc": 100,
    "k_per_decade_for_pk": 200,
    "z_max_pk": np.max(redshifts),
}

cosmo = Class()
cosmo.set(cosmo_parameters)
cosmo.compute()
background = cosmo.get_background()


output_data = [cosmo.get_transfer(z) for z in redshifts]



all = ["psi", "d_b", "d_cdm", "d_g", "t_b", "t_cdm", "t_g"]

# start with the curvature perturbation, all other quantities are related to this one
x, y, psi_ini, kx, ky, k, ft_psi_ini = initial_conditions.generate_gaussian_curvature_2d(sigma=sigma, size=size, points=points)
_, transfer_fcns = tf_tools.select_and_interpolate_tf(output_data, all, redshifts, cosmo_parameters)


psi = {}
for i, z in enumerate(redshifts):
    psi[z] = tf_tools.evolve_with_tf(k, transfer_fcns["psi"][i], ft_psi_ini)
    plot_utils.plot_density(x, y, z, psi[z], "psi", size, result_dir)


v_ini, ft_v_ini = initial_conditions.from_curvature_to_v_init(ft_psi_ini, k)

species = ["baryons", "dark_matter", "photon"]

density, theta = {}, {}

density["baryons"] = "d_b"
density["dark_matter"] = "d_cdm"
density["photon"] = "d_g"

theta["baryons"] = "t_b"
theta["dark_matter"] = "t_cdm"
theta["photon"] = "t_g"

for specy in species:

    delt, thet = density[specy], theta[specy]

    for i, z in enumerate(redshifts):

        delta_z = tf_tools.evolve_with_tf(k, transfer_fcns[delt][i], ft_psi_ini)
        plot_utils.plot_density(x, y,  z, delta_z, specy, size, result_dir, radial_profile=True)
        
        #v_z = tf_tools.evolve_with_tf(k, transfer_fcns[thet][i], ft_v_ini)
        #plot_utils.plot_density_and_velocity(x,y, z, delta_z, v_z, specy, result_dir)

        if specy == "photon":
            plot_utils.plot_density(x, y,  z, 1 / 4 * delta_z + psi[z], "Sachs Wolfe", size, result_dir, radial_profile=True)

