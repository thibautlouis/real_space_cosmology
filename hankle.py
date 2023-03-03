import numpy as np
import os, sys
from scipy.interpolate import InterpolatedUnivariateSpline
import hankl
import initial_conditions
import pickle
import camb_class
import dict_utils


d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])


anim_dir = "movies"
os.makedirs(anim_dir, exist_ok=True)

params = d["params"]
gauge = d["gauge"]
divide_by_k2 = d["divide_by_k2"]
sigma = d["sigma_pert"]
species = d["species"]

if divide_by_k2 == True:
    name = "_divided_by_k2"
else:
    name = ""

scale_factor  = 10 ** np.linspace(d["log10a_ini"], d["log10a_final"], d["n_a"])
redshifts = 1 / scale_factor - 1

eta, rs = camb_class.get_eta_and_rs(params, redshifts)

k_data, evolution_delta, evolution_v = camb_class.get_tf_class(params, species, redshifts, kmax=d["kmax"], gauge=gauge)
delta_k = initial_conditions.generate_gaussian_radial_fourier(k_data, sigma)

k_min, k_max = 10**-4, np.max(k_data)
my_k = np.logspace(np.log10(k_min), np.log10(k_max), 10 ** 4 )
nsteps, nspecies = len(redshifts),  len(species)


delta_all = np.zeros((10000, nsteps, nspecies))
v_all = np.zeros((10000, nsteps, nspecies))


for i, z in enumerate(redshifts):
    for j, specy in enumerate(species):
    
        tf_delta_interp = InterpolatedUnivariateSpline(k_data, evolution_delta[:, i, j])
        tf_v_interp = InterpolatedUnivariateSpline(k_data, evolution_v[:, i, j])

        delta_k_now =  delta_k(my_k) * tf_delta_interp(my_k)
        v_k_now = delta_k(my_k) * tf_v_interp(my_k)

        if divide_by_k2 == True:
            delta_k_now /= my_k ** 2
            v_k_now /= my_k ** 2
            
        v_k_now *= -my_k #theta = grad. v

        r, delta = hankl.P2xi(my_k, delta_k_now, l=0, ext=1)
        r, v = hankl.P2xi(my_k, v_k_now, l=0, ext=1)

        delta_all[:, i, j]  = delta.real
        v_all[:, i, j]  = v.real

output = {"r": r,
          "delta_all": delta_all,
          "v_all": v_all,
          "species": species,
          "redshifts": redshifts,
          "conformal_time": eta,
          "sound_horizon": rs}
      
output_file = f"{anim_dir}/delta_outputs_{gauge}{name}.pkl"
def save(output, filename):
    pickle.dump(output, open(filename, "wb"))
save(output, output_file)





