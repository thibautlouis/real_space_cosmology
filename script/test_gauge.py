import camb_class
import numpy as np
import pylab as plt
import dict_utils
import sys

d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])
params = d["params"]

scale_factor  = 10 ** np.linspace(d["log10a_ini"], d["log10a_final"], d["n_a"])
redshifts = 1 / scale_factor - 1

species = ["baryons", "cdm", "photons", "theta_baryons", "theta_neutrinos", "psi"]



k_data, evolution_sync = camb_class.get_tf_class(params, species, redshifts, kmax=100, gauge="syncronous")
k_data, evolution_newtonian = camb_class.get_tf_class(params, species, redshifts, kmax=100, gauge="newtonian")


for i, z in enumerate(redshifts):
    for j, specy in enumerate(species):

        plt.loglog()
        plt.title(f"z={z}, {specy}")
        
        plt.plot(k_data, np.abs(evolution_newtonian[:, i, j]), label="newt", linestyle="-")
        plt.plot(k_data, np.abs(evolution_sync[:, i, j]), label="sync", linestyle="--")

        plt.xlabel("k")
        plt.legend()
        plt.show()
