import numpy as np
import hankl
import pylab as plt
from classy import Class
import initial_conditions
import plot_utils
import tf_tools
import os
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.animation import FuncAnimation
import pickle
import time
import camb





def get_tf_camb(params, species, redshifts, kmax=100, kmin=10**-6, nk=1024):

    camb_convention = {}
    camb_convention["photons"] = "delta_photon"
    camb_convention["baryons"] = "delta_baryon"
    camb_convention["cdm"] = "delta_cdm"
    camb_convention["neutrinos"] = "delta_neutrino"

    species_camb = []
    for specy in species:
        species_camb += [camb_convention[specy]]
    
    pars = camb.CAMBparams(scalar_initial_condition="initial_adiabatic")
    pars_cosmo = {"ombh2": params["ombh2"],
                  "omch2": params["omch2"],
                  "omk": params["omk"],
                 }
    accuracy = 3
    pars.set_cosmology(H0=params["H0"], **pars_cosmo)
    pars.set_accuracy(AccuracyBoost=accuracy, lAccuracyBoost=accuracy)
    data = camb.get_transfer_functions(pars)

    ks = 10 ** np.linspace(np.log10(kmin), np.log10(kmax), nk)
    ks *= H0 / 100

    evolution = data.get_redshift_evolution(ks, redshifts, species_camb, lAccuracyBoost=accuracy)
    # return an array nk, nz, nspecies
    
    return ks, evolution


def get_tf_class(params, species, redshifts, kmax=100, gauge="synchronous"):
    cosmo = Class()
    cosmo_parameters = {
        "h": params["H0"] / 100,
        "omega_b": params["ombh2"],
        "omega_cdm": params["omch2"],
        "Omega_k": params["omk"],
        "output": "mTk, vTk",
        "gauge": gauge,
        "evolver": "1",
        "P_k_max_h/Mpc": kmax,
        "k_per_decade_for_pk": 100,
        "z_max_pk": np.max(redshifts),
    }
    
    class_convention = {}
    class_convention["photons"] = "d_g"
    class_convention["baryons"] = "d_b"
    class_convention["cdm"] = "d_cdm"
    class_convention["neutrinos"] = "d_ur"
    class_convention["psi"] = "psi"
    class_convention["theta_photons"] = "t_g"
    class_convention["theta_baryons"] = "t_b"
    class_convention["theta_cdm"] = "t_cdm"
    class_convention["theta_neutrinos"] = "t_ur"

    species_delta_class = []
    species_v_class = []

    for specy in species:
        species_delta_class += [class_convention[specy]]
        species_v_class += [class_convention["theta_"+ specy]]

    cosmo.set(cosmo_parameters)
    cosmo.compute()
    output_data = [cosmo.get_transfer(z) for z in redshifts]
    k = output_data[0]["k (h/Mpc)"]
    k *= params["H0"] / 100
    
    nk, nz, nspecies = len(k), len(redshifts), len(species)
    
    # make it in the camb format
    
    evolution_delta = np.zeros((nk, nz, nspecies))
    evolution_v = np.zeros((nk, nz, nspecies))
    
    for  i, z in enumerate(redshifts):
        for j, s in enumerate(species):
            evolution_delta[:, i, j] = output_data[i][species_delta_class[j]]
            if (s == "cdm") & (gauge == "synchronous"): continue # cdm has zero velocity in synchronous gauge
            evolution_v[:, i, j] = output_data[i][species_v_class[j]]

    
    return k, evolution_delta, evolution_v

        





def get_eta_and_rs(params, z):

    cosmo = Class()
    cosmo_parameters = {
          "h": params["H0"] / 100,
          "omega_b": params["ombh2"],
          "omega_cdm": params["omch2"],
          "Omega_k": params["omk"],
      }
      
    cosmo.set(cosmo_parameters)
    cosmo.compute()
    background = cosmo.get_background()
    z_class = background["z"]
    eta = background["conf. time [Mpc]"]
    sound_horizon = background["comov.snd.hrz."]
    
    eta_interp = InterpolatedUnivariateSpline(z_class[::-1], eta[::-1])
    sound_horizon_interp = InterpolatedUnivariateSpline(z_class[::-1], sound_horizon[::-1])
    return eta_interp(z), sound_horizon_interp(z)

