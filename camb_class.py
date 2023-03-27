import numpy as np
from classy import Class
from scipy.interpolate import InterpolatedUnivariateSpline
import camb
import camb.correlations
import hankl
from camb.symbolic import *
from IPython.display import display



def get_tf_camb(params, species, redshifts, kmax=100, kmin=10**-6, nk=4096):

    camb_convention = {}
    camb_convention["photons"] = "delta_photon"
    camb_convention["baryons"] = "delta_baryon"
    camb_convention["tot"] = "delta_tot"
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
    ks *= params["H0"] / 100

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
        "YHe": 0.245,
        "output": "mTk, vTk",
        "gauge": gauge,
        "evolver": "1",
        "P_k_max_h/Mpc": kmax,
        "k_per_decade_for_pk": 1000,
        "z_max_pk": np.max(redshifts),
    }
    
    class_convention = {}
    class_convention["tot"] = "d_tot"
    class_convention["photons"] = "d_g"
    class_convention["baryons"] = "d_b"
    class_convention["cdm"] = "d_cdm"
    class_convention["neutrinos"] = "d_ur"
    class_convention["psi"] = "psi"
    class_convention["phi"] = "phi"
    class_convention["theta_tot"] = "t_tot"
    class_convention["theta_photons"] = "t_g"
    class_convention["theta_baryons"] = "t_b"
    class_convention["theta_cdm"] = "t_cdm"
    class_convention["theta_neutrinos"] = "t_ur"

    species_delta_class = []
    species_v_class = []

    for specy in species:
        species_delta_class += [class_convention[specy]]
        if specy == "psi" or specy == "phi": continue
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
            if s == "psi" or s == "phi": continue
            evolution_v[:, i, j] = output_data[i][species_v_class[j]]

    
    return k, evolution_delta, evolution_v

        
def get_eta_and_rs(params, z):

    cosmo = Class()
    cosmo_parameters = {
          "h": params["H0"] / 100,
          "omega_b": params["ombh2"],
          "omega_cdm": params["omch2"],
          "Omega_k": params["omk"],
          "YHe": 0.245,
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


def get_cls(params, lmax=2500, accuracy=1):
    pars = camb.CAMBparams()
    pars_cosmo = {"ombh2": params["ombh2"],
                  "omch2": params["omch2"],
                  "omk": params["omk"],
                 }
                 
    pars.set_accuracy(AccuracyBoost=accuracy, lAccuracyBoost=accuracy)
    pars.set_cosmology(H0=params["H0"], **pars_cosmo)
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['unlensed_scalar']
    ls = np.arange(totCL.shape[0])
    TT, EE, BB, TE = totCL[:, 0], totCL[:, 1], totCL[:, 2], totCL[:, 3]

    return ls, TT, EE, BB, TE

def get_cl_and_pk_camb(params, lmax=2500, kmax=10):

    pars = camb.CAMBparams()
    pars_cosmo = {"ombh2": params["ombh2"],
                  "omch2": params["omch2"],
                  "omk": params["omk"],
                 }
                 
    accuracy = 3
    pars.set_accuracy(AccuracyBoost=accuracy, lAccuracyBoost=accuracy)
    pars.set_cosmology(H0=params["H0"], **pars_cosmo)
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_matter_power(redshifts=[0], kmax=kmax)
    pars.NonLinear = camb.model.NonLinear_none
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total']
    
    print(totCL.shape)
    ls = np.arange(totCL.shape[0])
    TT, EE, BB, TE = totCL[:, 0], totCL[:, 1], totCL[:, 2], totCL[:, 3]
    
    # also get the CMB correlation function
    theta = np.linspace(0.0001, 10, 10000)
    xvals = np.cos(theta * np.pi / 180)
    corr = camb.correlations.cl2corr(totCL, xvals, lmax)
    xi_cmb = corr[:,0] / corr[0,0]
    
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints = 1000)
    h = params["H0"] / 100
    k = kh * h
    
    # need to provide hankle with cst interval in logspace
    pk_int = InterpolatedUnivariateSpline(k, pk)
    kmin, kmax = np.min(k), np.max(k)
    ks = 10**np.linspace(np.log10(kmin), np.log10(kmax), 1024)
    
    r, xi = hankl.P2xi(ks, pk_int(ks), 0, n=0, lowring=False, ext=0, range=None, return_ext=False)
    xi = xi / xi[0]
    
    
    # and the large scale structure correlation function
    cl_lens = results.get_lens_potential_cls(lmax)
    l_lens = np.arange(2, cl_lens[:,0].size)
    phiphi = cl_lens[2:,0]
    
    out = {"ls": ls,
           "TT": TT,
           "EE": EE,
           "BB": BB,
           "TE": TE,
           "kh": kh,
           "pk": pk[0],
           "l_lens": l_lens,
           "cl_lens": phiphi,
           "theta": theta,
           "xi_cmb": xi_cmb,
           "r": r,
           "xi": xi}

    return out
    
def compute_pk_from_ini_and_TF(params, specy, redshift):

    k, evolution_delta, _ = get_tf_class(params, [specy], [redshift], kmax=10, gauge="synchronous")
    Tf = evolution_delta[:,0,0]
   # k, evolution_delta = get_tf_camb(params, [specy], [redshift], kmax=10, kmin=10**-4#)
    #import pylab as plt
    #plt.plot(k_class, evolution_delta_class[:,0,0])
    #plt.plot(k_camb, evolution_delta_camb[:,0,0])
    #plt.show()
    
    pk_init = params["As"] * (k / (0.05)) ** (params["ns"] - 1) * (2 * np.pi ** 2 / k ** 3) #initial condition given in term of R
    pk = pk_init * Tf ** 2
    h = params["H0"]/100
    kh, pk, pk_init = k/h, pk * h ** 3, pk_init * h ** 3 # make k in h^{-1}Mpc and Pk in h^{3} Mpc
    
    return kh, pk_init, Tf,  pk

def get_thermo_class(params):

    cosmo = Class()
    cosmo_parameters = {
          "h": params["H0"] / 100,
          "omega_b": params["ombh2"],
          "omega_cdm": params["omch2"],
          "Omega_k": params["omk"],
          "YHe": 0.245,
          "thermodynamics_verbose": 1}
    cosmo.set(cosmo_parameters)

    cosmo.compute()
    derived = cosmo.get_current_derived_parameters(["tau_rec", "conformal_age"])
    thermo = cosmo.get_thermodynamics()
    print(thermo.keys())


def decompose_cls_camb(params, lmax=2500, accuracy=1):

    pars = camb.CAMBparams()
    pars_cosmo = {"ombh2": params["ombh2"],
                  "omch2": params["omch2"],
                  "omk": params["omk"],
                 }

    pars.set_accuracy(AccuracyBoost=accuracy, lAccuracyBoost=accuracy)
    pars.set_cosmology(H0=params["H0"], **pars_cosmo)
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    monopole_source, ISW, doppler, quadrupole_source = get_scalar_temperature_sources()

    early_ISW = sympy.Piecewise( (ISW, 1/a - 1 > 30),(0, True))  #redshift > 30
    late_ISW = ISW - early_ISW

    names = ["dg", "psi", "mon", "ISW", "eISW", "LISW", "dop", "Q"]
    
    dg = make_frame_invariant(Delta_g / 4 * visibility, frame='Newtonian')
    psi = make_frame_invariant(Psi_N * visibility, frame='Newtonian')


    pars.set_custom_scalar_sources([dg, psi, monopole_source, ISW,early_ISW, late_ISW, doppler, quadrupole_source], source_names=names)
    data= camb.get_results(pars)
    dict = data.get_cmb_unlensed_scalar_array_dict(CMB_unit="muK")
    
    
    # also decompose the monopole into its two contribution
    display('Temperature monopole source in Newtonian gauge', newtonian_gauge(monopole_source))
    display('Doppler source in Newtonian gauge', newtonian_gauge(doppler))
    display('ISW source in Newtonian gauge', newtonian_gauge(ISW))


    return dict

