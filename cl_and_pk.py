import numpy as np
import pylab as plt
import os, sys
import dict_utils
import camb_class



d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])

spec_dir = "spectra"
os.makedirs(spec_dir, exist_ok=True)



run_name = "lcdm"
params = d[f"params_{run_name}"]

kh, pk_init, tf,  pk = camb_class.compute_pk_from_ini_and_TF(params, "tot", 0)

plt.loglog()
plt.plot(kh, pk_init, label="initial power spectrum")
plt.plot(kh, tf**2, label="transfer function (squared)")
plt.plot(kh, pk, '--', label="matter power spectrum ")
plt.xlabel(r"k [$h Mpc^{-1}$]", fontsize=15)
plt.legend(fontsize=12)
plt.savefig(f"{spec_dir}/fromTF_to_Pk.png", bbox_inches="tight")
plt.clf()
plt.close()


run_names = ["lcdm", "no_dm_high_baryons"]
spec_dict = {}
for run_name in run_names:
    params = d[f"params_{run_name}"]
    spec_dict[run_name] = camb_class.get_cl_and_pk_camb(params, kmax=10)
    
    

label_list = ["lcdm", "baryons only"]

plt.figure(figsize=(12,8))
plt.loglog()
for run_name, label in zip(run_names, label_list):
    my_dict  = spec_dict[run_name]
    plt.plot(my_dict["kh"], my_dict["pk"], label=label)
    
plt.legend(fontsize=15)
plt.ylim(1,5 * 10**4)
plt.xlabel(r"k [$h Mpc^{-1}$]", fontsize=20)
plt.ylabel(r"P(k) $[(h^{-1} Mpc)^{3}]$", fontsize=20)
plt.savefig(f"{spec_dir}/Pk_different_cosmo.png", bbox_inches="tight")
plt.clf()
plt.close()

plt.figure(figsize=(12,8))
plt.loglog()
for run_name, label in zip(run_names, label_list):
    my_dict  = spec_dict[run_name]
    plt.plot(my_dict["ls"], my_dict["TT"], label=label)
    
plt.legend(fontsize=15)
plt.savefig(f"{spec_dir}/Cl_different_cosmo.png", bbox_inches="tight")
plt.clf()
plt.close()


plt.figure(figsize=(12,8))
plt.loglog()
for run_name, label in zip(run_names, label_list):
    my_dict  = spec_dict[run_name]
    plt.plot(my_dict["theta"], my_dict["xi_cmb"], label=label)
    
plt.legend(fontsize=15)
plt.savefig(f"{spec_dir}/xi_cmb_different_cosmo.png", bbox_inches="tight")
plt.clf()
plt.close()

plt.figure(figsize=(12,8))
count = 1
for run_name, label in zip(run_names, label_list):
    my_dict  = spec_dict[run_name]
    id = np.where(my_dict["r"] < 600)
    plt.subplot(2, 1, count)
    plt.plot(my_dict["r"][id], my_dict["xi"][id], label=label)
    count += 1
    
plt.legend(fontsize=15)
plt.savefig(f"{spec_dir}/xi_different_cosmo.png", bbox_inches="tight")
plt.clf()
plt.close()


colors = ["blue", "orange"]
plt.figure(figsize=(12,8))
count = 1
for col, run_name, label in zip(colors, run_names, label_list):
    my_dict  = spec_dict[run_name]
    id = np.where(my_dict["r"] < 600)
    plt.subplot(2, 1, count)
    plt.plot(my_dict["r"][id], my_dict["xi"][id]*my_dict["r"][id]**2, label=label, color=col)
    plt.legend(fontsize=15)
    if count == 2:
        plt.xlabel(r"Mpc", fontsize=20)
    plt.ylabel(r"$r^{2} \xi(r) $", fontsize=20)

    count += 1
    
plt.savefig(f"{spec_dir}/xi_different_cosmo_rsquared.png", bbox_inches="tight")
plt.clf()
plt.close()
