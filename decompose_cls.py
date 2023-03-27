import numpy as np
import pylab as plt
import os, sys
import dict_utils
import camb_class
from pixell import curvedsky
from pspy import so_map, sph_tools



d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])

comp_dir = "components"
os.makedirs(comp_dir, exist_ok=True)

run_names = ["lcdm", "no_dm_high_baryons","no_dm"]
for run_name in run_names:
    params = d[f"params_{run_name}"]
    dict = camb_class.decompose_cls_camb(params)


    ls =np.arange(dict["TxT"].shape[0])
    lmax = len(ls)

    plt.figure(figsize=(8, 5))
    plt.loglog(ls, dict["monxmon"], color="C0", label="Monopole")
    plt.loglog(ls, dict["LISWxLISW"], color="C1", label="late ISW")
    plt.loglog(ls, dict["eISWxeISW"], ls="--", color="C1", label="early ISW")
    plt.loglog(ls, dict["dopxdop"], color="C2", label="Doppler")
    #plt.loglog(ls, dict["QxQ"], color="C3")
    plt.loglog(ls, dict["TxT"], lw=2, color="k", label="All")
    plt.xlabel("$\ell$", fontsize=12)
    plt.ylabel(r"$\ell(\ell+1)C_\ell/2\pi\quad [\mu {\rm K}^2]$", fontsize=12)
    plt.xlim(2, ls[-1])
    plt.ylim(1,10**4)
    plt.legend()
    plt.savefig(f"{comp_dir}/{run_name}_all_comp.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogx(ls, dict["psixpsi"], color="gray", label=r"$\psi \times \psi$")
    plt.semilogx(ls, dict["dgxdg"], color="red", label=r"$\Theta_{0} \times \Theta_{0}$")
    plt.semilogx(ls, 2 * dict["psixdg"], color="green", label=r"$ 2 \psi \times \Theta_{0} $")
    plt.semilogx(ls, dict["monxmon"], color="blue", label=r"$(\Theta_{0} + \psi) \times (\Theta_{0} + \psi)$")
    plt.legend(fontsize=12)
    plt.xlabel("$\ell$", fontsize=15)
    plt.ylabel(r"$\ell(\ell+1)C_\ell/2\pi\quad [\mu {\rm K}^2]$", fontsize=15)
    plt.xlim(2, ls[-1])
    plt.savefig(f"{comp_dir}/{run_name}_monopole.png", bbox_inches="tight")
    plt.clf()
    plt.close()


    all_comp = [ "dg", "psi", "dop", "ISW"] #neglecting Q for simplicity

    var_name = {}
    var_name["psi"] = "$\psi$"
    var_name["dg"] = "$ \Theta_{0}$"
    var_name["dop"] = "doppler"
    var_name["ISW"] = "ISW"

    ncomp = len(all_comp)
    comp_mat = np.zeros((ncomp, ncomp, lmax))
    cl = 0
    plt.figure(figsize=(16, 12))

    for i, c1 in enumerate(all_comp):
        for j, c2 in enumerate(all_comp):

            comp_mat[i,j, 2:lmax] = dict[f"{c1}x{c2}"][2:lmax] * 2 * np.pi / (ls[2:lmax] * (ls[2:lmax] + 1 ))
            if i >= j:
                if i != j:
                    plt.semilogx(ls, 2*dict[f"{c1}x{c2}"], label=r"2 %s $\times$ %s" % (var_name[c1], var_name[c2]))
                else:
                    plt.semilogx(ls, dict[f"{c1}x{c2}"], label=r"%s $\times$ %s" % (var_name[c1], var_name[c2]))

            cl += dict[f"{c1}x{c2}"]

    plt.semilogx(ls, cl, color='k', linestyle="--", label="combined")
    plt.legend(fontsize=16)
    plt.xlim(2, ls[-1])
    plt.ylim(-12000, 12000)
    plt.xlabel("$\ell$", fontsize=20)
    plt.ylabel(r"$\ell(\ell+1)C_\ell/2\pi\quad [\mu {\rm K}^2]$", fontsize=20)
    plt.savefig(f"{comp_dir}/{run_name}_real_all_comp.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    comp_alms = curvedsky.rand_alm(comp_mat, lmax=lmax, seed=0)
    ra0, ra1, dec0, dec1 = -10, 10, -10, 10
    res = 1
    template = so_map.car_template(1, ra0, ra1, dec0, dec1, res)

    my_map_tot = template.copy()
    my_map_tot.data = my_map_tot.data * 0
    for i, comp in enumerate(all_comp):
        my_map = sph_tools.alm2map(comp_alms[i], template)
        my_map.plot(file_name=f"{comp_dir}/{run_name}_{comp}", color_range=300)
        my_map_tot.data += my_map.data
        my_map_tot.plot(file_name=f"{comp_dir}/{run_name}_all_{i}", color_range=300)

    
