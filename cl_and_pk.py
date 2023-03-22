import numpy as np
import pylab as plt
import os, sys
import dict_utils
import camb_class



d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])

spec_dir = "spectra"
os.makedirs(spec_dir, exist_ok=True)

run_name = d["run_name"]
params = d[f"params_{run_name}"]

spec_dict = camb_class.get_cl_and_pk_camb(params)
plt.plot(spec_dict["ls"], spec_dict["TT"])
plt.show()
plt.loglog(spec_dict["kh"], spec_dict["pk"])
plt.show()
plt.loglog(spec_dict["l_lens"], spec_dict["cl_lens"])
plt.show()

