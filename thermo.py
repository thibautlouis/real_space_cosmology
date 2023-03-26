import numpy as np
import pylab as plt
import os, sys
import dict_utils
import camb_class



d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])
run_names = ["lcdm", "no_dm_high_baryons"]
for run_name in run_names:
    params = d[f"params_{run_name}"]
    camb_class.get_thermo_class(params)
