import numpy as np
import pylab as plt
import camb
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline
import os, sys
import dict_utils
import camb_class
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from functools  import partial

def generate_gaussian_random_field(init, k_2d, ks, pk):
    
    pk_int = InterpolatedUnivariateSpline(ks, pk)
    pk_2d = np.abs(pk_int(k_2d))
    ft = np.fft.fft2(init.copy())
    ft *= np.sqrt(pk_2d)
    im = np.fft.ifft2(ft).real
    return im
    
def save(output, filename):
    pickle.dump(output, open(filename, "wb"))


d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])

field_dir = "field_evol"
os.makedirs(field_dir, exist_ok=True)


gauge = d["gauge"]
species = d["species"]
run_name = d["run_name"]
params = d[f"params_{run_name}"]


size = 1000
points = 1024

np.random.seed(0)

xr = np.linspace(-size / 2.0, size / 2.0, points)
yr = np.linspace(-size / 2.0, size / 2.0, points)
kx = np.fft.fftfreq(points, d = xr[1] - xr[0]) * 2 * np.pi
ky = np.fft.fftfreq(points, d = yr[1] - yr[0]) * 2 * np.pi
kxr, kyr = np.meshgrid(kx, ky)
k_2d = np.sqrt(kxr ** 2 + kyr ** 2)
init = np.random.randn(points, points)


for epoch in ["post_decoupling",  "pre_decoupling"]:


    if epoch == "pre_decoupling":
        redshifts0 = np.linspace(18500, 3000, 80)
        redshifts1 = np.linspace(2800, 1100, 30)
        redshifts = np.append(redshifts0, redshifts1)
    
    if epoch == "post_decoupling":
        redshifts = np.linspace(1100,100, 70)
    

    k, evolution_delta, _ = camb_class.get_tf_class(params, species, redshifts, kmax=100, gauge=gauge)
    pk_init = params["As"] * (k / (0.05)) ** (params["ns"] - 1) * (2 * np.pi ** 2 / k ** 3) #initial condition given in term of R

    pk_final = pk_init[:, None, None] * evolution_delta ** 2

    field, pk = {}, {}

    for j, specy in enumerate(species):
        for i, z in enumerate(redshifts):
    
            im = generate_gaussian_random_field(init, k_2d, k, pk_final[:, i, j])
            im -= np.mean(im)
            
            field[z, specy] = im
            
            pk[z, specy] = pk_final[:, i, j]
    
    output = {"size": size,
              "field": field,
              "k": k,
              "pk": pk,
              "species": species,
              "redshifts": redshifts}
    
    output_file = f"{field_dir}/field_{gauge}_{run_name}_{epoch}.pkl"
    save(output, output_file)
    
    
