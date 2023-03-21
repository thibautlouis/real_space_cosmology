import numpy as np
import pylab as plt
import camb
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
    
    

d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])

anim_dir = "2d_anim"
os.makedirs(anim_dir, exist_ok=True)


params = d["params"]
gauge = d["gauge"]
species = d["species"]

epoch = "pre_decoupling"

if epoch == "pre_decoupling":
    redshifts0 = np.linspace(18500, 3000, 80)
    redshifts1 = np.linspace(2800, 1100, 30)
    redshifts = np.append(redshifts0, redshifts1)
    
if epoch == "post_decoupling":
    redshifts = np.linspace(1100,100, 70)
    

k, evolution_delta, _ = camb_class.get_tf_class(params, species, redshifts, kmax=100, gauge=gauge)

pk_init = params["As"] * (k / (0.05)) ** (params["ns"] - 1) * (2 * np.pi ** 2 / k ** 3) #initial condition given in term of R
pk_final = pk_init[:, None, None] * evolution_delta ** 2

size = 1000
points = 2048

np.random.seed(0)

xr = np.linspace(-size / 2.0, size / 2.0, points)
yr = np.linspace(-size / 2.0, size / 2.0, points)
kx = np.fft.fftfreq(points, d = xr[1] - xr[0]) * 2 * np.pi
ky = np.fft.fftfreq(points, d = yr[1] - yr[0]) * 2 * np.pi
kxr, kyr = np.meshgrid(kx, ky)
k_2d = np.sqrt(kxr ** 2 + kyr ** 2)
init = np.random.randn(points, points)


def animate_func(i, specy, min_max0, epoch):

    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(snapshots[i])
    ax.set_title(f"{specy}, z = {redshifts[i]:.1f}", fontsize=12)
    
    if (epoch == "pre_decoupling"):
        #fix the colorscale
        im.set_clim(-1.5*min_max0, 1.5*min_max0)
    else:
        vmax     = np.max(snapshots[i])
        vmin     = np.min(snapshots[i])
        min_max = np.minimum(np.abs(vmin), vmax)
        im.set_clim(-min_max, min_max)
        


    return [im]

fps = 7
for j, specy in enumerate(species):

    snapshots = []

    for i, z in enumerate(redshifts):
    
        print("z", z)

        im = generate_gaussian_random_field(init, k_2d, k, pk_final[:, i, j])
        im -= np.mean(im)
        snapshots += [im]
    
    
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    a = snapshots[0]
    
    max, min = np.max(a), np.min(a)
    min_max0 = np.minimum(np.abs(min), max)
    

    im = ax.imshow(a, cmap ="coolwarm", extent=[-size/2, size/2, -size/2, size/2])
    ax.set_xlabel("Mpc (comoving)", fontsize=12)
    ax.set_ylabel("Mpc (comoving)", fontsize=12)
    ax.set_title(f"{specy}, z = {redshifts[0]:.1f}", fontsize=12)
    
    if (epoch == "pre_decoupling"):
        im.set_clim(-1.5*min_max0, 1.5*min_max0)
    else:
        im.set_clim(min_max0, min_max0)

    cb = fig.colorbar(im, cax=cax)
    
    anim = animation.FuncAnimation(
                                   fig,
                                   partial(animate_func, specy=specy, min_max0=min_max0, epoch=epoch),
                                   frames = len(redshifts),
                                   interval = 1000/fps, # in ms
                                   )

    anim.save(f"{anim_dir}/2d_{specy}_{gauge}_{epoch}.mp4", fps=fps, extra_args=["-vcodec", "libx264"])

