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

def load(filename):
    output = pickle.load(open(filename, "rb"))
    return output
    
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


d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])

field_dir = "field_evol"
os.makedirs(field_dir, exist_ok=True)


gauge = d["gauge"]
run_name = d["run_name"]



for epoch in ["pre_decoupling", "post_decoupling"]:


    data_file = f"{field_dir}/field_{gauge}_{run_name}_{epoch}.pkl"
    data = load(data_file)
    species = data["species"]
    redshifts = data["redshifts"]
    size = data["size"]

    fps = 7
    for j, specy in enumerate(species):
        snapshots = []
        for i, z in enumerate(redshifts):
            snapshots += [data["field"][z, specy]]

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

        anim.save(f"{field_dir}/2d_{specy}_{gauge}_{run_name}_{epoch}.mp4", fps=fps, extra_args=["-vcodec", "libx264"])

