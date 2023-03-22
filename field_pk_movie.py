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
    
def animate_func(i, specy):

    if i % fps == 0:
        print( '.', end ='' )

    ymax = np.max(snapshots[i])
    ymin = 10**-4 * ymax

    ax.set_ylim(0.1*ymin, ymax*10)

    line.set_data(k, snapshots[i])
    ax.set_title(f"{specy}, z = {redshifts[i]:.1f}", fontsize=12)

    
    return [line]


d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])

field_dir = "field_evol"
os.makedirs(field_dir, exist_ok=True)


gauge = d["gauge"]
run_name = d["run_name"]



for epoch in ["post_decoupling",  "pre_decoupling"]:


    data_file = f"{field_dir}/field_{gauge}_{run_name}_{epoch}.pkl"
    data = load(data_file)
    species = data["species"]
    redshifts = data["redshifts"]
    size = data["size"]
    k = data["k"]
    fps = 7
    for j, specy in enumerate(species):
        snapshots = []
        for i, z in enumerate(redshifts):
            snapshots += [data["pk"][z, specy]]

        fig = plt.figure( figsize=(8,8) )
        ax = fig.add_subplot(111)
        




        line, = ax.loglog(k, snapshots[0])
        
        ymax = np.max(snapshots[i])
        ymin = 10**-4 * ymax

        ax.set_ylim(ymin, ymax*10)

        ax.set_xlabel("k (Mpc^-1) comoving)", fontsize=12)
        ax.set_title(f"{specy}, z = {redshifts[0]:.1f}", fontsize=12)
        
        
        anim = animation.FuncAnimation(
                                    fig,
                                    partial(animate_func, specy=specy),
                                    frames = len(redshifts),
                                    interval = 1000/fps, # in ms
                                    )

        anim.save(f"{field_dir}/Pk_{specy}_{gauge}_{run_name}_{epoch}.mp4", fps=fps, extra_args=["-vcodec", "libx264"])

