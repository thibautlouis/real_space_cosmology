import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import pickle 
import os, sys
import dict_utils
from scipy.interpolate import InterpolatedUnivariateSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable




def load(filename):
    output = pickle.load(open(filename, "rb"))
    return output


d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])

anim_dir = "movies"
os.makedirs(anim_dir, exist_ok=True)

gauge = d["gauge"]
divide_by_k2 = d["divide_by_k2"]

if divide_by_k2 == True:
    name = "_divided_by_k2"
else:
    name = ""

data_file = f"{anim_dir}/delta_outputs_{gauge}{name}.pkl"

data = load(data_file)

r = data["r"]
delta_all = data["delta_all"]
redshifts = data["redshifts"]
species = data["species"]

xr = np.linspace(-300, 300, 1000)
yr = np.linspace(-300, 300, 1000)
x, y = np.meshgrid(xr, yr)
r2d = np.sqrt(x**2 + y**2)
fps = 10



def animate_func(i):

    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(snapshots[i])
    ax.set_title(f"z = {redshifts[i]:.1f}", fontsize=12)
    vmax     = np.max(snapshots[i])
    vmin     = np.min(snapshots[i])

    im.set_clim(vmin, vmax)

    return [im]


for j, specy in enumerate(species):
    print(specy)
    snapshots =[]
    for i, z in enumerate(redshifts):
        delta = delta_all[:, i, j] * r ** 2
        delta_int = InterpolatedUnivariateSpline(r, delta)
        delta2d = delta_int(r2d)
        snapshots += [delta2d]
        
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    a = snapshots[0]
    
    im = ax.imshow(a, vmin=np.min(a), vmax=np.max(a), cmap ="coolwarm")
    cb = fig.colorbar(im, cax=cax)
    

    anim = animation.FuncAnimation(
                                   fig,
                                   animate_func,
                                   frames = len(redshifts),
                                   interval = 1000/fps, # in ms
                                   )

    anim.save(f"{anim_dir}/2d_{specy}_{gauge}{name}.mp4", fps=fps, extra_args=["-vcodec", "libx264"])

