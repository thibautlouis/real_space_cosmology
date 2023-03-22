import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import pickle 
import os, sys
import dict_utils




def delta_animation(data, lineplot, rmin=0, rmax=300, i_range=None, interval=30):

    r = data["r"]
    delta_all = data["delta_all"]
    w = (r>=rmin) & (r<rmax)
    r = r[w]
    delta = delta_all[w]

    conformal_time = data["conformal_time"]
    sound_horizon = data["sound_horizon"]
    redshifts = data["redshifts"]
    species = data["species"]
    n_species = len(species)

    c_ls = lineplot
            
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.2)
    lines = []
    for name in species:
        line, = ax.plot([], [], color=c_ls[name][0], ls=c_ls[name][1], lw=2, label=name)
        lines.append(line)
    #line = ax.axvline(0, color='k', ls='--', alpha=1, lw=1, label="Horizon")
    #lines.append(line)

    def init():

        ax.set_xlim(0, rmax)
        ax.set_ylabel(r"$r^{2}\delta(r)$", fontsize=12)
        ax.set_xlabel(r"$r$ [Mpc]", fontsize=12)
        return lines,

    def update(frame):

        ydatas = []
        for j in range(n_species):
            line = lines[j]
            ydata = delta[:, frame, j] * r ** 2
            line.set_data(r, ydata)
            ydatas.append(ydata)
    
        ymin = np.min(ydatas)
        ymax = np.max(ydatas)
        ymin_ = ymin - 0.05*(ymax-ymin)
        ymax_ = ymax + 0.05*(ymax-ymin)
        #line = lines[n_species]
       # line.set_xdata(conformal_time[frame])

        ax.set_ylim(ymin_, ymax_)
        ax.set_title(f"z = {redshifts[frame]:.1f}", fontsize=12)
        ax.legend(loc="upper right")
    
        #lines[6].set_data(sound_horizon[frame])
    
        ax.figure.canvas.draw()
        return lines,

    ani = FuncAnimation(fig, update, frames=i_range,interval=interval, init_func=init, blit=False)
    return ani


def load(filename):
    output = pickle.load(open(filename, "rb"))
    return output


d = dict_utils.so_dict()
d.read_from_file(sys.argv[1])

anim_dir = "green_function"
os.makedirs(anim_dir, exist_ok=True)

run_name = d["run_name"]
gauge = d["gauge"]
lineplot = d["lineplot"]
divide_by_k2 = d["divide_by_k2"]

if divide_by_k2 == True:
    name = "_divided_by_k2"
else:
    name = ""

data_file = f"{anim_dir}/delta_outputs_{gauge}_{run_name}_{name}.pkl"

data = load(data_file)

print("Making pk video....")
anim_delta = delta_animation(data, lineplot, i_range = np.arange(0, d["n_a"]), interval=10)

def save_anim(anim, fname, fps=60, **kwargs):
    writermp4 = animation.FFMpegWriter(fps=fps)
    anim.save(fname, writer=writermp4, **kwargs)

save_anim(anim_delta, f"{anim_dir}/delta_ylinear_{gauge}_{run_name}_{name}.mp4", fps=10, dpi=200)
#plt.show()

