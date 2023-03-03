import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def normalize(real):
    minimum, maximum = real.min(), real.max()
    real[real>0] = real[real>0]/abs(maximum)
    real[real<0] = real[real<0]/abs(minimum)
    return real

def plot_density(x, y, z, delta, name, size, plot_dir, cmap='coolwarm', vmin=None, vmax=None, radial_profile=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(delta, cmap=cmap,vmin=vmin,vmax=vmax, extent=(-size/2, size/2, -size/2, size/2))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.title(f"z={z}")
    #plt.show()
    plt.savefig(f"{plot_dir}/{name}_{z}.png")
    plt.clf()
    plt.close()
    
    if radial_profile == True:
        npts = delta.shape[0]
        r = np.sqrt(x ** 2 + y ** 2)
        r = r[npts // 2, npts // 2 :]
        plt.plot(r, delta[npts // 2, npts // 2 :] * r ** 2)
        plt.title(f"z={z}")
        #plt.show()
        plt.savefig(f"{plot_dir}/profile_{name}_{z}.png")
        plt.clf()
        plt.close()

    
def plot_density_and_velocity(x, y, z, delta, v, name, plot_dir, rmin=20, cmap='coolwarm', vmin=None, vmax=None):
    
    theta = np.arctan2(y, x)
    ex, ey = np.cos(theta), np.sin(theta)
    r = np.sqrt(x ** 2 + y ** 2)
    id = np.where(r < rmin)

    v[id] = 0
    v = normalize(v)
    vx = v * ex
    vy = v * ey
    
    fig, ax = plt.subplots(figsize=(12, 8))
    strm = ax.streamplot(x, y, vx, vy, color = v, cmap ='seismic', density = 2, linewidth=0.7)
#    print(x.min(),x.max())
    im = ax.imshow(delta, cmap =cmap, extent=(x.min(),x.max(),x.min(),x.max()))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.title(f"z={z}")
    #plt.show()
    plt.savefig(f"{plot_dir}/{name}_with_velocity_flow_{z}.png")
    plt.clf()#
    plt.close()


