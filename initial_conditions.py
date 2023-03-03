import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def generate_gaussian_radial_fourier(k, sigma):
    delta_k = - np.exp( - sigma ** 2 * k ** 2 / 2)
    delta_k_int = InterpolatedUnivariateSpline(k, delta_k)
    return delta_k_int
    



def generate_gaussian_curvature_2d(sigma, size, points):
    xr = np.linspace(-size / 2.0, size / 2.0, points)
    yr = np.linspace(-size / 2.0, size / 2.0, points)
    x, y = np.meshgrid(xr, yr)
    
    delta = - 1 / (2 * np.pi * sigma **2) * np.exp(-(x**2 + y**2) / (2 * sigma ** 2))
    
    delta -= np.mean(delta)
    delta = np.fft.fftshift(delta)

    ft_delta = np.fft.fft2(delta)
    kx = np.fft.fftfreq(delta.shape[0], d=xr[1] - xr[0]) * 2 * np.pi
    ky = np.fft.fftfreq(delta.shape[0], d=xr[1] - xr[0]) * 2 * np.pi
    kxr, kyr = np.meshgrid(kx, ky)
    k = np.sqrt(kxr ** 2 + kyr ** 2)

    return x, y, delta, kxr, kyr, k, ft_delta


def from_curvature_to_v_init(ft_psi_ini, k):
    
    # for the velocity we won't care about the normalisation and only care
    # about the sign and spatial variation for the moment, so assume
    # a unknown normalisation, also note that we want the velocity field
    # while class compute the divergence of the velocity field
    # theta = -i \vec{k}.\vec{v}
    
    ft_v = k * ft_psi_ini
    v = np.fft.fftshift(np.fft.ifft2(ft_v).real)
    
    return v, ft_v
