from classy import Class
from math import pi
import matplotlib
import matplotlib.pyplot as plt


common_settings = {# LambdaCDM parameters
'h':0.67810,
'omega_b':0.02238280,
'omega_cdm':0.12038,
'A_s':2.100549e-09,
'n_s': 0.9660499,
'tau_reio':0.05430842,
# output and precision parameters
'output':'tCl,mTk,vTk',
'l_max_scalars':5000,
'P_k_max_1/Mpc':10.0,
'gauge':'newtonian'
}

M = Class()
M.set(common_settings)
M.compute()
derived = M.get_current_derived_parameters(['z_rec','tau_rec','conformal_age'])
print (derived.keys())
z_rec = derived['z_rec']
z_rec = int(1000.*z_rec)/1000. # round down at 4 digits after coma
print ('z_rec=',z_rec)
tau_0_minus_tau_rec_hMpc = (derived['conformal_age']-derived['tau_rec'])*M.h()

M.empty() # reset input parameters to default, before passing a new parameter set
M.set(common_settings)
M.set({'z_pk':z_rec})
M.compute()

one_time = M.get_transfer(z_rec)
print (one_time.keys())
k = one_time['k (h/Mpc)']
Theta0 = 0.25*one_time['d_g']
phi = one_time['phi']
psi = one_time['psi']
theta_b = one_time['t_b']
# compute related quantitites
R = 3./4.*M.Omega_b()/M.Omega_g()/(1+z_rec)  # R = 3/4 * (rho_b/rho_gamma) at z_rec
zero_point = -(1.+R)*psi                     # zero point of oscillations: -(1.+R)*psi
Theta0_amp = max(Theta0.max(),-Theta0.min()) # Theta0 oscillation amplitude (for vertical scale of plot)
print ('At z_rec: R=',R,', Theta0_amp=',Theta0_amp)

background = M.get_background() # load background table
print (background.keys())
#
background_tau = background['conf. time [Mpc]'] # read confromal times in background table
background_z = background['z'] # read redshift
background_kh = 2.*pi*background['H [1/Mpc]']/(1.+background['z'])/M.h() # read kh = 2pi aH = 2pi H/(1+z) converted to [h/Mpc]
background_ks = 2.*pi/background['comov.snd.hrz.']/M.h() # read ks = 2pi/rs converted to [h/Mpc]

from scipy.interpolate import interp1d
kh_at_tau = interp1d(background_tau,background_kh)
ks_at_tau = interp1d(background_tau,background_ks)
#
# finally get these scales
#
tau_rec = derived['tau_rec']
kh = kh_at_tau(tau_rec)
ks = ks_at_tau(tau_rec)
print ('at tau_rec=',tau_rec,', kh=',kh,', ks=',ks)
fig, (ax_Tk, ax_Tk2, ax_Cl) = plt.subplots(3,sharex=True,figsize=(8,12))
fig.subplots_adjust(hspace=0)
##################
#
# first figure with transfer functions
#
##################
ax_Tk.set_xlim([3.e-4,0.5])
ax_Tk.set_ylim([-1.1*Theta0_amp,1.1*Theta0_amp])
ax_Tk.tick_params(axis='x',which='both',bottom='off',top='on',labelbottom='off',labeltop='on')
ax_Tk.set_xlabel(r'$\mathrm{k} \,\,\,  \mathrm{[h/Mpc]}$')
ax_Tk.set_ylabel(r'$\mathrm{Transfer}(\tau_\mathrm{dec},k)$')
ax_Tk.xaxis.set_label_position('top')
ax_Tk.grid()
#
ax_Tk.axvline(x=kh,color='r')
ax_Tk.axvline(x=ks,color='y')
#
ax_Tk.annotate(r'Hubble cross.',
                xy=(kh,0.8*Theta0_amp),
                xytext=(0.15*kh,0.9*Theta0_amp),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
ax_Tk.annotate(r'sound hor. cross.',
                 xy=(ks,0.8*Theta0_amp),
                 xytext=(1.3*ks,0.9*Theta0_amp),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
#
ax_Tk.semilogx(k,psi,'y-',label=r'$\psi$')
ax_Tk.semilogx(k,phi,'r-',label=r'$\phi$')
ax_Tk.semilogx(k,zero_point,'k:',label=r'$-(1+R)\psi$')
ax_Tk.semilogx(k,Theta0,'b-',label=r'$\Theta_0$')
ax_Tk.semilogx(k,(Theta0+psi),'c',label=r'$\Theta_0+\psi$')
ax_Tk.semilogx(k,theta_b,'g-',label=r'$\theta_b$')
#
ax_Tk.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
plt.show()
