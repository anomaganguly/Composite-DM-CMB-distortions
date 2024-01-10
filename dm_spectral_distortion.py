# Python code to :
# find mu and y param -> delE/E constraint from COBE


import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
from scikits.odes import ode
from scipy.special import zeta
from scipy.interpolate import LinearNDInterpolator
import scipy.integrate as integrate
from scipy.optimize import bisect
from itertools import chain
from colossus.cosmology import cosmology
from colossus.utils import constants
from collections import OrderedDict

# mass unit = Msun/h
# mass func unit = (h/Mpc)**3 in comoving coord

# mass_expo = 0.1 #m_f#
# delE = 0.01 #s_d#

# num0 = str(mass_expo)
# num1 = str(delE)


cosmo = cosmology.setCosmology('planck18')

# Defining the constants
c = constants.C
mpc = constants.MPC
kpc = constants.KPC
H0 = cosmo.H0
hp = constants.H
kb = constants.KB
mp = constants.M_PROTON
msun = constants.MSUN
G = constants.G_CGS
me = 9.10938e-28 #mass in gm
mp = 1.6726219e-24 #mass in gm
qe = 4.8e-10
kbev = 8.617e-5
h = H0/100
sigma_T = 6.652e-25
Mev = 1.783e-33*(1.e6)
a10H = 2.85e-15
alpha = 2*np.pi*qe**2/(hp*c)
z_hrec = 1100
z_eq = 3500
z_bb = np.log10(2.E6) 
z_c = np.log10(1.E5) 
aR = 8*np.pi**5*kb**4/(15*c**3*hp**3)
bR = 16*np.pi*kb**3*zeta(3)/(c**3*hp**3)
cm_Mev = 5.06E10

# Dark matter model parameters
g10 = 3.                     # Ratio of triplet to singlet degeneracy factors


# Reading the COBE-FIRAS monopole spectrum

# Reference1 = Table 4 of Fixsen et al. 1996 ApJ 473, 576.
# Reference2 = Fixsen & Mather 2002 ApJ 581, 817.
# Column 1 = frequency from Table 4 of Fixsen et al., units = cm^-1
# Column 2 = FIRAS monopole spectrum computed as the sum of a T_0 = 2.725 K BB spectrum and the residual in column 3, units = MJy/sr
# Column 3 = residual monopole spectrum from Table 4 of Fixsen et al., units = kJy/sr
# Column 4 = spectrum uncertainty (1-sigma) from Table 4 of Fixsen et al., units = kJy/sr
# Column 5 = modeled Galaxy spectrum at the Galactic poles from Table 4 of Fixsen et al., units = kJy/sr

# path1 = '/Users/anoma/Documents/paper1_code/cobe_quasar/COBE_data.txt'
# path1 = '/Users/anoma/Documents/paper2_code/cobe_zoo/cobe_mcmc/COBE_data.txt'
path1 = 'COBE_data.txt'
data_cobe = np.genfromtxt(path1).T

freq = data_cobe[0]*c*1.E-9  #cm-1 to GHz
intensity = data_cobe[1]*1.E3 #kJy.sr-1   1 Jy.sr-1 = 10^−23 erg⋅s−1⋅cm−2⋅Hz−1.sr-1
residual = data_cobe[2]
del_intensity = data_cobe[3]
gal_intensity = data_cobe[4]

galnu = interpolate.interp1d(freq, gal_intensity) # in cgs units
T_0 = 2.725

# Transition temperature range for DM absorption in 50-10^5 redshift range
T_starmax = np.max(freq)*(1.E5)*1.E9*hp/kb
T_starmin = np.min(freq)*(1+50)*1.E9*hp/kb

###################################### Required functions ################################

# Blackbody specific intensity
def B(nu):
    return (2*hp*(nu*1.E9)**3/c**2)*(1/(np.exp(hp*nu*1.E9/(kb*T_0))-1))*1.E20 # cgs to kJy.sr-1

# Derivative of blackbody specific intensity
def dBdT(nu):
    return (2*hp*(nu*1.E9)**3/c**2)*(-1/(np.exp(hp*nu*1.E9/(kb*T_0))-1)**2)*(-hp*(nu*1.E9)*np.exp(hp*nu*1.E9/(kb*T_0))/(kb*T_0**2))*1.E20 # cgs to kJy.sr-1

# Planck spectrum energy density
def epl(z):
    return aR*Tcmb(z)**4

# Planck spectrum number density
def npl(z):
    return bR*Tcmb(z)**3

# Dimensionless frequency w.r.t. CMB temperature
def xcmb(z, T_star):
    return T_star/Tcmb(z)

# Rate coeff for diff electromagnetic processes
def Kc(z):
    return 3.226e-5*((1+10**z)/(1+2.e6))**4

def Kdc(z):
    return 2.397e-9*((1+10**z)/(1+2.e6))**5

def Kbr(z):
    return 3.41e-11*((1+10**z)/(1+2.e6))**(5./2.)

def xc(z):
    return np.sqrt((Kbr(z)+Kdc(z))/Kc(z))

# Expansion rate 
def H_z(z):
    return cosmo.Hz(10**z)*(1.e5/mpc)

# CMB temperature (K)
def Tcmb(z):
    return 2.726*(1+10**z)
    
# Transition frequency in MHz
def f0(T_star):
    return (kb*T_star/hp)*1.e-6 

# CMB intensity
def Jcmb(z, T_star):
    if T_star/Tcmb(z) > 100:
        return (2*hp*(f0(T_star)*1.e0)**3/c**2)*(np.exp(-T_star/Tcmb(z)))
    else:
        return (2*hp*(f0(T_star)*1.e0)**3/c**2)*(1/(np.exp(T_star/Tcmb(z))-1))
    
# Frequency (GHz) to redshift
def invert1(fh, T_star):
    return (f0(T_star)*1.E-3/fh) - 1

# Redshift to frequency (GHz)
def invert2(zh, T_star):
    return f0(T_star)*1.E-3/(1+zh)

# DM number density
def ndm(z, mdm):
    return (cosmo.rho_m(10**z)-cosmo.rho_b(10**z))*(msun*h**2/(kpc**3))/mdm

# DM decoupling redshift (Energy exchange rate from baryon to DM is less than Hubble rate)
def fzdec(z, T_star, mdm, alpha_A):
    return Edmb(z, [Tcmb(z), Tcmb(z)], T_star, mdm, alpha_A) - 1

# DM decoupling redshift
def zdecal(T_star, mdm, eps):
    mevdm = (mdm/Mev)
    zdecmin = z_eq          # min allowed decoupling redshift so that matter radiation equality is not disturbed
    zdecmax = -1 + mevdm*1.e6/(20*kbev*2.726) # max decoupling redshift at which DM becomes non-relativistic
    if np.sign(fzdec(np.log10(zdecmin), T_star, mdm, eps))<0 and np.sign(fzdec(np.log10(zdecmax), T_star, mdm, eps))<0:#dm decoupled from cmb via scattering before zdecmax
        return zdecmax
    elif np.sign(fzdec(np.log10(zdecmin), T_star, mdm, eps))>0 and np.sign(fzdec(np.log10(zdecmax), T_star, mdm, eps))>0:#dm coupled to cmb via scattering after zdecmin
        return 0.
    else:
        zdm_dec = 10**(bisect(fzdec, np.log10(zdecmin), np.log10(zdecmax), xtol=1.e-10, args = (T_star, mdm, eps)))
        return zdm_dec

# DM temperature (K)
def Tdmad(z, zdeco):
    if z >= zdeco:
        return Tcmb(z)
    else:
        return Tcmb(zdeco)*((1+10**z)/(1+10**zdeco))**2
        
# Ratio of DM population 
def n10(T, T_star):
    if T>=0.:
        return g10*np.exp(-T_star/T)
    else:
        return 0.
    
def n01(T, T_star):
        return (1/g10)*np.exp(T_star/T)
    
# DM collisional coupling
def C10(z, mdm, a1, beta, Tdm):
    evdm = (mdm/Mev)*1.E6
    if Tdm > 0:
        coll = ndm(z, mdm)*a1*mdm*c*np.power((8*kbev*Tdm/(np.pi*evdm)),beta)
        return coll
    else:
        return 0.
    
def C01(z, T_star, mdm, a1, beta, Tdm):
    return C10(z, mdm, a1, beta, Tdm)*n10(Tdm, T_star)

# DM transition Einstein Coefficients
def EC(T_star, alpha_A):
    A10 =  alpha_A*a10H                             # Einstein coeff of spontaneous emission 
    B10 = (c**2/(2*hp*(f0(T_star)*1.e0)**3))*A10    # Stimulated emission coeff 
    B01 = g10*B10                                   # Absorption coeff 
    return [A10, B10, B01]

# Transition rates for level 0  
def f_n0(z, Tex, T_star, mdm, alpha_A, a1, beta, Tdm):
    return -1/(H_z(z)*(1+10**z))*((n10(Tex, T_star)*C10(z, mdm, a1, beta, Tdm) - C01(z, T_star, mdm, a1, beta, Tdm)) + (n10(Tex, T_star)*EC(T_star, alpha_A)[0] + (n10(Tex, T_star)*EC(T_star, alpha_A)[1] - EC(T_star, alpha_A)[2])*Jcmb(z, T_star)))

# Transition rates for level 1
def f_n1(z, Tex, T_star, mdm, alpha_A, a1, beta, Tdm):
    if T_star/Tcmb(z) > 500: 
        return 1/(H_z(z)*(1+10**z))*((C10(z, mdm, a1, beta, Tdm) - C10(z, mdm, a1, beta, Tdm)*np.exp(-T_star*(1/Tdm-1/Tex))) + EC(T_star, alpha_A)[0] + (EC(T_star, alpha_A)[0]*np.exp(-T_star/Tcmb(z)) - EC(T_star, alpha_A)[0]*(1/(np.exp(T_star*(1/Tcmb(z)-1/Tex))-np.exp(-T_star/Tex)))))
    else:
        return 1/(H_z(z)*(1+10**z))*((C10(z, mdm, a1, beta, Tdm) - C10(z, mdm, a1, beta, Tdm)*np.exp(-T_star*(1/Tdm-1/Tex))) + EC(T_star, alpha_A)[0] + (EC(T_star, alpha_A)[0]*1/(np.exp(T_star/Tcmb(z))-1) - EC(T_star, alpha_A)[0]*(1/(np.exp(T_star*(1/Tcmb(z)-1/Tex))-np.exp(-T_star/Tex)))))

# Optical depth due to DM
def tau_dm(z, T, T_star, mdm, alpha_A):
    return c**3*EC(T_star, alpha_A)[0]*ndm(z, mdm)/(8*np.pi*(f0(T_star)*1.e6)**3*H_z(z))*(1-np.exp(-T_star/T))*g10/(1+n10(T, T_star))

# Differential brightness temperature today
def Tdb(z, T, T_star, mdm, alpha_A):
    if tau_dm(z, T, T_star, mdm, alpha_A).all() < 1.e-5:
        return (1/(np.exp(T_star/T)-1)-1/(np.exp(T_star/Tcmb(z))-1))*(tau_dm(z, T, T_star, mdm, alpha_A))*(T_star/(1.+10**z))
    else:
        return (1/(np.exp(T_star/T)-1)-1/(np.exp(T_star/Tcmb(z))-1))*(1-np.exp(-tau_dm(z, T, T_star, mdm, alpha_A)))*(T_star/(1.+10**z))

# Brightness temperature today
def Tb(z, T, T_star, mdm, alpha_A):
    return Tdb(z, T, T_star, mdm, alpha_A) + (T_star/(np.exp(T_star/Tcmb(z))-1))/(1+10**z)

# Velocity averaged scattering cros-section
def sigmadmb(mdm, mbgm, eps, sctype):
    mevdm = (mdm/Mev)
    mb = (mbgm/Mev)
    mudmb = mevdm*mb/(mevdm+mb)
    sigma_es = 16*np.pi*eps**2*alpha**2*mudmb**2/(3*mevdm**4)*(1/cm_Mev)**2
    sigma_is1 = 8*np.pi*eps**2*alpha**2*mudmb**2/(mevdm**2*mb**2)*(1/cm_Mev)**2
    sigma_is2 = 32*np.pi*eps**2*mudmb**2/(mevdm**2*mb**2)*(1/cm_Mev)**2
    if sctype == 0:
        return sigma_es
    else:
        return (sigma_is1, sigma_is2/sigma_is1)
    
# Elastic scattering energy transfer rate
def Rdmb_es(z, T, mdm, epsdm):
    mevdm = (mdm/Mev)
    mevp = (mp/Mev)
    meve = (me/Mev)

    Rdme = 16*ne(z)*sigmadmb(mdm, me, epsdm, 0)*c*mevdm*meve/(np.sqrt(2*np.pi)*(meve+mevdm)**2)*np.sqrt(kbev*Tg(z)*1.e-6/(meve) + kbev*T[1]*1.e-6/(mevdm))
    Rdmp = 16*n_p(z)*sigmadmb(mdm, mp, epsdm, 0)*c*mevdm*mevp/(np.sqrt(2*np.pi)*(mevp+mevdm)**2)*np.sqrt(kbev*Tg(z)*1.e-6/(mevp) + kbev*T[1]*1.e-6/(mevdm))
    Rdmhep = 16*nhe_p(z)*sigmadmb(mdm, 4*mp, epsdm, 0)*c*mevdm*4*mevp/(np.sqrt(2*np.pi)*(4*mevp+mevdm)**2)*np.sqrt(kbev*Tg(z)*1.e-6/(4*mevp) + kbev*T[1]*1.e-6/(mevdm))
    Rdmhepp = 16*nhe_pp(z)*4*sigmadmb(mdm, 4*mp, epsdm, 0)*c*mevdm*4*mevp/(np.sqrt(2*np.pi)*(4*mevp+mevdm)**2)*np.sqrt(kbev*Tg(z)*1.e-6/(4*mevp) + kbev*T[1]*1.e-6/(mevdm))
    return Rdme+Rdmp+Rdmhep+Rdmhepp

# Inelastic scattering energy transfer rate
def Rdmb_is(z, T, T_star, mdm, epsdm):
    mevdm = (mdm/Mev)
    mevp = (mp/Mev)
    meve = (me/Mev)

    fnr_dist = lambda x: 2*np.sqrt(x/np.pi)*np.exp(-x)
    fnr_ei = integrate.quad(fnr_dist, T_star/Tcmb(z), 1.E4, limit = 10000, epsrel=1.e-6)[0]

    if kbev*Tcmb(z)*1.e-6>0.5/20:
        mt = meve*1.E6/(kbev*Tcmb(z))
        f_np = lambda x: 4*np.pi*2*(kb*Tcmb(z)/(c*hp))**3*x**2*1/(np.exp(np.sqrt(x**2+mt**2))+1) if (np.sqrt(x**2+mt**2)<50) else 4*np.pi*2*(kb*Tcmb(z)/(c*hp))**3*x**2*np.exp(-np.sqrt(x**2))
        f_e = integrate.quad(f_np, T_star/Tcmb(z), 1.E4, limit = 100000, epsrel=1.e-8)[0]/integrate.quad(f_np, 0., 1.E4, limit = 100000, epsrel=1.e-8)[0]
    else:
        f_e = fnr_ei

    Rfdme = 16*ne(z)*sigmadmb(mdm, me, epsdm, 1)[0]*c*mevdm*meve/(np.sqrt(2*np.pi)*(meve+mevdm)**2)*np.sqrt(kbev*Tg(z)*1.e-6/(meve) + kbev*T[1]*1.e-6/(mevdm))
    Rdme = Rfdme + 6*f_e*Rfdme*sigmadmb(mdm, me, epsdm, 1)[1]*(kbev*Tg(z)*1.e-6/(meve) + kbev*T[1]*1.e-6/(mevdm))
    
    Rfdmp = 16*n_p(z)*sigmadmb(mdm, mp, epsdm, 1)[0]*c*mevdm*mevp/(np.sqrt(2*np.pi)*(mevp+mevdm)**2)*np.sqrt(kbev*Tg(z)*1.e-6/(mevp) + kbev*T[1]*1.e-6/(mevdm))
    Rdmp = Rfdmp + 6*fnr_ei*Rfdmp*sigmadmb(mdm, mp, epsdm, 1)[1]*(kbev*Tg(z)*1.e-6/(mevp) + kbev*T[1]*1.e-6/(mevdm))
    
    Rfdmhep = 16*nhe_p(z)*sigmadmb(mdm, 4*mp, epsdm, 1)[0]*c*mevdm*4*mevp/(np.sqrt(2*np.pi)*(4*mevp+mevdm)**2)*np.sqrt(kbev*Tg(z)*1.e-6/(4*mevp) + kbev*T[1]*1.e-6/(mevdm))
    Rdmhep = Rfdmhep + 6*fnr_ei*Rfdmhep*sigmadmb(mdm, 4*mp, epsdm, 1)[1]*(kbev*Tg(z)*1.e-6/(4*mevp) + kbev*T[1]*1.e-6/(mevdm))
    
    Rfdmhepp = 16*nhe_pp(z)*4*sigmadmb(mdm, 4*mp, epsdm, 1)[0]*c*mevdm*4*mevp/(np.sqrt(2*np.pi)*(4*mevp+mevdm)**2)*np.sqrt(kbev*Tg(z)*1.e-6/(4*mevp) + kbev*T[1]*1.e-6/(mevdm))
    Rdmhepp = Rfdmhepp + 6*fnr_ei*Rfdmhepp*sigmadmb(mdm, 4*mp, epsdm, 1)[1]*(kbev*Tg(z)*1.e-6/(4*mevp) + kbev*T[1]*1.e-6/(mevdm))
    
    return Rdme+Rdmp+Rdmhep+Rdmhepp

# Ratio of energy exchange from baryon to DM w.r.t. Hubble rate
def Edmb(z, T, T_star, mdm, alpha_A):
    mevdm = (mdm/Mev)
    epsdm = np.sqrt(alpha_A*(0.068/T_star)**3*(mevdm/0.5)**2)
    return Rdmb_is(z, T, T_star, mdm, epsdm)/((3./2.)*H_z(z))

# Energy exchange from CMB to DM
def Edmcmb(z, T, T_star, mdm, alpha_A, a1):
    mevdm = (mdm/Mev)
    if ndm(z, mdm)*a1*mdm*c*np.power((8*kbev*T[1]*1.e-6/(np.pi*mevdm)),0.5)/H_z(z)>1:
         return edot(z, T[0], T_star, mdm, alpha_A)/((3./2.)*ndm(z, mdm)*H_z(z)*kb*T[1])
    else:
         return 0.

# Solving the thermal history of dark matter 
# Excitation + DM temperature evolution
# Tex=T[0] and Tdm=T[1] ode function
def f_Tex(z, T, Tdot, T_star, mdm, alpha_A, a1, beta):
    Tdot[1] = (1/np.log10(np.exp(1)))*(2*T[1] -(-Edmcmb(z, T, T_star, mdm, alpha_A, a1)*T[1]))*10**z/(1+10**z)                                          
    Tdot[0] = (T[0]**2/T_star)*(10**z)*(f_n1(z, T[0], T_star, mdm, alpha_A, a1, beta, T[1])-f_n0(z, T[0], T_star, mdm, alpha_A, a1, beta, T[1]))

# Only DM temperature evolution from CMB heating 
# Tdm=T[0] ode function
def f_Tdm(z, T, Tdot, T_star, mdm, alpha_A, a1):
    Tdot[0] = (1/np.log10(np.exp(1)))*(2*T[0] -(-Edmcmb(z, [T[0], T[0]], T_star, mdm, alpha_A, a1)*T[0]))*10**z/(1+10**z)                                            

# Rate of energy per unit vol absorption/emission from CMB to DM
def edot(z, T, T_star, mdm, alpha_A):
    factor = -kb*T_star*alpha_A*a10H*ndm(z, mdm)*(1/(1+n10(T, T_star)))
    if xcmb(z, T_star)<100:
        return factor*((g10 - n10(T, T_star))/(np.exp(xcmb(z, T_star))-1)-n10(T, T_star))
    else:
        return factor*((g10 - n10(T, T_star))*(np.exp(-xcmb(z, T_star)))-n10(T, T_star))

# Rate of number of photons per unit vol absorption/emission from CMB to DM
def ndot(z, T, T_star, mdm, alpha_A):
    return edot(z, T, T_star, mdm, alpha_A)/(kb*T_star)

###################### y-distortion ###########################################

# Reading electron, proton and ion fraction and gas temperature
# path2 = '/Users/anoma/Documents/paper2_code/cobe_zoo/cobe_mcmc/rec.txt'
path2 = 'rec.txt'
data2 = np.genfromtxt(path2, skip_header=4).T

x0 = data2[0] #z
x1 = data2[1] #xe = Ne/NH = xp
x2 = data2[2] #xeH = NeH/NH
x3 = data2[3] #xe_He = Ne_He/NH
x4 = data2[4] #Tg

x5 = []
x6 = []
for i in range (len(x0)):
    x = x1[i]-x2[i]-x3[i]
    if x>=0:
        x5.append(x3[i]-x)
        x6.append(x)
    else:
        x5.append(x3[i])
        x6.append(0)

xeint = interpolate.interp1d(x0, x1)
xp = interpolate.interp1d(x0, x2)
xHe_p = interpolate.interp1d(x0, x5)
xHe_pp = interpolate.interp1d(x0, x6)
Tgint = interpolate.interp1d(x0, x4)

# Reading the gaunt factors
# path3 = '/Users/anoma/Documents/paper2_code/cobe_zoo/cobe_mcmc/gaunt_factors.txt'
path3 = 'gaunt_factors.txt'
data3 = np.genfromtxt(path3, skip_header=1).T
xval = data3[0]
yval = data3[1]
gval = data3[2]

gff = LinearNDInterpolator(list(zip(xval, yval)), gval)

def Tg(z):
    if 10**z<25000:
        return Tgint(10**z)
    else:
        return Tcmb(z)
    
# Baryon number density
def nb(z):
    return cosmo.rho_b(10**z)*(msun*h**2/(kpc**3))/mp

# Electron number density
def ne(z):
    me_ev = (me/Mev)*1.E6
    g = 2
    mt = me_ev/(kbev*Tcmb(z))
    # Integrating the Fermi-Dirac distribution to get the electron number density
    f_ne = lambda x: 4*np.pi*g*(kb*Tcmb(z)/(c*hp))**3*x**2*1/(np.exp(np.sqrt(x**2+mt**2))+1) if (np.sqrt(x**2+mt**2)<50) else 4*np.pi*g*(kb*Tcmb(z)/(c*hp))**3*x**2*np.exp(-np.sqrt(x**2))
    if kbev*Tcmb(z)>me_ev*50:
        return 2*2*3*zeta(3)*(kb*Tcmb(z)/((hp/(2*np.pi))*c))**3/(4*np.pi**2)
    elif 10**z>8.E7:
        n_e = integrate.quad(f_ne, 0., 1.E4, limit = 10000, epsrel=1.e-6)
        return 2*n_e[0] #e- and e+
    elif 2.5E4<10**z<=8.E7:
        return (0.76+0.24/2)*nb(z)
    else:
        return xeint(10**z)*0.76*nb(z)
    
vne = np.vectorize(ne)

# Positron number density
def neplus(z):
    me_ev = (me/Mev)*1.E6
    g = 2
    mt = me_ev/(kbev*Tcmb(z))
    f_ne = lambda x: 4*np.pi*g*(kb*Tcmb(z)/(c*hp))**3*x**2*1/(np.exp(np.sqrt(x**2+mt**2))+1) if (np.sqrt(x**2+mt**2)<50) else 4*np.pi*g*(kb*Tcmb(z)/(c*hp))**3*x**2*np.exp(-np.sqrt(x**2))
    if kbev*Tcmb(z)>me_ev*50:
        return 2*3*zeta(3)*(kb*Tcmb(z)/((hp/(2*np.pi))*c))**3/(4*np.pi**2)
    elif 10**z>8.E7:
        n_e = integrate.quad(f_ne, 0., 1.E4, limit = 10000, epsrel=1.e-6)
        return n_e[0]
    else: 
        return 0.

vneplus = np.vectorize(neplus)

# Weighted photon number and energy density calculated by integrating the Planck spectrum from DM production ll to inf
def photon_ne(z, mdm):
    g = 2
    lim = 13.4*(mdm*1.e6/Mev)/(2*kbev*Tcmb(z))
    mt = 0 #me_ev/(kbev*Tcmb(z))
    f_np = lambda x: 4*np.pi*g*(kb*Tcmb(z)/(c*hp))**3*x**2*1/(np.exp(np.sqrt(x**2+mt**2))-1) if (np.sqrt(x**2+mt**2)<50) else 4*np.pi*g*(kb*Tcmb(z)/(c*hp))**3*x**2*np.exp(-np.sqrt(x**2))
    e_np = lambda x: 4*np.pi*g*np.log(2*x*kbev*Tcmb(z)/(mdm*1.e6/Mev))*(kb*Tcmb(z)/(c*hp))**3*x**2*1/(np.exp(np.sqrt(x**2+mt**2))-1) if (np.sqrt(x**2+mt**2)<50) else \
        4*np.pi*g*np.log(2*x*kbev*Tcmb(z)/(mdm*1.e6/Mev))*(kb*Tcmb(z)/(c*hp))**3*x**2*np.exp(-np.sqrt(x**2))
    npp = integrate.quad(f_np, lim, 1.E4, limit = 100000, epsrel=1.e-8) #number density
    epp = integrate.quad(e_np, lim, 1.E4, limit = 100000, epsrel=1.e-8) #energy density
    return (npp[0]/(npl(z)), epp[0]/(npl(z)))

vphoton_ne = np.vectorize(photon_ne)

def pp_rate(z, qdm, mdm):
    rdm = alpha*hp/(2*np.pi*mdm*c)
    sigma_pp = alpha*qdm**4*rdm**2*((28/9)*np.log(2*kbev*Tcmb(z)/(mdm*1.e6/Mev))-218/27)
    if sigma_pp<0:
        return 0.
    else:
        return (vne(z)*sigma_pp*c)
    
vpprate = np.vectorize(pp_rate)

def pp_rate2(z, qdm, mdm):
    rdm = alpha*hp/(2*np.pi*mdm*c)
    sigma_pp = alpha*qdm**4*rdm**2*((28/9)*vphoton_ne(z, mdm)[1]-218/27*vphoton_ne(z, mdm)[0])
    return (vne(z)*sigma_pp*c)

def pp_rate3(z, qdm, mdm):
    rdm = alpha*hp/(2*np.pi*mdm*c)
    sigma_pp = np.pi*rdm**2
    return (vne(z)*sigma_pp*c)
    
def edm_prod(z, qdm, mdm):
    me_ev = (me/Mev)*1.E6
    mdm_ev = (mdm/Mev)*1.E6
    sigmaev = (np.pi*alpha**2*qdm**2/(me/Mev)**2)*np.sqrt(1-(mdm/me)**2)*(1+0.5*(mdm/me)**2)*(1/cm_Mev)**2
    return vneplus(z)*sigmaev*c

vedm_prod = np.vectorize(edm_prod)

# Proton number density
def n_p(z):
    if 10**z>25000:
        return xp(25000)*0.76*nb(z)
    else:
        return xp(10**z)*0.76*nb(z)
    
# He+ number density
def nhe_p(z):
    if 10**z>10**5/(kbev*2.725):
        return 0.
    elif 10**z>25000:
        return xHe_p(25000)*0.76*nb(z)
    else:
        return xHe_p(10**z)*0.76*nb(z)
    
# He2+ number density
def nhe_pp(z):
    if 10**z>10**5/(kbev*2.725):
        return 0.
    elif 10**z>25000:
        return xHe_pp(25000)*0.76*nb(z)
    else:
        return xHe_pp(10**z)*0.76*nb(z)

# Dimensionless frequency w.r.t. electron temperature
def xe(z, T_star):
    return T_star/Tg(z)

def gamma2(z, q):
    return np.log10((q**2)*1.579E5/Tg(z))

def u(z, T_star):
        return np.log10(xe(z, T_star))

# Gaunt factors
def gbr(z, T_star):
    return gff(u(z, T_star), gamma2(z, 1))*(xp(10**z)*0.76) + gff(u(z, T_star), gamma2(z, 1))*(xHe_p(10**z)*0.76) + 4*gff(u(z, T_star), gamma2(z, 2))*(xHe_pp(10**z)*0.76)

# Bremsstrahlung absorption coefficient
def rate_brem(z, T_star):
    if 0 < 10**z < 25000:
        return (ne(z)*sigma_T*c*alpha*nb(z)/np.sqrt(24*(np.pi**3)))*(me*c**2)**(7./2.)*(hp/(me*c))**3*gbr(z, T_star)*((1 - np.exp(-xe(z, T_star)))/xe(z, T_star)**3)*(kb*Tg(z))**(-7./2.)
    elif 10**z >= 25000:
        return 3.41e-11*((1+10**z)/(2.e6))**(5./2.)*((1 - np.exp(-xe(z, T_star)))/xe(z, T_star)**3)
    else:
        return 0.0

# Min redshift till which bremsstrahlung can erase the signal
def f_zbmin(z, T_star):
    return rate_brem(z, T_star) - H_z(z)

def zbmin(T_star):
    if np.sign(f_zbmin(np.log10(50.), T_star)) == np.sign(f_zbmin(np.log10(2.e6), T_star)):
        return 1.
    else:
        return 10**(bisect(f_zbmin, np.log10(50.), np.log10(2.e6), args=(T_star,),xtol=1.e-10))

# Rate of CMB heating of dark matter
def Rdm_heating(z, T_star, mdm, A10dm, a1, beta):
    return Rcomp(z, mdm, a1, A10dm, beta, Tcmb(z), T_star, Tcmb(z))[1]*(2./3.)*(T_star/Tcmb(z))*(T_star/Tcmb(z))*1/(1+(1./3.)*np.exp(T_star/Tcmb(z)))-1

# Min redshift till which collisions and heating are dominant
def zdmmin(T_star, mdm, A10dm, a1, beta):
    z0 = np.log10(50.)

    ######################## CMB heating DM #######################
    if np.sign(Rdm_heating(z_bb, T_star, mdm, A10dm, a1, beta))==np.sign(Rdm_heating(z0, T_star, mdm, A10dm, a1, beta)):
        sol1 = 0.
    else:
        sol1 = bisect(Rdm_heating, np.log10(50.), z_bb, args=((T_star, mdm, A10dm, a1, beta)), xtol=1.e-10)

    ######################## DM collisions causing heating of DM by CMB #######################
    if np.sign(C10(z_bb, mdm, a1, beta, Tcmb(z_bb))/H_z(z_bb)-1)==np.sign(C10(z0, mdm, a1, beta, Tcmb(z0))/H_z(z_bb)-1):
        sol2 = 0.
    else:
        sol2 = bisect(lambda z: C10(z, mdm, a1, beta, Tcmb(z))/H_z(z)-1 , np.log10(50.), z_bb, xtol=1.e-10)
    return (sol1, sol2)

# Dark matter energy density
def edm(mdm, z):
    return (3./2.)*ndm(z, mdm)*kb*Tcmb(z)

# contribution from mu parameter to u for dark matter temp in equilibrium with CMB in redshift range 2x10^6 to 10^5 (delE large y distortion=0)
def u_dm(mdm, T_star, z_lmin, z_lmax):
    zmin = 10**z_lmin
    zmax = 10**z_lmax
    fn = lambda zd: np.exp(xc(np.log10(zd)))*np.exp(-xc(np.log10(zd))/xcmb(np.log10(zd), T_star))
    if z_lmin<z_c:
        nint = integrate.quad(fn, 10**z_c, zmax, limit = 10000, epsrel=1.e-12)
        mupar = 1.4*(-edm(mdm, 3.)/epl(3.))*\
                (np.log((1+zmax)/(1+10**z_c))-(4./3.)*(aR/(bR*kb))*(2.726/T_star)*nint[0])
        zb_min = zbmin(T_star)
        if zmin<zb_min:
            zmin = zb_min
            #print(zmin)
        ypar = 0.25*(-edm(mdm, 3.)/epl(3.))*(np.log((1+10**z_c)/(1+zmin)))
    else:
        nint = integrate.quad(fn, zmin, zmax, limit = 10000, epsrel=1.e-12)
        mupar = 1.4*(-edm(mdm, 3.)/epl(3.))*\
                (np.log((1+zmax)/(1+zmin))-(4./3.)*(aR/(bR*kb))*(2.726/T_star)*nint[0])
        ypar = 0.
    return (mupar, ypar) 

# Solving for Tex+Tdm
def Tex_solver(T_star, mdm, alpha_A, a1, beta, teval, Tdm, Tex):
    solver = ode('cvode', lambda z, T, Tdot: f_Tex(z, T, Tdot, T_star, mdm, alpha_A, a1, beta), old_api=False, rtol=1.e-12, max_steps=1.e8)
    sol = solver.solve(teval, [Tex, Tdm])
    return sol

# Solving for Tdm only
def Tdm_solver(T_star, mdm, alpha_A, a1, teval, Tdm):
    solver = ode('cvode', lambda z, T, Tdot: f_Tdm(z, T, Tdot, T_star, mdm, alpha_A, a1), old_api=False, rtol=1.e-12, max_steps=1.e8)
    sol = solver.solve(teval, [Tdm])
    return sol

# Flattening the list
def flatten_list(inlist1, inlist2, inlist3):
    inlist12 = list(chain.from_iterable(inlist1))
    inlist22 = list(chain.from_iterable(inlist2))
    inlist32 = list(chain.from_iterable(inlist3))
    pairlist = np.round(np.transpose([inlist12]+[inlist22]+[inlist32]), 14).tolist()
    orderlist = []
    for x in pairlist:
        if not x in orderlist:
          orderlist+=[x]
    olist = np.transpose(orderlist)
    return (np.array(olist[0]), np.array(olist[1]), np.array(olist[2])) 

############################### Spectral distortions ################################################
    
# Comparing rates
def Rcomp(z, mdm, a1, A10dm, beta, Tdm, T_star, Texdm):
    C_H = C10(z, mdm, a1, beta, Tdm)*Texdm/(T_star*H_z(z))*(1+g10*np.exp(-T_star/Texdm))
    A_H = A10dm*Texdm/(T_star*H_z(z))*(1+g10*np.exp(-T_star/Texdm))/(1-np.exp(-T_star/Tcmb(z)))
    C_A = C_H/A_H
    return (C_H, A_H, C_A)

# DM global signal
def dm_signal(f, T_star, mdm, eps, a1, beta):
    mevdm = (mdm/Mev)
    alpha_A = eps**2*(T_star/0.068)**3*(0.5/mevdm)**2
    A10dm = alpha_A*a10H
    zdec = zdecal(T_star, mdm, alpha_A)
    
    # print(T_star, mevdm, a1, eps, beta, zdec)

    if eps == 0. or a1==0:
        print('zero electric charge')
        return ((0., 0., 0.), (0., 0.), (0., 0.)) #(np.zeros(f.shape), np.zeros(f.shape)) #no signal
    elif zdec==0.:
        print('DM coupled after z_eq')
        return ((10., 10., 10.), (10., 10.), (10., 10.)) # penalise??  (np.ones(f.shape), np.ones(f.shape)) #not allowed!
    else:
        # Range of z for T_ex evaluation 
        zmin = np.log10(z_hrec)
        zmax = np.log10(zdec)
        if zmax<z_bb:
            mu_i, y_i = u_dm(mdm, T_star, z_bb, zmax)
        else:
            mu_i = 0.
            y_i = 0.
            
        Tdm = Tdmad(zmax, zmax)
        ch, ah, ca = Rcomp(zmax, mdm, a1, A10dm, beta, Tdm, T_star, Tdm)
        initial_rates = (ch, ah, ca)
        
        z_arr = []
        Tex_arr = []
        Tdm_arr=[]
        Tex = Tdm
        if ah>1 and ch>1:   # collisions and radiative transitions above hubble
            if ca>1:        # collisions > radiative transitions
                if ca>1.e4 and ch>1.e4: # collisions >> radiative transitions and collisions >> Huuble
                    while(zmax-1>=np.log10(zmin)):
                        teval = np.linspace(np.log10(10**zmax), np.log10(10**(zmax-0.01)), num=4)
                        solution = Tdm_solver(T_star, mdm, alpha_A, a1, teval, Tdm)
                        z_arr.append(solution.values.t)
                        ysol = solution.values.y
                        Tex_arr.append(ysol[:,0])
                        Tdm_arr.append(ysol[:,0])
                        Tex = Tex_arr[-1][-1]
                        Tdm = Tdm_arr[-1][-1]
                        zmax = z_arr[-1][-1]
                        ch, ah, ca = Rcomp(zmax, mdm, a1, A10dm, beta, Tdm, T_star, Tex)
                        if ca<1.e4 and ch<1.e4: # C and A start becoming comparable
                            break
                # collisions > radiative transitions ... comparable rates
                zc = zmax
                zc0 = zmax
                while(zc-1>=np.log10(zmin)):
                    teval = np.linspace(np.log10(10**zc), np.log10(10**(zc-1.)), num=100) #evaluation points in redshift for T_ex
                    solution = Tex_solver(T_star, mdm, alpha_A, a1, beta, teval, Tdm, Tex)
                    z_arr.append(solution.values.t)
                    Tex_arr.append(solution.values.y[:,0])
                    Tdm_arr.append(solution.values.y[:,1])
                    zc = z_arr[-1][-1]
                    Tdm = Tdm_arr[-1][-1]
                    Tex = Tex_arr[-1][-1]
                    if abs(zc - zc0)/zc <=1.e-15:
                        zc = zc-0.001
                    ch, ah, ca = Rcomp(zc, mdm, a1, A10dm, beta, Tdm, T_star, Tex)
                    zc0 = z_arr[-1][-1]
                    if ah>1.e4 and ca<1.e-4 and zc0<z_c: # A>1000C and A>1000H: radiation dominates fully no signal
                        break
            else: # ca<1  collisions < radiative transitions
                if ah>1.e5 and ca<1.e-5: # A>1000C and A>1000H: radiation dominates fully and mu+y distortion contributes to u 
                    # if zmax > z_bb:
                    #    zumax = z_bb
                    # else:
                    zumax = zmax
                    
                    zm1, zm2 = zdmmin(T_star, mdm, A10dm, a1, beta)
                    # print(zm1, zm2)
                    if zm1!=0 and zm2==0:
                        zumin = zm1
                    elif zm1==0 and zm2!=0:
                        zumin = zm2
                    else:
                        if zm1>zm2:
                            zumin = zm1
                        else:
                            zumin = zm2
                        
                    if zm1==zm2==0 or zumin>zumax:
                        mudm_par = 0.
                        ydm_par = 0.
                    else:
                        mudm_par, ydm_par = u_dm(mdm, T_star, zumin, zumax)
                    return (initial_rates, (mudm_par+mu_i, 0.), (ydm_par+y_i, 0.)) #(np.zeros(f.shape), np.zeros(f.shape)) #no signal
                else: 
                # collisions < radiative transitions near Hubble
                # print(zmax)
                    zc = zmax
                    zc0 = zmax
                    while(zc-1>=np.log10(zmin)):
                        teval = np.linspace(np.log10(10**zc), np.log10(10**(zc-1.)), num=100) #evaluation points in redshift for T_ex
                        solution = Tex_solver(T_star, mdm, alpha_A, a1, beta, teval, Tdm, Tex)
                        z_arr.append(solution.values.t)
                        Tex_arr.append(solution.values.y[:,0])
                        Tdm_arr.append(solution.values.y[:,1])
                        zc = z_arr[-1][-1]
                        Tdm = Tdm_arr[-1][-1]
                        Tex = Tex_arr[-1][-1]
                        if abs(zc - zc0)/zc <=1.e-15:
                            zc = zc-0.001
                        ch, ah, ca = Rcomp(zc, mdm, a1, A10dm, beta, Tdm, T_star, Tex)
                        # print(zc, ch, ah*(2./3.)*(T_star/Tex)*(T_star/Tdm)*1/(1+(1/3)*np.exp(T_star/Tex)), ca)
                        zc0 = z_arr[-1][-1]
                        if ah>1.e4 and ca<1.e-4 and zc0<z_c: # A>1000C and A>1000H: radiation dominates fully no signal
                            break
        elif ah>1 and ch<1:
            if ah>1.e5 and ca<1.e-5: # A>1000C and A>1000H: radiation dominates fully and mu+y distortion contributes to u 
                    zumax = zmax
                    
                    zm1, zm2 = zdmmin(T_star, mdm, A10dm, a1, beta)
                    # print(zm1, zm2)
                    if zm1!=0 and zm2==0:
                        zumin = zm1
                    elif zm1==0 and zm2!=0:
                        zumin = zm2
                    else:
                        if zm1>zm2:
                            zumin = zm1
                        else:
                            zumin = zm2
                        
                    if zm1==zm2==0 or zumin>zumax:
                        mudm_par = 0.
                        ydm_par = 0.
                    else:
                        mudm_par, ydm_par = u_dm(mdm, T_star, zumin, zumax)
                    return (initial_rates, (mudm_par+mu_i, 0.), (ydm_par+y_i, 0.)) 
            else:
                # radiative transitions > collisions ... comparable rates
                zc = zmax
                zc0 = zmax
                while(zc-1>=np.log10(zmin)):
                    teval = np.linspace(np.log10(10**zc), np.log10(10**(zc-1.)), num=100) #evaluation points in redshift for T_ex
                    solution = Tex_solver(T_star, mdm, alpha_A, a1, beta, teval, Tdm, Tex)
                    z_arr.append(solution.values.t)
                    Tex_arr.append(solution.values.y[:,0])
                    Tdm_arr.append(solution.values.y[:,1])
                    zc = z_arr[-1][-1]
                    Tdm = Tdm_arr[-1][-1]
                    Tex = Tex_arr[-1][-1]
                    if abs(zc - zc0)/zc <=1.e-15:
                        zc = zc-0.001
                    ch, ah, ca = Rcomp(zc, mdm, a1, A10dm, beta, Tdm, T_star, Tex)
                    zc0 = z_arr[-1][-1]
                    if ah>1.e4 and ca<1.e-4 and zc0<z_c: # A>1000C and A>1000H: radiation dominates fully no signal
                        break
        elif ah<1 and ch>1:
            if ch>1.e4 and ca>1.e4: # collisions >> Hubble and collisions >> radiative transitions
                while(zmax-1>=np.log10(zmin)):
                    teval = np.linspace(np.log10(10**zmax), np.log10(10**(zmax-0.01)), num=4)
                    solution = Tdm_solver(T_star, mdm, alpha_A, a1, teval, Tdm)
                    z_arr.append(solution.values.t)
                    ysol = solution.values.y
                    Tex_arr.append(ysol[:,0])
                    Tdm_arr.append(ysol[:,0])
                    Tex = Tex_arr[-1][-1]
                    Tdm = Tdm_arr[-1][-1]
                    zmax = z_arr[-1][-1]
                    ch, ah, ca = Rcomp(zmax, mdm, a1, A10dm, beta, Tdm, T_star, Tex)
                    if ch<1.e4 and ca<1.e4: # C and A start becoming comparable
                        break
            # collisions > Hubble and rad transitions < Hubble ... comparable rates
            zc = zmax
            zc0 = zmax
            while(zc-1>=np.log10(zmin)):
                teval = np.linspace(np.log10(10**zc), np.log10(10**(zc-1.)), num=100) #evaluation points in redshift for T_ex
                solution = Tex_solver(T_star, mdm, alpha_A, a1, beta, teval, Tdm, Tex)
                z_arr.append(solution.values.t)
                Tex_arr.append(solution.values.y[:,0])
                Tdm_arr.append(solution.values.y[:,1])
                zc = z_arr[-1][-1]
                Tdm = Tdm_arr[-1][-1]
                Tex = Tex_arr[-1][-1]
                if abs(zc - zc0)/zc <=1.e-15:
                    zc = zc-0.001
                ch, ah, ca = Rcomp(zc, mdm, a1, A10dm, beta, Tdm, T_star, Tex)
                zc0 = z_arr[-1][-1]
                if ah>1.e4 and ca<1.e-4 and zc0<z_c: # A>1000C and A>1000H: radiation dominates fully no signal
                    break
        else: # ah<1 and ch<1
            # collisions, rad transitions, Hubble ... comparable rates
            zc = zmax
            zc0 = zmax
            while(zc-1>=np.log10(zmin)):
                teval = np.linspace(np.log10(10**zc), np.log10(10**(zc-1.)), num=100) #evaluation points in redshift for T_ex
                solution = Tex_solver(T_star, mdm, alpha_A, a1, beta, teval, Tdm, Tex)
                z_arr.append(solution.values.t)
                Tex_arr.append(solution.values.y[:,0])
                Tdm_arr.append(solution.values.y[:,1])
                zc = z_arr[-1][-1]
                Tdm = Tdm_arr[-1][-1]
                Tex = Tex_arr[-1][-1]
                if abs(zc - zc0)/zc <=1.e-15:
                    zc = zc-0.001
                ch, ah, ca = Rcomp(zc, mdm, a1, A10dm, beta, Tdm, T_star, Tex)
                zc0 = z_arr[-1][-1]
                if ah>1.e4 and ca<1.e-4 and zc0<z_c: # A>1000C and A>1000H: radiation dominates fully no signal
                    break

      
        if len(z_arr)==0:
            print(ah, ch, ca)
            print('problem!')
            return (initial_rates, (2., 2.), (2., 2.)) #(np.zeros(f.shape), np.zeros(f.shape))
        else:
            flat_z, flat_Tex, flat_Tdm = flatten_list(z_arr, Tex_arr, Tdm_arr)
            T_exint = interpolate.interp1d(flat_z , flat_Tex, kind='cubic')

################################ mu/y signal ###########################
      
        f_mu = lambda zd: (1.4005704)*(10**zd)/(H_z(zd)*(1+10**zd)*np.log10(np.exp(1)))*((edot(zd, T_exint(zd), T_star, mdm, alpha_A)/epl(zd))\
                    -(4./3.)*np.exp(xc(zd))*np.exp(-xc(zd)/xcmb(zd, T_star))*(ndot(zd, T_exint(zd), T_star, mdm, alpha_A)/npl(zd))) \
                    if xcmb(zd, T_star)>1 else (1.4005704)*(10**zd)/(H_z(zd)*(1+10**zd)*np.log10(np.exp(1)))*((edot(zd, T_exint(zd), T_star, mdm, alpha_A)/epl(zd))\
                    -(4./3.)*(ndot(zd, T_exint(zd), T_star, mdm, alpha_A)/npl(zd)))
         
        if flat_z[0]>z_bb:
            zmu_max = z_bb
        else:
            zmu_max = flat_z[0]
        
        if zmu_max>z_c and flat_z[-1]<z_c:
            zmu_min = z_c
        elif zmu_max>flat_z[-1] and flat_z[-1]>z_c:
            zmu_min = flat_z[-1]
        else:
            zmu_max = 0.
            zmu_min = 0.

        if zmu_max == zmu_min:
            mu_dm = (0., 0.)
        else:
            # print(zmu_min , zmu_max)
            mu_dm = integrate.quad(f_mu, zmu_min, zmu_max, limit=1000000, epsrel=1.e-14)
        
        f_y = lambda zd: (0.25)*10**zd/(H_z(zd)*(1+10**zd)*np.log10(np.exp(1)))*(edot(zd, T_exint(zd), T_star, mdm, alpha_A)/epl(zd))
        zb_min = np.log10(zbmin(T_star))

        if flat_z[0]>z_c:
            zy_max = z_c
        else:
            zy_max = flat_z[0]
        
        if zy_max>zb_min and flat_z[-1]<zb_min:
            zy_min = zb_min
        elif zy_max>flat_z[-1] and flat_z[-1]>zb_min:
            zy_min = flat_z[-1]
        else:
            zy_max = 0.
            zy_min = 0.

        if zy_max == zy_min or zb_min==0:
            y_dm = (0., 0.)
        else:
            y_dm = integrate.quad(f_y, zy_min, zy_max, limit = 10000000, epsrel=1.e-14)
        return (initial_rates, (mu_dm[0]+mu_i, mu_dm[1]), (y_dm[0]+y_i, y_dm[1])) 



# COBE limits
cobe_mu = 9.E-5
cobe_y = 1.5E-5

# COBE limits
pixie_mu = 5.2E-8
pixie_y = 6.6E-9

# DM model parameters
mdm_p = 1.E2   #MeV
# delE_p = 1.E3  #eV

# Path to the file
file = '/home/anoma/final_codes/mu_y_dm/mdm_'+str(mdm_p)+'_MeV_f.txt'

epsarr = np.logspace(-8.0, -3.0, num=200)
delEarr = np.logspace(-4.0, 4.0, num=200)
mu_par = np.zeros(epsarr.shape)
y_par = np.zeros(epsarr.shape)

# Open the file for writing
with open(file, 'w') as f:
    f.write('DM mass='+str(mdm_p)+' MeV\n')
    f.write('epsilon\t delta_E*\t C/H\t A/H\t C/A\t mu param\t mu_err\t y param\t y_err\t delE/E\n')
    for i in range(0, epsarr.shape[0]):
        for j in range(0, delEarr.shape[0]):
            print(epsarr[i], delEarr[j])
            f.write('%E\t'%epsarr[i])
            f.write('%E\t'%delEarr[j])
            rate, mu, y = dm_signal(freq, delEarr[j]/kbev, mdm_p*Mev, eps=epsarr[i], a1=2.0, beta=0.5)
            delEbyE = 0.71429*mu[0]+4*y[0]
            f.write('%E\t'%rate[0])
            f.write('%E\t'%rate[1])
            f.write('%E\t'%rate[2])
            f.write('%E\t'%mu[0])
            f.write('%E\t'%mu[1])
            f.write('%E\t'%y[0])
            f.write('%E\t'%y[1])
            f.write('%E\n'%delEbyE)

