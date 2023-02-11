import numpy as np
from scipy.integrate import simps
from Project.constants import Constants as c
from scipy import special

pi = c.pi
G = c.G
rho_crit = c.rho_crit
kpc = c.kpc
yr = c.yr
Gyr = c.Gyr
vel_light = c.vel_light

#### general
def chkary(a):
	if not isinstance(a,np.ndarray):
		if isinstance(a,list):
			return np.array(a)
		else:
			return np.array([a])
	else:
		return a
    
def log(x):
	x = chkary(x)
	if not np.all([x <= 0]) or np.isnan(x):
		if len(x) == 1:
			return np.log10(x)[0]
		else:
			return np.log10(x)
	else:
		return np.ones(len(x))*-np.inf	
    
def ln(x):
	x = chkary(x)
	if not np.all([x <= 0]) or np.all(np.isnan(x)):
		if len(x) == 1:
			return np.log(x)[0]
		else:
			return np.log(x)
	else:
		return np.ones(len(x))*-np.inf
    
def E(x):
	return 10**x

def po(a,b):
	return a**b

def frac(a,b):
	try:
		if a == np.inf or a == -np.inf:
			return a
		elif np.isnan(a):
			return np.inf
		elif b == 0:
			return (a/np.abs(a))*np.inf
		else:
			return a/b
	except:
		return a/b
    
def tanh(x):
	x = chkary(x)
	indx1 = np.nonzero(x > 30)
	x[indx1] = 30.
	A = (np.exp(2.*x) - 1.)/(np.exp(2.*x) + 1.)
	if len(A) == 1:
		return A[0]
	else:
		return A
    
def arctan(x):
	x = chkary(x)
	indx1 = np.nonzero(x > 1e3)
	indx2 = np.nonzero(x < -1e3)
	x[indx1] = pi/2.
	x[indx2] = -pi/2.
	A = np.arctan(x)
	if len(A) ==1:
		return A[0]
	else:
		return A
    
####  Numerical methods

def Falsepos(f,a,b,max_iter,args = [],eps=1e-5,flag = ''):
	i = 0
	success = False
	while (i <= max_iter):
		c = frac(a*f(b,*args) - b*f(a,*args),f(b,*args)-f(a,*args))
		fc = f(c,*args)

		if i != 0:
			if np.abs(fc-fd) <= eps:
				if np.abs(fc) <= eps:
					success = True
					break
		
		if fc == 0.:
			break
		elif fc * f(a,*args) < 0:
			b = c
		else:
			a = c
		fd = fc	
		i += 1
	return {'iterations':i,'root':c,'success':success,'flag':flag}

def NewtonRap(f,devf,x):
	h = f(x)/devf(x)	
	while abs(h) >= 1e-6:
		h = f(x)/devf(x)
		x = x - h
	return x

######## DARK MATTER COMPONENT

def r200_nfw(logm200):
	M200 = 10**(logm200-10.)
	R200 = po( 1e10*3.*M200/(4.*pi*200.*rho_crit) , 1./3.)
	return R200

def rs_nfw(logm200,c200):
    m200 = 10**logm200
    return (3.*m200/(200.*4.*np.pi*c.rho_crit))**(1./3.)/c200

def c200_nfw(logm200,rs):
	r200 = r200_nfw(logm200)
	return r200/rs

def rho0_nfw(logm200,c200): #Msun/kpc^3
	rs = rs_nfw(logm200,c200)
	g = ln(1. + c200) - frac(c200,(1. + c200))
	m200 = E(logm200 - 10.)
	return frac(1e10*m200,4.*pi*po(rs,3)*g)

def mass_nfw(logm200,c200,R):
	rs = rs_nfw(logm200,c200)
	rho0 = rho0_nfw(logm200,c200)*1e-10
	gr = ln(frac(rs+R,rs)) - frac(R,rs+R)
	return 4.*pi*rho0*gr*po(rs,3)*1e10

def v_nfw(logm200,c200,R):
	mr = mass_nfw(logm200,c200,R)
	return po(G*mr/R,0.5)

def density_nfw(logm200,c200,r):
	rs = rs_nfw(logm200,c200)
	rho0 = rho0_nfw(logm200,c200)
	x = frac(r,rs)
	return frac(rho0,x*po(1. + x,2))

#### core profile (logm200, c200, r1)

def rhob_rb_burk(logm200,c200,r1):
	rs = rs_nfw(logm200,c200)
	if r1 == np.inf or np.isnan(r1) or r1 <= 0:
		return np.inf,0
	y = frac(r1,rs)
	gg = 4.*(ln(1. + y) - frac(y,1.+y))*frac(po(1. + y,2),po(y,2))
	rho0 = rho0_nfw(logm200,c200)
	def func(x):
		if x == np.inf or np.isnan(x) or x <= 0 or x == -np.inf:
			return 1
		else:
			ff = ( ln((1. + po(x,2))*po(1. + x,2)) - 2.*arctan(x))*(1. + x)*(1. + po(x,2))/po(x,3)
			return ff - gg
	X = Falsepos(func,1.,10.,70)
	if X['success'] == True:
		X = X['root']
	else:
		X = np.inf
	rb = r1/X
	rhob = rho0*frac((1. + X)*(1. + po(X,2)),y*po(1. + y,2))
	return rhob,rb

def mass_burk(rhob,rb,R):
	kk = pi*rhob*po(rb,3)
	xmax = R/rb
	return kk * (ln((1. + po(xmax,2))*po(1. + xmax,2)) - 2.*arctan(xmax))

def density_burk(rhob,rb,r):
	y = frac(r,rb)
	return frac(   rhob , (1.+y)*(1. + po(y,2))    )

def mass_sidm(logm200,c200,r1,r):
	rhob,rb = rhob_rb_burk(logm200,c200,r1)
	r = chkary(r)
	if len(r) == 1:
		if r <= r1:
			return mass_burk(rhob,rb,r[0])
		else:
			return mass_nfw(logm200,c200,r[0])
	else:
		indx1 = np.nonzero(r <= r1)
		indx2 = np.nonzero(r >  r1)
		m1 = mass_burk(rhob,rb,r[indx1])
		m2 = mass_nfw(logm200,c200,r[indx2])
		
		mass = np.zeros(r.shape)
		mass[indx1] = m1
		mass[indx2] = m2
		return mass
    
def v_sidm(logm200,c200,r1,R):
	m = mass_sidm(logm200,c200,r1,R)
	return po(m*G/R,0.5)

def density_sidm(logm200,c200,r1,r):
	rhob,rb = rhob_rb_burk(logm200,c200,r1)
	r = chkary(r)
	if len(r) ==1:
		if r <= r1:
			return density_burk(rhob,rb,r[0])
		else:
			return density_nfw(logm200,c200,r[0])
	else:
		indx1 = np.nonzero(r <= r1)
		indx2 = np.nonzero(r > r1)
		d1 = density_burk(rhob,rb,r[indx1])
		d2 = density_nfw(logm200,c200,r[indx2])
		
		dens = np.zeros(r.shape)
		dens[indx1] = d1
		dens[indx2] = d2
		return dens
    
######## BARYONIC COMPONENT	

def v_disk(vg,vd,yd,r):
	return po( vg*np.abs(vg)+yd*vd*np.abs(vd) ,0.5)

def v_bulge(vb,yb,r):
	return po(yb*vb*np.abs(vb),0.5)

def v_photomeric(vg,vd,vb,yd,yb,r):
	return po(np.abs(vg)*vg + yd*vd*np.abs(vd) + yb*vb*np.abs(vb),0.5)

def mass_exp(logmstar,rd,rmax):
	sig = 2.*pi*E(logmstar - 10.)/po(rd,2)
	return 2.*pi*sig*po(rd,2)*(1. - (np.exp(-1.*rmax/rd) * (1. + rmax/rd)))*1e10

def sig_exp(logmstar,rd):
	return E(logmstar)/(2.*pi*po(rd,2))

def v_exp(logmstar,rd,r):
	mstar = E(logmstar-10.)
	y = frac(r,2.*rd)
	try:
		vsq = 2.*G*frac(mstar,rd)*po(y,2)* (special.i0(y)*special.k0(y) - special.i1(y)*special.k1(y))*1e10
	except:
		print (logmstar, rd, special.i0(y)*special.k0(y))
	return po(vsq,0.5)

#### SCALING RELATIONS
def lmstar_moster_13(logm200,fb = c.fb):
	m200 = 10**logm200/(1. - fb)
	N10 = 0.0351
	M10 = 11.590
	B10 = 1.376
	C10 = 0.608
	M1 = 10**(M10 - 10.)
	M = m200*1e-10
	mm = M/M1
	ratio = 2.*N10/(mm**(-1.*B10) + mm**C10)
	return np.log10(ratio*m200)

def f_behroozi(x,alp,delta,gam):
    fx = -np.log10(10**(alp*x) + 1.) + delta*(((np.log10(1. + np.exp(x)))**gam)/(1 + np.exp(10**(-x))))
    return fx

def lmstar_behroozi_13(logm200,fb = c.fb):
    eps = 10**-1.777
    M1 = 10**11.514
    alp_behroozi = -1.412
    delta_behroozi = 3.508
    gam_behroozi = 0.316
    M200 = 10**logm200/(1. - fb)
    x = logm200 - 11.514
    logmstar = np.log10(eps*M1) + f_behroozi(x,alp_behroozi,delta_behroozi,gam_behroozi) - f_behroozi(0.,alp_behroozi,delta_behroozi,gam_behroozi)
    return logmstar

def lmstar_behroozi_19(logm200,fb = c.fb):
    logm200 = logm200 - np.log10(1.-fb)
    eps = -1.435
    alpha = 1.963
    beta = 0.482
    gamma = 10**-1.034
    delta = 0.411
    logM1 = 12.035
    x = logm200 - logM1
    logmstar = eps - np.log10(10**(-alpha*x) + 10**(-beta*x)) + gamma*(np.exp(-0.5*(x/delta)**2)) + logM1
    return logmstar

def lmstar_behroozi_nbp(logm200,fb = c.fb):
    logm200 = logm200 - np.log10(1.-fb)
    if isinstance(logm200, (float,int)):
        logm200 = [logm200]
    logmstar = []
    shm = lambda x: 10**lmstar_behroozi_13(x)/10**x
    for lm in logm200:
        if 11.3 <= lm <= 12.95:
            logmstar.append(np.log10(shm(11.3)*10**lm))
        else:
            logmstar.append(lmstar_behroozi_13(lm))
    if len(logmstar) == 1:
        return logmstar[0]
    else:
        return np.array(logmstar)

def lc200_SR(lm200,fb = c.fb,h = c.h): #SR-ScalingRelation
	"""
	returns log of c200 from duccon and macccio relation for an observed 
	galaxy mass of lm200. This m200 is then scale up to account for the fact
	that the relation was obtained for dark matter only simulations.
	"""
	m200 = 10**(lm200-12.)
	m200 = m200/(1. - fb)
	return 0.905 - 0.101*(np.log10(m200*h))

def lm200_SR(lc200,fb = c.fb,h = c.h):
	"""
	Returns log10 of m200 from the duccon and maccio relation, given a logc200.
	"""
	return 0.905 - lc200 - np.log10(h) + np.log10(1.-fb) + 12