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


#z_gauss,w_gauss = np.loadtxt("/home/manush/Desktop/Project_multinest/required_files/gauss80.dat",unpack = True)


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

"""
DO NOT USE THIS. UNITS ARE WRONG. MULITPLY WITH 1e-20.
def Density_to_potential(dens_func,B,args = (),A =0.):
	# IN ERGS
	z,w = z_gauss,w_gauss
	inte1 = 0.
	for i in range(80):
		y1 = (B-A)*(z[i] + 1.)/2.0 + A
		inte2 = 0.
		for j in range(80):
			y2 = (y1 - A)*(z[j] + 1.)/2.0 + A
			inte2 = inte2 + (((y1 - A)/2.0) * w[j] * dens_func(y2, *args) * (y2**2) )
		inte1 = inte1 + ((B-A)/2.0) * (w[i]*dens_func(y1,*args)*inte2*y1)
	pt = inte1 * (4.*pi)**2
	return pt*1.989*1e63*G
"""	

######## DARK MATTER COMPONENT

####  NFW (logm200,rs)

def r200_nfw(logm200):
	M200 = 10**(logm200-10.)
	R200 = po( 1e10*3.*M200/(4.*pi*200.*rho_crit) , 1./3.)
	return R200

def c200_nfw(logm200,rs):
	r200 = r200_nfw(logm200)
	return r200/rs

def rho0_nfw(logm200,rs): #Msun/kpc^3
	c200 = c200_nfw(logm200,rs)
	g = ln(1. + c200) - frac(c200,(1. + c200))
	m200 = E(logm200 - 10.)
	return frac(1e10*m200,4.*pi*po(rs,3)*g)

def mass_nfw(logm200,rs,R):
	rho0 = rho0_nfw(logm200,rs)*1e-10
	gr = ln(frac(rs+R,rs)) - frac(R,rs+R)
	return 4.*pi*rho0*gr*po(rs,3)*1e10

def v_nfw(logm200,rs,R):
	mr = mass_nfw(logm200,rs,R)
	return po(G*mr/R,0.5)

def density_nfw(logm200,rs,r):
	rho0 = rho0_nfw(logm200,rs)
	x = frac(r,rs)
	return frac(rho0,x*po(1. + x,2))
  
def potential_nfw(logm200,rs,r):
  rho0 = rho0_nfw(logm200,rs)
  return -4*pi*G*rho0*(rs**3)*np.log(1 + (r/rs))/r

####  sidmNFW (logm200,rs,r1)

def rhob_rb_burk(logm200,rs,r1):
	if r1 == np.inf or np.isnan(r1) or r1 <= 0:
		return np.inf,0
	y = frac(r1,rs)
	gg = 4.*(ln(1. + y) - frac(y,1.+y))*frac(po(1. + y,2),po(y,2))
	rho0 = rho0_nfw(logm200,rs)
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

def potential_burk(rhob,rb,r):
    if isinstance(r, (float,int)):
      r = [r]
    r = np.concatenate([r,[1e7]])
    Pote = []
    for r_ in r:
        r__ = np.logspace(-2,np.log10(r_),400)
        pote = np.trapz(mass_burk(rhob,rb,r__)/(r__**2),r__)
        Pote.append(pote)
    Pote = np.array(Pote)*G
    result = Pote[:-1] - Pote[-1]
    if len(result) == 1: return result[0]
    else: return result
  
def density_burk(rhob,rb,r):
	y = frac(r,rb)
	return frac(   rhob , (1.+y)*(1. + po(y,2))    )

def mass_sidm(logm200,rs,r1,r):
	rhob,rb = rhob_rb_burk(logm200,rs,r1)
	r = chkary(r)
	if len(r) == 1:
		if r <= r1:
			return mass_burk(rhob,rb,r[0])
		else:
			return mass_nfw(logm200,rs,r[0])
	else:
		indx1 = np.nonzero(r <= r1)
		indx2 = np.nonzero(r >  r1)
		m1 = mass_burk(rhob,rb,r[indx1])
		m2 = mass_nfw(logm200,rs,r[indx2])
		
		mass = np.zeros(r.shape)
		mass[indx1] = m1
		mass[indx2] = m2
		return mass
			

	
def v_sidm(logm200,rs,r1,R):
	m = mass_sidm(logm200,rs,r1,R)
	return po(m*G/R,0.5)

def v_burk(rhob,rb,r):
    m = mass_burk(rhob,rb,r)
    return po(m*G/r,0.5)
    
def density_sidm(logm200,rs,r1,r):
	rhob,rb = rhob_rb_burk(logm200,rs,r1)
	r = chkary(r)
	if len(r) ==1:
		if r <= r1:
			return density_burk(rhob,rb,r[0])
		else:
			return density_nfw(logm200,rs,r[0])
	else:
		indx1 = np.nonzero(r <= r1)
		indx2 = np.nonzero(r > r1)
		d1 = density_burk(rhob,rb,r[indx1])
		d2 = density_nfw(logm200,rs,r[indx2])
		
		dens = np.zeros(r.shape)
		dens[indx1] = d1
		dens[indx2] = d2
		return dens
      
def potential_sidm(lm,rs,rc,r):
    if isinstance(r, (float,int)):
      r = [r]
    r = np.concatenate([r,[1e7]])
    Pote = []
    for r_ in r:
        r__ = np.logspace(-2,np.log10(r_),400)
        pote = np.trapz(mass_sidm(lm,rs,rc,r__)/(r__**2),r__)
        Pote.append(pote)
    Pote = np.array(Pote)*G
    result = Pote[:-1] - Pote[-1]
    if len(result) == 1: return result[0]
    else: return result
  
#### kaplinghat SIDM 
# def density_kap(logm200,)
    
  

#### coreNFW (logm200,rs,rc)

def shallow(logm200,rs,k = 0.04):
	Mrs = mass_nfw(logm200,rs,rs)
	tdyn = 2.0*pi*po(frac(po(rs,3),G*Mrs),0.5)
	tdyn = tdyn*kpc/yr
	tdyn = tdyn/Gyr
	tSF = 14.
	q = k*tSF/tdyn
	return tanh(q)

def mass_cnfw(logm200,rs,rc,r):
	n = shallow(logm200,rs)
	rho0 = rho0_nfw(logm200,rs)
	mnfw = mass_nfw(logm200,rs,r)
	f = tanh(r/rc)
	return mnfw*po(f,n)

def v_cnfw(logm200,rs,rc,R):
	try:
		M = []
		for r in R:
			mr = mass_cnfw(logm200,rs,rc,r)
			M.append(mr)
		M = np.array(M)
	except:
		M = mass_cnfw(logm200,rs,rc,R)
	return po(G*M/R,0.5)

def density_cnfw(logm200,rs,rc,r):
	n = shallow(logm200,rs)
	rho0 = rho0_nfw(logm200,rs)
	mnfw = mass_nfw(logm200,rs,r)
	f = tanh(r/rc)
	term1 = density_nfw(logm200,rs,r)*po(f,n)
	term2 = n * mnfw * po(f,n-1.) * frac(1. - po(f,2),4.*pi*rc*po(r,2))
	return term1 + term2
  
######## DC14

def abc_dc14(X):
  # if (X < -4.1) or (X > -1.3):
  #   return 1,3,1
  # else:
    a = 2.94 - np.log10((10**(X + 2.33))**(-1.08) + (10**(X+2.33))**2.29)
    b = 4.23 + 1.34*X + 0.26*(X**2)
    c = -0.06 + np.log10((10**(X+2.56))**(-0.68) + (10**(X+2.56)))
    return a,b,c
  
def rho0_dc14(logm200,rs,a,b,c,r200):
    r = np.logspace(-2,np.log10(r200),200)
    integrand = r**2/((r/rs)**c*(1 + (r/rs)**a)**((b-c)/a))
    integral = np.trapz(integrand, r)
    return 10**logm200/(4*np.pi*integral)
  
def Csph_by_Cdm(X):
    return (1.0 + 0.00003*np.exp(3.4*(X+4.5)))
  
def rs_dc14(logm200,c200,X,a,b,c):
    csph = c200*Csph_by_Cdm(X)
    r_2 = r200_nfw(logm200)/csph
    return r_2*((2 - c)/(b-2))**(-1./a)

def density_dc14(logm200,logmstar,rs,r):
    """
    rs here is rs_dc14 different from NFW rs
    """
    X = logmstar - logm200
    a,b,c = abc_dc14(X)
    r200 = r200_nfw(logm200)
    rho0 = rho0_dc14(logm200,rs,a,b,c,r200)
    return rho0/((r/rs)**c*(1 + (r/rs)**a)**((b-c)/a))
  
def mass_dc14(logm200,logmstar,rs,r):
    """
    rs here is rs_dc14 different from NFW rs
    """
    single = False
    if isinstance(r,(float,int)):
      single = True
      r = [r]
    X = logmstar - logm200
    a,b,c = abc_dc14(X)
    r200 = r200_nfw(logm200)
    rho0 = rho0_dc14(logm200,rs,a,b,c,r200)
    M = []
    for ri in r:
      r_ = np.logspace(-2,np.log10(ri),200)
      integrand = r_**2/(((r_/rs)**c)*(1+(r_/rs)**a)**((b-c)/a))
      integral = np.trapz(integrand,r_)
      M.append(4*np.pi*rho0*integral)
    if single: return M[0]
    else: return np.array(M)
  
def v_dc14(logm200,logmstar,rs,r):
    """
    rs here is rs_dc14 different from NFW rs
    """
    m = mass_dc14(logm200,logmstar,rs,r)
    return np.sqrt(G*m/r)
    

######## BARYONIC COMPONENT	

def v_disk(vg,vd,yd,r):
	return po( vg*np.abs(vg)+yd*vd*np.abs(vd) ,0.5)

def v_bulge(vb,yb,r):
	return po(yb*vb*np.abs(vb),0.5)

def v_photomeric(vg,vd,vb,yd,yb,r):
	return po(np.abs(vg)*vg + yd*vd*np.abs(vd) + yb*vb*np.abs(vb),0.5)

def mass_bary(vg,vd,vb,yd,yb,r):
	vd = v_disk(vg,vd,yd,r)
	vb = v_bulge(vb,yb,r)
	v_bary = po(vd**2 + vb**2,0.5)
	M = po(v_bary,2)*r #/G is there, but put in last step
	Mbary = [M[0]]
	for i in range(len(M)-1):
		Mbary.append(np.abs(M[i+1]-M[i]))
	Mbary = np.array(Mbary)
	return np.sum(Mbary)/G

def mass_star(vd,vb,yd,yb,r):
	v = np.sqrt(yd*vd*np.abs(vd) + yb*vb*np.abs(vb))
	M = (v**2)*r
	Mstar = [M[0]]
	for i in range(len(M)-1):
		Mstar.append(np.abs(M[i+1] - M[i]))
	Mstar = np.array(Mstar)
	return np.sum(Mstar)/G

def SHM_radius(Mstar,R,vd,vb,yd,yb = 0.):
	"""
	stellar half mass radius 
	"""
	MR = R*(yd*vd*np.abs(vd) + yb*vb*np.abs(vb))/G
	check = MR/(0.5*Mstar)
	k = 0
	for i in range(len(R)):
		if check[i] < 1:
			k = i
	#if (k == 0):
	#	Rhalf = R[-1]
	#else:
	slope = (MR[k+1]-MR[k])/(R[k+1]-R[k])
	intercept = MR[k] - (slope*R[k])
	Rhalf = (Mstar*0.5 - intercept)/slope
	return Rhalf
	
# def mass_exp(logmstar,rd,rmax):
# 	sig = sig_exp(logmstar,rd
# 	return 2.*pi*sig*po(rd,2)*(1. - (np.exp(-1.*rmax/rd) * (1. + rmax/rd)))*1e10
	
def sig_exp(logmstar,rd):
	return E(logmstar)/(2.*pi*po(rd,2))	
  
def mass_exp(logmstar,rd,r):
    return E(logmstar) * (1- (np.exp(-r/rd)*(1+(r/rd))))

def v_exp(logmstar,rd,r):
	mstar = E(logmstar-10.)
	y = frac(r,2.*rd)
	try:
		vsq = 2.*G*frac(mstar,rd)*po(y,2)* (special.i0(y)*special.k0(y) - special.i1(y)*special.k1(y))*1e10
	except:
		print (logmstar, rd, special.i0(y)*special.k0(y))
  
	return po(vsq,0.5)
	
######## PARTICLE PHYSICS
def cs_Born(mX,mphi,aX,v): #for repulsive and attractive
	v = v/(vel_light*1e-3) #now dim of v is unitless
	R = mX*v/(mphi*1e-3)    #mX in GeV and mphi in MeV
	fac = np.log(1. + R**2) - (R**2/(1. + R**2))
	ans = frac(8.*pi*po(aX,2),po(mX,2)*po(v,4)) * fac
	ans = ans*0.389*1e-27 # cross-section in units of cm^2
	return ans/(mX*1.8*1e-24) # cm^2/g
	
def cs_Born2(mX,mphi,aX,v):
	v = v/(vel_light*1e-3)
	mphi = mphi*1e-3	
	const = 4.*pi*(aX**2)/((mX**2)*(v**4))
	t1 = 6*np.log((((mX**2)*(v**2))/(2.*(mphi**2)))+1.)
	f1 = ((4.*(mX**2)*(v**2))+(6.*(mphi**2)))/(((mX**2)*(v**2))+(2.*(mphi**2)))
	f2 = np.log(1. + (((mX**2)*(v**2))/(mphi**2)))
	t2 = f1*f2
	ans = const*(t1-t2)
	ans = ans*0.389*1e-27
	return ans/(mX*1.8*1e-24)
	
	


def cs_Clas(mX,mphi,aX,v): #for repulsive only
	mphi = mphi*1e-3 #MeV to GeV
	if isinstance(v,(list,np.ndarray)) == False:
		check = False
		V = [v]
	else:
		check = True
		V = v
	C = []
	B = []
	V = V/(vel_light*1e-3) # v is unitless now
	for v in V:
		b = 2.*aX*mphi/(mX*(v**2))
		#B.append(b)
		#print b,v*(vel_light*1e-3)
		
		if b <= 1.:
			fac = ln(1. + po(b,-2))*(b**2)
			C.append(frac(2.*pi,mphi**2)*fac)
		else:
			fac = (ln(2.*b)-ln(ln(2.*b)))**2
			C.append(frac(pi,mphi**2)*fac)	
		
	C = np.array(C)
	C = C*0.389*1e-27 # cross-section in units of cm^2
	Cm = C/(mX*1.8*1e-24)  # cm^2/g
	#B = np.array(B)
	if len(Cm) == 1:
		return Cm[0]
	else:
		return Cm
	
	
def cs_ClasAtt(mX,mphi,aX,v):
	mphi = mphi*1e-3
	if isinstance(v,(list,np.ndarray)) == False:
		check = False
		V = [v]
	else:
		check = True
		V = v
	C = []
	B = []
	V = V/(vel_light*1e-3)
	for v in V:
		b = 2.*aX*mphi/(mX*(v**2))
		#B.append(b)
		if b <= 0.1:
			fac = ln(1. + 1./b)*(b**2)
			C.append(fac*(frac(4.*pi,mphi**2)))
		elif b > 0.1 or b <= 1.0e3:
			fac = frac(b**2,1. + 1.5*(b**1.65))
			C.append(fac*frac(8.*pi,mphi**2))
		elif b > 1.0e3:
			fac = ln(b) + 1. - 0.5*po(ln(b),-1)**2
			C.append(fac*frac(pi,mphi**2))
	C = np.array(C)
	C = C*0.389*1e-27
	Cm = C/(mX*1.8*1e-24)
	#B = np.array(B)
	if len(Cm) == 1:
		return Cm[0]
	else:
		return Cm
					
#### SCALING RELATIONS
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


def mu0Dmodel(logm200,slope = 0.329,intercept = -1.323): #return logmu0D
	return slope*logm200 + intercept
	
def mu0Dscatter():	
	return 0.293845 # scatter from fig1 in report. obs. scatter on mu0D

"""
def SHM_ratio(logm200):
	# Returns mstar/m200
	if logm200 > 50.:
		print ("are you sure your logm200 is correct at this stage?\n This is from SHM_ratio in profiles")
	m200 = 10**logm200
	N10 = 0.0351
	M10 = 11.590
	B10 = 1.376
	C10 = 0.608

	M1 = 10**(M10 - 10.)
	M = m200*1e-10

	mm = M/M1

	return 2.*N10/(mm**(-1.*B10) + mm**C10)	

def f_behroozi(x,alp,delta,gam):
    fx = -np.log10(10**(alp*x) + 1.) + delta*(((np.log10(1. + np.exp(x)))**gam)/(1 + np.exp(10**(-x))))
    return fx

def SHM_behroozi(logm200):
    eps = 10**-1.777
    M1 = 10**11.514
    alp_behroozi = -1.412
    delta_behroozi = 3.508
    gam_behroozi = 0.316
    M200 = 10**logm200
    x = logm200 - 11.514
    logmstar = np.log10(eps*M1) + f_behroozi(x,alp_behroozi,delta_behroozi,gam_behroozi) - f_behroozi(0.,alp_behroozi,delta_behroozi,gam_behroozi)
    return logmstar

def lmstar_behroozi_19(logm200):
    eps = -1.435
    alpha = 1.963
    beta = 0.482
    gamma = 10**-1.034
    delta = 0.411
    logM1 = 12.035
    x = logm200 - logM1
    logmstar = eps - np.log10(10**(-alpha*x) + 10**(-beta*x)) + gamma*(np.exp(-0.5*(x/delta)**2)) + logM1
    return logmstar
"""

#### r1 as a function of m200,c200,mu0D
def R1(m200,c200,mu0D):
	r200 = r200_nfw(np.log10(m200))
	rs = r200/c200
	
	def f(x):
		rhob,rb = rhob_rb_burk(np.log10(m200),rs,x)
		if rb == 0 or rhob <= 0 or np.isnan(rb) or np.isnan(rhob) or rb <= 0:
			return 1.
		else:
			#print rhob,rb
			return  np.log10(rhob*rb*1e-6) - np.log10(mu0D)

	fp = Falsepos(f,0.1,2*rs,100)
	if fp['success'] == True:
		r1 = fp['root']
	else:
		r1 = np.inf
	return r1
	
#### statistics
def pdf_lognormal(q, lnq_mean, lnq_sigma):
	"""
	lnq_mean and lnq_sigma are the mean and sigma of the variable ln(q).
	Lognormal pdf returns the pdf for q whose distribution is lognormal 
	=> ln(q) is distributed normally (gaussian).
	"""
	lnq = np.log(q)
	pdf = (1./(q*np.sqrt(2.*np.pi)*lnq_sigma))*np.exp(-((lnq - lnq_mean)**2)/(2.*(lnq_sigma**2)))
	return pdf

def pdf_log10normal(q, logq_mean, logq_sigma):
	logq = np.log10(q)
	pdf1 = np.log10(np.exp(1))/(q*np.sqrt(2.*np.pi)*logq_sigma)
	pdf2 = np.exp(-((logq - logq_mean)**2)/(2.*(logq_sigma**2)))
	pdf = pdf1 * pdf2	
	return pdf
	
def pdfx_log10normal(x, logq_sigma):
	"""
	x = (q/q_mean)
	"""	
	logx = np.log10(x)
	pdf1 = np.log10(np.exp(1))/(x*np.sqrt(2.*np.pi)*logq_sigma)
	pdf2 = np.exp(-(logx**2)/(2.*(logq_sigma**2)))
	return pdf1*pdf2


# ENEGETICS
def PE_cusp(lm,rs,rhig = None):
    if not rhig:
        rhig = r200_nfw(lm)
    r1 = np.linspace(0.1,1,100)
    r2 = np.linspace(1,rhig,1000)
    r = np.concatenate((r1,r2))
    
    dens_scale = density_nfw(lm,rs,rs)
    mass_scale = mass_nfw(lm,rs,rs)
    
    integrand = density_nfw(lm,rs,r)*mass_nfw(lm,rs,r)*r/(dens_scale*mass_scale)
    integral = np.trapz(integrand,r)
    return -4*np.pi*c.G*integral*c.M_sun * 1e6 * c.ergs * dens_scale * mass_scale
def PE_core(lm,rs,rc,rhig = None):
    if not rhig:
        rhig = r200_nfw(lm)
    r1 = np.linspace(0.1,1,100)
    r2 = np.linspace(1,rhig,1000)
    r = np.concatenate((r1,r2))
    
    dens_scale = density_nfw(lm,rs,rs)
    mass_scale = mass_nfw(lm,rs,rs)
    
    integrand = density_sidm(lm,rs,rc,r)*mass_sidm(lm,rs,rc,r)*r/(dens_scale*mass_scale)
    integral = np.trapz(integrand,r) 
    return -4*np.pi*c.G*integral*c.M_sun * 1e6 * c.ergs * dens_scale * mass_scale
def PE_diff(lm,rs,rc):
    pe_cusp = PE_cusp(lm,rs,rhig = rc)
    pe_core = PE_core(lm,rs,rc,rhig = rc)
    diff = (pe_core - pe_cusp)/2.
    return diff