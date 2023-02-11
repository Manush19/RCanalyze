import numpy as np
import scipy as sp
from scipy import stats
import os,sys,json
sys.path.append('../../')
import Project.profiles as pp


def get_rs(logm200,fb = False):
    """
    returns the value of nfw scale radius (rs) from the 
    dutton&maccio m200-c200 relation from the mass 10**logm200.
    
    If B, then fb = True
    """
    if fb:
        c200 = 10**pp.lc200_SR(logm200)
    else:
        c200 = 10**pp.lc200_SR(logm200, fb = 0)
    r200 = pp.r200_nfw(logm200)
    return r200/c200

def randseed(seed = 4534):
    np.random.seed(seed)
    randseeds = np.random.randint(10,100000,size = 10000,dtype = int)
    return randseeds 
randseeds = randseed()

def get_truncNorm(mean,sig,n_sig,seed = randseeds):
    """
    Return a "len(seed)" number of random values from a
    truncated gaussian distribution centered at "mean" and standard deviation 
    "sig" and truncated at "mean +- n_sig*sig". 
    """
    X = stats.truncnorm(-n_sig,n_sig,loc=mean,scale=sig)
    if seed:
        return X.rvs(1,random_state = seed)
    else:
        return X.rvs(1)

def get_vel_nfw(logm200,R,Nsig=1.,fb = False, seeds = randseeds):
    """
    get a mock nfw (dm only) velocity curve for m200 = 10**logm200
    at the given radius's R. The intrincic scatter of the rotation 
    curve is given by the get_truncNorm fuction for each radius.
    """
    if not fb:
        lc200 = pp.lc200_SR(logm200,fb = 0.)
    else:
        lc200 = pp.lc200_SR(logm200)
    r200 = pp.r200_nfw(logm200)
    V = []
    i = 0
    for r in R:
        if seeds:
            c200 = 10**get_truncNorm(lc200,0.11,Nsig,seeds[i])
        else:
            c200 = 10**get_truncNorm(lc200,0.11,Nsig,False)
        rs = r200/c200
        v = pp.v_nfw(logm200,rs,r)[0]
        V.append(v)
        i += 1
    return np.array(V)

def get_vel_sidm(logm200,r1,R,Nsig=1.,fb=False,seeds = randseeds):
    if fb == False:
        lc200 = pp.lc200_SR(logm200,fb = 0.)
    else:
        lc200 = pp.lc200_SR(logm200)
    r200 = pp.r200_nfw(logm200)
    V = []
    i = 0
    for r in R:
        if seeds:
            c200 = 10**get_truncNorm(lc200,0.11,Nsig,seeds[i])
        else:
            c200 = 10**get_truncNorm(lc200,0.11,Nsig,False)
        rs = r200/c200
        v = pp.v_sidm(logm200,rs,r1,r)[0]
        V.append(v)
        i += 1
    return np.array(V)

def SHM_nobump_behroozi(logm200):
    if 11.3 <= logm200 <= 12.95:
        lm = 11.3
    else:
        lm = logm200
    return 10**pp.SHM_behroozi(lm)/10**lm

def log_mstar(logm200):
    lmstar = pp.SHM_behroozi(logm200)
    return lmstar

def log_mstar_nobump(logm200):
    lmstar = np.log10(SHM_nobump_behroozi(logm200)*10**logm200)
    return lmstar
                                 
def log_Mstar(logm200):
    ratio = pp.SHM_ratio(logm200)
    return np.log10(ratio * 10**logm200)

def log_mstarbf(logm200,shm):
    return np.log10(shm * 10**logm200)

def lmstar_from_shm(logm200,shm):
    if shm == 'behroozi':
        logmstar = log_mstar(logm200)
    elif shm == 'moster':
        logmstar = log_Mstar(logm200)
    elif shm == 'nobump_behroozi':
        logmstar = log_mstar_nobump(logm200)
    elif shm == 'behroozi_19':
        logmstar = pp.lmstar_behroozi_19(logm200)
    elif shm == 'no_bary':
        return np.zeros(len(logm200))
    elif isinstance(shm,int) or isinstance(shm,float):
        logmstar = log_mstarbf(logm200,shm)
    else:
        print ('Unrecognized SHM, chose either [behroozi, moster or int/float]')
    return logmstar
    
    
def get_vel_star(logm200,r,shm):
    if shm == 'behroozi':
        logmstar = log_mstar(logm200)
    elif shm == 'moster':
        logmstar = log_Mstar(logm200)
    elif shm == 'nobump_behroozi':
        logmstar = log_mstar_nobump(logm200)
    elif shm == 'behroozi_19':
        logmstar = pp.lmstar_behroozi_19(logm200)
    elif shm == 'no_bary':
        return np.zeros(len(r))
    elif isinstance(shm,int) or isinstance(shm,float):
        logmstar = log_mstarbf(logm200,shm)
    else:
        print ('Unrecognized SHM, chose either [behroozi, moster or int/float]')
    hs = 10**(-2.46+0.281*logmstar)
    vstr = pp.v_exp(logmstar,hs,r)
    return np.array(vstr)

def get_vel_gas(logm200,r,shm):
    if shm == 'behroozi':
        logmstar = log_mstar(logm200)
    elif shm == 'moster':
        logmstar = log_Mstar(logm200)
    elif shm == 'nobump_behrooiz':
        logmstar = log_mstar_nobump(logm200)
    elif shm == 'behroozi_19':
        logmstar = pp.lmstar_behroozi_19(logm200)
    elif shm == 'no_bary':
        return np.zeros(len(r))
    elif isinstance(shm,int) or isinstance(shm,float):
        logmstar = log_mstarbf(logm200,shm)
    MH = 10**(logmstar)*10**(-0.43*logmstar + 3.75)
    mgas = 1.3*MH
    hs = 10**(-2.46+0.281*logmstar)
    hg = 1.5*hs
    v = pp.v_exp(logmstar,hg,r)
    return v


