import numpy as np
import scipy as sp
from scipy import stats
import os,sys,json
sys.path.append('../../')
import Project.profiles as pp
import Project.new_profiles as pnp

def get_truncNorm(mean, sig, n_sig, seed = None):
    """
    Return a "len(seed)" number of random values from a
    truncated gaussian distribution centered at "mean" and standard deviation 
    "sig" and truncated at "mean +- n_sig*sig". 
    """
    if n_sig == 0:
        return [mean]
    X = stats.truncnorm(-n_sig,n_sig,loc = mean, scale = sig)
    if seed:
        return X.rvs(1,random_state = seed)
    else:
        return X.rvs(1)

class Mkrot:
    def __init__(self,lm,rscales,r1byrs,RF,Nsig= 0.5,Nlc_sig = 2,lc_sig = 0.11,shm = 'behroozi_19'):
        """
        Nsig = no. of sigma scatter in c200 for intrisic rotation curve scatter of Vc
        Nlc_sig = no. of sigma for the scatter of mean c200 of the rotation curve.
        lc_sig = scatter of M200-C200 relation.
        """
        self.lm = lm
        self.Nlc_sig = Nlc_sig
        self.lc_sig = lc_sig
        self.Nsig = Nsig
        self.shm = shm
        if self.shm != 'dmo':
            self.fb = False
            self.lc200_mean = pp.lc200_SR(self.lm,fb = 0.)
        else:
            self.fb = True
            self.lc200_mean = pp.lc200_SR(self.lm)
        c200 = 10**get_truncNorm(self.lc200_mean, 0.11, Nlc_sig,False)[0]
        r200 = pp.r200_nfw(self.lm)
        self.c200 = np.around(c200,2)
        self.lc200 = np.log10(self.c200)
        self.r200 = r200
        self.rs = np.around(r200/c200,2)
        self.rscales = rscales
        self.r = self.rscales*self.rs
        self.r1byrs = r1byrs
        self.find_r1_from_rdstar = False
        if r1byrs > 0 and isinstance(r1byrs, (float)):
            self.r1 = self.rs*self.r1byrs
        elif r1byrs == 'rdstar' or r1byrs == 'p5rdstar':
            self.find_r1_from_rdstar = True
        elif r1byrs == 0:
            self.r1byrs = np.around(np.random.uniform(0.3,0.9),2)
            self.r1 = self.rs*self.r1byrs
        elif r1byrs == -1:
            self.r1byrs = np.around(np.random.uniform(0.4,0.8),2)
            self.r1 = self.rs*self.r1byrs
        elif r1byrs == -2:
            self.r1byrs = np.around(np.random.uniform(0.5,0.7),2)
            self.r1 = self.rs*self.r1byrs
        self.RF = RF
        self.found_bary_relations = False
        
    def mk_vel_nfw(self):
        V = []
        for r in self.r:
            c200_r = 10**get_truncNorm(self.lc200, self.lc_sig, self.Nsig, False)[0]
            rs = self.r200/c200_r
            v = pp.v_nfw(self.lm,rs,r)
            V.append(v)
        return np.array(V)
    
    def mk_vel_sidm(self,**kwargs):
        V = []
        if self.find_r1_from_rdstar:
            self.bary_relations(kwargs.get('nsig'),kwargs.get('sparc_mgas'),kwargs.get('sig_SHM'))
            self.found_bary_relations = True
            if self.r1byrs == 'rdstar':
                self.r1 = self.rdstar
            elif self.r1byrs == 'p5rdstar':
                self.r1 = self.rdstar*0.5
            self.r1byrs = self.r1/self.rs
        for r in self.r:
            c200_r = 10**get_truncNorm(self.lc200, self.lc_sig, self.Nsig, False)[0]
            rs = self.r200/c200_r
            v = pp.v_sidm(self.lm,rs,self.r1,r)
            V.append(v)
        return np.array(V)
    
    def lmstar_from_shm(self):
        if self.shm == 'behroozi_13':
            logmstar = pnp.lmstar_behroozi_13(self.lm)
        elif self.shm == 'moster':
            logmstar = pnp.lmstar_moster_13(self.lm)
        elif self.shm == 'nobump':
            logmstar = pnp.lmstar_behroozi_nbp(self.lm)
        elif self.shm == 'behroozi_19':
            logmstar =  pnp.lmstar_behroozi_19(self.lm)
        elif self.shm == 'dmo':
            logmstar =  0
        elif isinstance(self.shm,int) or isinstance(self.shm,float):
            logmstar = np.log10(self.shm*10**self.lm)
        else:
            print ('Unrecognized SHM, choose either [behroozi, moster or int/flat')
        return logmstar
    
    def bary_relations(self,nsig,sparc_mgas,sig_SHM):
        """
        If sparc_mgas is True, the mgas is obtained from mstar using the 
        relations obtained from SPARC galactic fits.
        
        If sig_SHM == 'constant', the scatter for SHM (Mstar as a function of M200) is 
        taken as a constant value of 0.3. Else give sig_SHM == function(logM200) which
        returns the mass dependent scatter in dex.
        """
        if nsig != 0:
            self.yd = np.around(10**get_truncNorm(np.log10(0.5),0.1,3)[0],2)
        else:
            self.yd = 0.5
        
        if self.shm == 'dmo':
            self.lmstar = 0
            self.lmgas = 0
            self.rdstar = 0
            self.rdgas = 0
        else:
            self.lmstar_mean = self.lmstar_from_shm()
            if sig_SHM == 'constant':
                self.sig_lmstar = 0.3
            else:
                self.sig_lmstar = sig_SHM(self.lm)
            self.lmstar = np.around(get_truncNorm(self.lmstar_mean, self.sig_lmstar, 
                                                  nsig, False)[0],2)
            if not sparc_mgas:
                self.lmgas_mean = 0.57*self.lmstar + 3.86
            else:
                self.lmgas_mean = 0.52*self.lmstar + 4.44
            self.sig_lmgas = 0.47
            self.lmgas = np.around(get_truncNorm(self.lmgas_mean, self.sig_lmgas,
                                                 nsig, False)[0],2)
            self.lrdgas_mean = 0.59*self.lmgas - 4.80
            self.sig_lrdgas = 0.14
            self.rdgas = np.around(10**get_truncNorm(self.lrdgas_mean, self.sig_lrdgas, 
                                                     nsig, False)[0],2)
            self.lrdstar_mean = 0.91*np.log10(self.rdgas) - 0.38
            self.sig_lrdstar = 0.21
            self.rdstar = np.around(10**get_truncNorm(self.lrdstar_mean, self.sig_lrdstar,
                                                      nsig, False)[0],2)
                               
    def mk_vel_star_gas(self,nsig = 2,sparc_mgas = True,sig_SHM = 'constant'):
        if not self.found_bary_relations:
            self.bary_relations(nsig,sparc_mgas,sig_SHM)
            self.found_bary_relations = True
        if self.shm == 'dmo':
            return np.zeros(len(self.r)),np.zeros(len(self.r))
        else:
            vstar = pp.v_exp(self.lmstar,self.rdstar,self.r)/np.sqrt(self.yd)
            vgas = pp.v_exp(self.lmgas,self.rdgas,self.r)
        return vstar,vgas
    
    def get_cdfdict(self,errdict):
        cdfdict = {}
        for key in errdict:
            err = errdict[key]
            x,counts = np.unique(err, return_counts = True)
            cumsum = np.cumsum(counts)
            cdf = cumsum/cumsum[-1]
            cdfdict[key] = [x,cdf]
        return cdfdict
    
    def get_custom_rand(self,x, cdf):
        np.random.seed()
        r = np.random.rand()
        indx = np.absolute(cdf - r).argmin()
        return x[indx]
    
    def get_errModel(self,cdfdict):
        errm = []
        i = 0
        for key in cdfdict:
            x,cdf = cdfdict[key]
            N_dpts = self.RF[i]
            for j in range(N_dpts):
                errm.append(self.get_custom_rand(x,cdf))
            i += 1
        return np.array(errm)
    
    def mk_verr(self,errdict,v):
        cdfdict = self.get_cdfdict(errdict)
        err_check = 'NOT OK'
        while err_check == 'NOT OK':
            err = self.get_errModel(cdfdict)
            err = err[:len(self.r)]
            ve = v*err/100.
            if np.any(v - ve < 0):
                err_check = 'NOT OK'
            else:
                err_check = 'OK'
        return ve
    
    def  write_rotcurv(self,filename,v,ve,vg,vs,vd):
        file1 = open(filename,'w')
        for j in range(len(self.r)):
            file1.write('%.8f    %.8f    %.8f    %.8f    %.8f     %.8f\n'%(self.r[j],
                                                                           v[j],
                                                                           ve[j],
                                                                           vg[j], 
                                                                           vs[j],
                                                                           vd[j]))
        file1.close()
        
    def get_galdetails(self,rn,r1):
        details = {'lm':self.lm,
                   'rs':self.rs,
                   'rn':rn,
                   'r1byrs':self.r1byrs,
                   'r1':r1,
                   'lmstar':self.lmstar,
                   'rdstar':self.rdstar,
                   'lmgas':self.lmgas,
                   'rdgas':self.rdgas,
                   'shm':self.shm,
                   'c200':self.c200,
                   'yd':self.yd}
        return details
    
        
            
        
        
