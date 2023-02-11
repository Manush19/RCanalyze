import numpy as np
import scipy as sp
import sys,json,os
sys.path.append('../../')
import Project.profiles as pp
from Project.constants import Constants as pc
import Project.new_profiles as pnp
import Project.plot_assist as pa
import matplotlib.pyplot as plt

class EI_vdf:
    def __init__(self, model, **kwargs):
        """
        model: dict with dict_keys as the keys for components, where 'dm' is
               for dark matter, 'disk' is for disk and 'bulge' is for bulge.
               model[component] = list with first parameter the key representing
               the profile (e.g. 'exp' for exponential profile, 'cusp' for NFW 
               profile, 'core' for cored DM profile) and the parameters of the 
               profile are to be followed. e.g. ['cusp', 11.5, 10], where the
               logM200 = 11.5 and rs = 10 kpc for NFW profile.
        """
        self.dm = model['dm'] if 'dm' in model.keys() else False
        self.disk = model['disk'] if 'disk' in model.keys() else False
        self.bulge = model['bulge'] if 'bulge' in model.keys() else False

        if self.dm:
            if self.dm[0] == 'cusp':
                self.lm,self.rs = self.dm[1],self.dm[2]
            elif self.dm[0] == 'core':
                self.lm,self.rs,self.rc = self.dm[1],self.dm[2],self.dm[3]
            else:
                self.dm_func = self.dm[1]
            
        # radius at which mass of baryons saturates
        self.r_sat = kwargs.get('r_sat') if 'r_sat' in kwargs.keys() else 50 
        
        self.r_infinity = 1e7 # infinity..
        self.R = np.logspace(-2,7,400) # radius array
        
        Mtot = np.array([self.mass_tot(r) for r in self.R])
        self.Mtot_fun = sp.interpolate.interp1d(self.R, Mtot, kind = 'cubic', fill_value = 'extrapolate')
        Ptot = np.array([-self.pote_tot(r) for r in self.R])
        self.Ptot_fun = sp.interpolate.interp1d(self.R, Ptot, kind = 'cubic', fill_value = 'extrapolate')
        tot = np.array([self.second_derivative_fn(r) for r in self.R])
        self.second_derivative = sp.interpolate.interp1d(self.R, tot, kind = 'cubic', fill_value = 'extrapolate')

        
    def density_dm(self,r):
        if self.dm:
            if self.dm[0] == 'cusp':
                return pp.density_nfw(self.lm,self.rs,r)
            elif self.dm[0] == 'core':
                return pp.density_sidm(self.lm,self.rs,self.rc,r)
            else: return self.dm_func(r)
        else: return 0
            
    def mass_bary(self,r):
        if self.disk:
            if self.disk[0] == 'exp':
                mass_disk = pp.mass_exp(self.disk[1],self.disk[2],r)
            else: mass_disk == 0
        else: mass_disk = 0
        
        if self.bulge:
            if self.bulge[0] == 'exp':
                mass_bulge = pp.mass_exp(self.bulge[1],self.bulge[2],r)
            else: mass_bulge = 0
        else: mass_bulge = 0
        return mass_disk + mass_bulge
    
    def mass_dm(self,r):
        if self.dm:
            if self.dm[0] == 'cusp':
                return pp.mass_nfw(self.lm,self.rs,r)
            elif self.dm[0] == 'core':
                return pp.mass_sidm(self.lm,self.rs,self.rc,r)
            else: 
              single = False
              if isinstance(r, (float,int)): single,r = True,[r]
              mass = np.zeros(len(r))
              for i in range(len(r)):
                r_ = np.logspace(-3,np.log10(r[i]),300)
                mass[i] = np.trapz(self.density_dm(r_)*r_**2,r_)*4*np.pi
              if single: return mass[0]
              else: return mass
        else: return 0
    
    def mass_tot(self,r):
        if r < self.r_sat:
            return self.mass_bary(r) + self.mass_dm(r)
        else:
            return self.mass_bary(self.r_sat) + self.mass_dm(r)
        
    def pote_bary(self,r):
        if r > self.r_sat:
            return -pc.G*self.mass_bary(self.r_sat)/r
        else:
            integrand = lambda r_: self.mass_bary(r_)/r_**2
            t1 = sp.integrate.quad(integrand, r, self.r_sat, limit = 10000)[0]
            t2 = self.mass_bary(self.r_sat)/self.r_sat
            return -pc.G*(t1 + t2)
        
    def pote_dm(self,r):
        if self.dm:
            if self.dm[0] == 'cusp':
                return pp.potential_nfw(self.lm,self.rs,r)
            elif self.dm[0] == 'core':
                return pp.potential_sidm(self.lm,self.rs,self.rc,r)
            else:
                single = False
                if isinstance(r, (float,int)): single,r = True,[r]
                r = np.concatenate([r,[1e7]])
                pote = np.zeros(len(r))
                for i in range(len(r)):
                  r_ = np.logspace(-3,np.log10(r[i]),300)
                  pote[i] = np.trapz(self.mass_dm(r_)/(r_**2),r_)*pc.G
                Pote = pote[:-1]-pote[-1]
                if single: return Pote[0]
                else: return Pote
        else: return 0
    
    def pote_tot(self,r):
        return self.pote_bary(r) + self.pote_dm(r)
        
    def Vmax(self,r):
        return np.sqrt(2*self.Ptot_fun(r))
    
    def second_derivative_fn(self,r):
        density_1st_derivative = lambda r_: sp.misc.derivative(self.density_dm, r_, dx = 1e-3, order = 3)
        potential_1st_derivative = lambda r_: -pc.G*self.Mtot_fun(r_)/r_**2
        first_derivative = lambda r_: density_1st_derivative(r_)/potential_1st_derivative(r_)
        return -sp.misc.derivative(first_derivative, r, dx = 1e-3, order = 3)
    
    def get_vdf(self,r,n = 32):
        """
        r : float
            radius in kpc at which VDF has to be found
        """
        self.n = n
        vesc = self.Vmax(r)
        # v = np.arange(0,vesc+vesc/(self.n-1),vesc/(self.n-1))
        v = np.linspace(0,vesc,self.n)
        
        pote_here = self.Ptot_fun(r)
        Ene = np.array([pote_here-0.5*v_**2 for v_ in v])
        
        Rz = np.ones(self.n)
        for i in range(0,self.n-1,1):
            def eqsn(x):
                return self.Ptot_fun(x) - Ene[i]
            Rz[i] = sp.optimize.fsolve(eqsn,x0 = r)
            
        I = np.zeros(self.n)
        for i in range(0,self.n-1,1):
            def integrand(x):
                x = np.array(x)
                return self.second_derivative(x)*(1/(np.sqrt(8)*np.pi**2))/np.sqrt(np.abs(Ene[i]-self.Ptot_fun(x)))
            I[i] = sp.integrate.quad(integrand,Rz[i],1000*Rz[i],points = Rz[i],limit = 10000)[0]

        vdist = I/self.density_dm(r)
        try:
          distr = 4.*np.pi*v**2*vdist
        except:
          print (len(v), len(vdist), r, vesc)
        vdf = np.vstack((v,distr))
        return vdf.transpose()
      
      
class Sidm:
    def __init__(self,**kwargs):
        if 'lm' and 'rs' in kwargs.keys():
            key = 'nfw_to_iso'
            self.lm, self.rs = kwargs.get('lm'),kwargs.get('rs')
            self.ρs = pp.rho0_nfw(self.lm,self.rs)
        elif 'ρ0' and 'σ0' in kwargs.keys():
            key = 'iso_to_nfw'
            self.ρ0,self.σ0 = kwargs.get('ρ0'),kwargs.get('σ0')
        else:
            print ('Please give any one there paris: (lm,rs),(ρ0,σ0)')
    
        self.lmstar = kwargs.get('lmstar') if 'lmstar' in kwargs.keys() else 0
        self.rdstar = kwargs.get('rdstar') if 'rdstar' in kwargs.keys() else 1
        self.Σ0star = 10**(self.lmstar)/(2*np.pi*self.rdstar**2) if self.lmstar else 0
        
        self.tage = kwargs.get('tage') if 'tage' in kwargs.keys() else 10.
        self.σbym = kwargs.get('σbym') if 'σbym' in kwargs.keys() else 3.
        self.unit = 1e-10 * pc.Gyr * pc.yr * 1e3 * pc.M_sun / pc.kpc**3
        self.r = np.logspace(-2,3,400)
        
        if key == 'nfw_to_iso': self.iso_from_nfw()
        elif key == 'iso_to_nfw': self.nfw_from_iso()
        
        
    def ρ_bary(self,r):
        return self.Σ0star*np.exp(-r/self.rdstar)/(2*r)
    
    def ρ_nfw(self,r,**kwargs):
        ρs = kwargs.get('ρs') if 'ρs' in kwargs.keys() else self.ρs
        rs = kwargs.get('rs') if 'rs' in kwargs.keys() else self.rs
        return ρs/((r/rs)*(1 + (r/rs))**2)
    
    def ρ_iso(self,r,**kwargs):
        ρ0 = kwargs.get('ρ0') if 'ρ0' in kwargs.keys() else self.ρ0
        σ0 = kwargs.get('σ0') if 'σ0' in kwargs.keys() else self.σ0
        Φf = kwargs.get('Φf') if 'Φf' in kwargs.keys() else self.Φf
        return ρ0*np.exp(-Φf(r)/σ0**2)
    
    def ρ_kap(self,r):
        single = False
        if isinstance(r,(float,int)): r,single = [r],True
        ρ = np.zeros(len(r))
        for i in range(len(r)):
            if r[i] < self.r1: ρ[i] = self.ρ_iso(r[i])
            else: ρ[i] = self.ρ_nfw(r[i])
        if single: return ρ[0]
        else: return ρ
    
    def M_nfw(self,r,**kwargs):
        ρs = kwargs.get('ρs') if 'ρs' in kwargs.keys() else self.ρs
        rs = kwargs.get('rs') if 'rs' in kwargs.keys() else self.rs
        return 4*np.pi*ρs*(rs**3)*(np.log((r+rs)/rs) - (r/(r+rs)))
        
    def M_iso(self,r,**kwargs):
        ρ0 = kwargs.get('ρ0') if 'ρ0' in kwargs.keys() else self.ρ0
        σ0 = kwargs.get('σ0') if 'σ0' in kwargs.keys() else self.σ0
        Φf = kwargs.get('Φf') if 'Φf' in kwargs.keys() else self.Φf
        single = False
        if isinstance(r, (float,int)): r,single = [r],True
        M = np.zeros(len(r))
        for i in range(len(r)):
            r_ = np.logspace(-2,np.log10(r[i]),400)
            M[i] = 4*np.pi*np.trapz(self.ρ_iso(r_,ρ0 = ρ0,σ0 = σ0,Φf = Φf)*r_**2,r_)
        if single: return M[0]
        else: return M
    
    def M_kap(self,r):
        single = False
        if isinstance(r,(float,int)): r,single = [r],True
        M = np.zeros(len(r))
        for i in range(len(r)):
            if r[i] < self.r1: M[i] = self.M_iso(r[i])
            else: M[i] = self.M_nfw(r[i])
        if single: return M[0]
        else: return M
    
    def Φ_tot(self,**kwargs):
        ρ0 = kwargs.get('ρ0') if 'ρ0' in kwargs.keys() else self.ρ0
        σ0 = kwargs.get('σ0') if 'σ0' in kwargs.keys() else self.σ0
        def f(u,r):
            k1 = 4*np.pi*pc.G
            k2 = 4*np.pi*pc.G*ρ0
            t1 = k1*(self.ρ_bary(r))
            t2 = 2*u[1]/r
            t3 = k2*np.exp(-u[0]/σ0**2)
            return (u[1],t1-t2+t3)
        us = sp.integrate.odeint(f,[0,0],self.r)
        return us[:,0]
    
        
    def get_r1(self,**kwargs):
        ρ0 = kwargs.get('ρ0') if 'ρ0' in kwargs.keys() else self.ρ0
        σ0 = kwargs.get('σ0') if 'σ0' in kwargs.keys() else self.σ0
        Φf = kwargs.get('Φf') if 'Φf' in kwargs.keys() else self.Φf
        ρ1 = np.sqrt(np.pi)/(self.tage*self.σbym*4*σ0) / self.unit
        def solve_r1(lr1):
            r1 = 10**lr1
            return np.abs(np.log10(self.ρ_iso(r1,ρ0=ρ0,σ0=σ0,Φf=Φf))-np.log10(ρ1))/np.log10(ρ1)
        m_r1 = Minuit(solve_r1,lr1 = 1)
        m_r1.limits['lr1'] = (-1,np.log10(200))
        m_r1.errordef = Minuit.LIKELIHOOD
        min_res,m_r1 = qH.quadhop(m_r1,['lr1'],[1])
        return 10**min_res['globalmin'][0]
    
    def nfw_from_iso(self):
        self.Φf = sp.interpolate.interp1d(self.r,self.Φ_tot(),kind = 'cubic', fill_value = 'extrapolate')
        self.r1 = self.get_r1()
        self.ρ1 = self.ρ_iso(self.r1)
        self.M1 = self.M_iso(self.r1)
        ρs_func = lambda rs_: self.ρ1 * (self.r1/rs_) * (1+(self.r1/rs_))**2
        def solve_rs(x):
            return np.abs(np.log10(self.M_nfw(ρs = ρs_func(x),rs=x,r=self.r1)) - np.log10(self.M1))/np.log10(self.M1)
        m_rs = Minuit(solve_rs,x = 1)
        m_rs.limits['x'] = (0.1,200)
        m_rs.errordef = Minuit.LIKELIHOOD
        min_res,m = qH.quadhop(m_rs,['x'],[1])
        self.min_res = min_res
        self.rs = min_res['globalmin'][0]
        self.ρs = ρs_func(self.rs)
        def solve_lm(x):
            r200 = pp.r200_nfw(x)
            m200 = self.M_nfw(r200)
            return np.abs(np.log10(m200)-np.log10(pp.mass_nfw(x,self.rs,r200)))/np.log10(m200)
        m_lm = Minuit(solve_lm,x = 10)
        m_lm.limits['x'] = (7,14)
        m_lm.errordef = Minuit.LIKELIHOOD
        min_res,m = qH.quadhop(m_lm,['x'],[10])
        self.lm = min_res['globalmin'][0]
        
    def check_1(self,r1,ρ0,σ0,Φf):
        rm = self.r[self.r < r1][-3]
        ρ__iso = self.ρ_iso(rm,ρ0=ρ0,σ0=σ0,Φf=Φf)
        ρ__nfw = self.ρ_nfw(rm)
        if ρ__iso < ρ__nfw: return False
        else: return True      
    
    def check_sol(self,minima):
        ρ0,σ0 = 10**minima[0],10**minima[1]
        Φf = sp.interpolate.interp1d(self.r,self.Φ_tot(ρ0=ρ0,σ0=σ0),kind='cubic',fill_value='extrapolate')
        r1 = self.get_r1(ρ0=ρ0,σ0=σ0,Φf=Φf)
        rm = self.r[self.r < r1][-5]
        if self.ρ_nfw(rm) < self.ρ_iso(rm,ρ0=ρ0,σ0=σ0,Φf=Φf): return True
        else: return False
        
    def iso_from_nfw(self):
        def solve_iso(lρ0,lσ0):
            ρ0 = 10**lρ0
            σ0 = 10**lσ0
            Φf = sp.interpolate.interp1d(self.r,self.Φ_tot(ρ0=ρ0,σ0=σ0),kind='cubic',fill_value='extrapolate')
            r1 = self.get_r1(ρ0=ρ0,σ0=σ0,Φf=Φf)
            M1_nfw = self.M_nfw(r1)
            M1_iso = self.M_iso(r1,ρ0=ρ0,σ0=σ0,Φf=Φf)
            ρ1_nfw = self.ρ_nfw(r1)
            ρ1_iso = self.ρ_iso(r1,ρ0=ρ0,σ0=σ0,Φf=Φf)
            return np.abs(np.log10(M1_nfw)-np.log10(M1_iso))/np.log10(M1_nfw) + np.abs(np.log10(ρ1_nfw)-np.log10(ρ1_iso))/np.log10(ρ1_nfw)
        tol,etol = 1e-5,1
        guess = [7,2]
        cnt = 0
        while etol > tol:
            cnt += 1
            m = Minuit(solve_iso,lρ0=guess[0],lσ0=guess[1])
            m.limits['lρ0'] = [np.log10(5e6),9]
            m.limits['lσ0'] = [1,np.log10(500)]
            m.errordef = Minuit.LIKELIHOOD
            min_res,m = qH.quadhop(m,['lρ0','lσ0'],guess)
            guess = min_res['globalmin']
            etol = min_res['globalfun']
            if not cnt%3:
                tol *= 10
                
        min_res['cnt'] = cnt
        self.min_res = min_res
        self.ρ0 = 10**min_res['globalmin'][0]
        self.σ0 = 10**min_res['globalmin'][1]
        self.Φf = sp.interpolate.interp1d(self.r,self.Φ_tot(),kind='cubic',fill_value='extrapolate')
        self.r1 = self.get_r1()
        self.ρ1 = self.ρ_nfw(self.r1)
        self.M1 = self.M_nfw(self.r1)
        
    def print_results(self,full = False):
        print ('logρ0 = %.2f'%np.log10(self.ρ0))
        print ('σ_v0  = %.2f'%self.σ0)
        print ('r1    = %.2f'%self.r1)
        print ('rs    = %.2f'%self.rs)
        print ('lM200 = %.2f'%self.lm)
        print ('r200  = %.2f'%(pp.r200_nfw(self.lm)))
        if full:
            print ('logρs = %.2f'%np.log10(self.ρs))
            print ('σ/m   = %.2f'%self.σbym)
            print ('logM1 = %.2f'%np.log10(self.M1))
            print ('logρ1 = %.2f'%np.log10(self.ρ1))
            print ('logM_s= %.2f'%self.lmstar)
            print ('Rd_s  = %.2f'%self.rdstar)
            lsig = np.log10(self.Σ0star) if self.Σ0star else 0
            print ('logΣ0 = %.2f'%lsig)
            
    def v_nfw(self,r):
        return np.sqrt(pc.G*self.M_nfw(r)/r)
    
    def v_kap(self,r):
        return np.sqrt(pc.G*self.M_kap(r)/r)
            
    def plot_density(self,label = '', c = 'k',ax = None):
        if not ax:
            fig,ax = plt.subplots()
        ax.plot(self.r,self.ρ_kap(self.r), c = c, label = 'SIDM_%s'%label)
        ax.plot(self.r,self.ρ_nfw(self.r), c = c, label = 'NFW_%s'%label, ls = '--')
        ax.scatter(self.r1,self.ρ1, c = c, marker = '.', s = 200)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        return ax
        
    def plot_velocity(self,label = '',c = 'k',ax = None):
        if not ax:
            fig,ax = plt.subplots()
        ax.plot(self.r,self.v_kap(self.r),c = c, label = 'SIDM_%s'%label)
        ax.plot(self.r,self.v_nfw(self.r),c = c, label = 'NFW_%s'%label, ls = '--')
        ax.scatter(self.r1,self.v_kap(self.r1),c = c, marker = '.', s = 200)
        ax.set_xlim(0,100)
        ax.legend()
        return ax
      
class Dc14:
    def __init__(self,lm,c2,ls):
        self.lm = lm
        self.c2 = c2
        self.ls = ls
        self.r200 = pp.r200_nfw(self.lm)
        self.rs= self.r200/self.c2
        self.r = np.logspace(-2,np.log10(self.r200),100)
        self.a,self.b,self.c = pp.abc_dc14(self.ls-self.lm)
        self.X = self.ls-self.lm
        self.rs_dc14 = pp.rs_dc14(self.lm,self.c2,self.X,self.a,self.b,self.c)
        
    def ρ_dc14(self,r):
        return pp.density_dc14(self.lm,self.ls,self.rs_dc14,r)
    def ρ_nfw(self,r):
        return pp.density_nfw(self.lm,self.rs,r)
    def M_dc14(self,r):
        return pp.mass_dc14(self.lm,self.ls,self.rs_dc14,r)
    def M_nfw(self,r):
        return pp.mass_nfw(self.lm,self.rs,r)
    def v_dc14(self,r):
        return pp.v_dc14(self.lm,self.ls,self.rs_dc14,r)
    def v_nfw(self,r):
        return pp.v_nfw(self.lm,self.rs,r)