import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import sys,os,json
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import Project.profiles as pp
import Project.new_profiles as pnp

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# purple = (0.25098039215686274, 0.0, 0.29411764705882354, 1.0)
# green = (0.0, 0.26666666666666666, 0.10588235294117647, 1.0)
pr = truncate_colormap(plt.get_cmap('PRGn'), 0.45,0)
gn = truncate_colormap(plt.get_cmap('PRGn_r'), 0,.45)
pr_r = truncate_colormap(plt.get_cmap('PRGn'), 0,0.45)
gn_r = truncate_colormap(plt.get_cmap('PRGn_r'), 0.45,0)
prgn = plt.get_cmap('PRGn')
prgn_r = plt.get_cmap('PRGn_r')
purple = pr(0.6)
green = gn(0.4)

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def axs_MCR(axs = None, nrow =1, ncol = 2, wspace = 0.2, hspace = 0.2, nsig = [1,2],figsize = None,
            sharex = False, sharey = False,xlabel_size = 13, ylabel_size = 14):
    axsexists = False
    if np.any(axs):
        axsexists = True
        if not isinstance(axs, (list,np.ndarray)):
            axs = [axs]
    else:
        if not figsize:
            figsize = (6*ncol,4*nrow)
        fig,axs = plt.subplots(nrow, ncol, figsize = figsize, gridspec_kw = {'wspace':wspace, 
                               'hspace':hspace},sharex = sharex, sharey = sharey)
        if nrow == 1 and ncol == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

    lm = np.linspace(8,14,20)
    lc = pp.lc200_SR(lm)
    if isinstance(nsig, (float,int)):
        nsig = [nsig]
    for ax in axs:
        ax.plot(10**lm,10**lc, c = 'k')
        for ns in nsig:
            ax.fill_between(10**lm, 10**(lc + ns*0.11), 10**(lc - ns*0.11), color = 'grey', alpha = 0.25)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('$M_{200}\,(\mathrm{M_{\odot}})$',size = xlabel_size)
        ax.set_ylabel('$c_{200}$', size = ylabel_size)
        ax.set_xlim(5e8,9e13)
        ax.set_ylim(1.5,55)
        ax.set_yticks([2,5,10,20,3,50],[2,5,10,20,3,50])
        ax.tick_params(which = 'both', direction = 'in', right = True, top = True)
    if len(axs) == 1:
        axs = axs[0]
    if axsexists:
        return axs
    else:
        return fig,axs

def scatt_MCR(ax,mcr,efmt = '.', ealpha = 0.2, marker = 'o', s = 100, alpha = 0.5, elw = 1,clr = None,**kwargs):
    ax.errorbar(mcr[0],mcr[1],xerr = np.array([mcr[2]]).T, yerr = np.array([mcr[3]]).T, fmt = efmt,
                color = 'none', ecolor = 'k', capsize = 2, alpha = ealpha,elinewidth = elw)
    if not clr:
      clr = mcr[4]
    ax.scatter(mcr[0],mcr[1],color = clr,marker = marker, s = s, alpha = alpha, **kwargs)
    
def axs_SHM(axs = None, nrow =1, ncol = 2, wspace = 0.2, hspace = 0.2, nsig = [1,2],figsize = None,
            sharex = False, sharey = False,xlabel_size = 13, ylabel_size = 13, shm='behroozi_19'):
    axsexists = False
    if np.any(axs):
        axsexists = True
        if not isinstance(axs, (list,np.ndarray)):
            axs = [axs]
    else:
        if not figsize:
            figsize = (6*ncol,4*nrow)
        fig,axs = plt.subplots(nrow, ncol, figsize = figsize, gridspec_kw = {'wspace':wspace, 
                               'hspace':hspace},sharex = sharex, sharey = sharey)
        if nrow == 1 and ncol == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

            
    lm = np.linspace(8,14,20)
    if shm == 'behroozi_19':
        lms = pnp.lmstar_behroozi_19(lm)
    else:
        print ('other shm"s are not included yet.')
        return 0

    if isinstance(nsig, (float,int)):
        nsig = [nsig]
    for ax in axs:
        ax.plot(10**lm,10**lms, c = 'k')
        for ns in nsig:
            ax.fill_between(10**lm, 10**(lms + ns*0.3), 10**(lms - ns*0.3), color = 'grey', alpha = 0.25)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('$M_{200}\,(\mathrm{M_{\odot}})$',size = xlabel_size)
        ax.set_ylabel('$M_{\mathrm{star}}\,(\mathrm{M_{\odot}})$', size = ylabel_size)
        ax.set_xlim(5e8,9e13)
        ax.set_ylim(1e5,1e12)
        # ax.set_yticks([2,5,10,20,3,50],[2,5,10,20,3,50])
        ax.tick_params(which = 'both', direction = 'in', right = True, top = True)
    if len(axs) == 1:
        axs = axs[0]
    if axsexists:
        return axs
    else:
        return fig,axs

def scatt_SHM(ax,shm,efmt = '.', ealpha = 0.2, marker = 'o', s = 100, alpha = 0.5, elw = 1,clr = None,**kwargs):
    ax.errorbar(shm[0],shm[1],xerr = np.array([shm[2]]).T, yerr = np.array([shm[3]]).T, fmt = efmt,
                color = 'none', ecolor = 'k', capsize = 2, alpha = ealpha,elinewidth = elw)
    if not clr:
      clr = shm[4]
    ax.scatter(shm[0],shm[1],color = clr,marker = marker, s = s, alpha = alpha, **kwargs)
    
    
def plot_RC(gal,sparcd,ax = None,prior = 'lcdm_reg',DM = False, plot_type = 'all',bands = False):
    fillx = lambda ax,x,y,clr: ax.fill_betweenx(y,x[0]-x[1],x[0]+x[1],color = clr,alpha = 0.1)
    if not np.any(ax):
        fig,ax = plt.subplots(1,1,figsize = (6,4))
    Gald = sparcd[prior][gal]
    ax.errorbar(Gald['r'],Gald['vc'],Gald['ve'],color = 'k',fmt = '.',capsize=2)
    r = np.array(Gald['r'])
    if plot_type == 'best_fit':
        bestfit = sparcd[prior][gal]['best_fit']
        gald = sparcd[prior][gal][bestfit]
        clr = green if bestfit == 'cusp_fit' else purple
        ax.plot(r,gald['vcmodel'],color = clr,lw = 2)
        ax.plot(r,gald['vbary'],color = clr,lw = 1, ls = '-.')
        if DM:
          ls = '-' if bestfit == 'cusp_fit' else '--'
          ax.plot(r,gald['vdark'],color = 'k',lw = 1, ls = ls)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        rs = gald['rs']
        if rs[0] < xlim[1]:
            ax.plot([rs[0],rs[0]],ylim,c = clr, ls = '-')
            if bands:
                fillx(ax,rs,ylim,clr)
        if bestfit == 'core_fit':
            rc = gald['r1']
            if rc[0] < xlim[1]:
                ax.plot([rc[0],rc[0]],ylim,c = clr,ls = '--')
                if bands:
                    fillx(ax,rc,ylim,clr)
            ax.plot(r,gald['vnfw_core'],ls = ':',color = clr)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    if plot_type == 'all':
        for fit in ['cusp_fit','core_fit']:
            clr = green if fit == 'cusp_fit' else purple
            gald = sparcd[prior][gal][fit]
            ax.plot(r,gald['vcmodel'],color = clr,lw = 2)
            ax.plot(r,gald['vbary'],color = clr,lw = 1,ls = '-.')
            if DM:
              ls = '-' if fit == 'cusp_fit' else '--'
              ax.plot(r,gald['vdark'],color = 'k',lw = 1, ls = ls)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            rs = gald['rs']
            if rs[0] < xlim[1]:
                ax.plot([rs[0],rs[0]],ylim,c = clr,ls = '-')
                if bands:
                    fillx(ax,rs,ylim,clr)
            if fit == 'core_fit':
                rc = gald['r1']
                if rc[0] < xlim[1]:
                    ax.plot([rc[0],rc[0]],ylim,c = clr,ls = '--')
                    if bands:
                        fillx(ax,rc,ylim,clr)
                ax.plot(r,gald['vnfw_core'],ls = ':',color = clr)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    return ax
        