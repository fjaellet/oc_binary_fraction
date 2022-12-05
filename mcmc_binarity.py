import numpy as np
import os
from astropy.table import Table, vstack

from scipy import interpolate
from scipy.stats import gaussian_kde, norm
import scipy.optimize as op
import time
import emcee
import corner
from multiprocessing import Pool

from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=22, usetex=True)

def lnlike(theta, x, y, yerr):
    """
    Polynomial MS + parallel binary sequence (both with constant scatter)
    """
    # Unpack theta: First the polynomial coeffs, then the binarity parameters
    #p0,p1,p2,p3,p4, fb, DeltaG, sigMS, sigBS = theta
    fb, DeltaG, sigMS, sigBS = theta[-4:] 
    polycoeffs_ms            = np.array(theta[:-4][::-1])
    polycoeffs_bs            = polycoeffs_ms.copy()
    polycoeffs_bs[-1]        = polycoeffs_ms[-1] + DeltaG
    assert polycoeffs_ms[-1] != polycoeffs_bs[-1]
    # Define the functional form of the MS and the BS
    mainseq   = np.poly1d(polycoeffs_ms)
    binseq    = np.poly1d(polycoeffs_bs)
    # Compute the likelihood - as a product of the individual data point probabilities
    summand_1 = (1-fb) * np.exp( - (y - mainseq(x))**2. / (2.*(yerr**2. + sigMS**2.))) / np.sqrt(yerr**2. + sigMS**2.)
    summand_2 =    fb  * np.exp( - (y -  binseq(x))**2. / (2.*(yerr**2. + sigBS**2.))) / np.sqrt(yerr**2. + sigBS**2.)
    logL      =  np.sum(np.log(summand_1 + summand_2))
    return logL

def lnlike2(theta, x, y, xerr, yerr):
    """
    Polynomial MS + parallel binary sequence (both with constant scatter).
    This takes into account uncertainties in the colour (x axis) too 
    """
    # Unpack theta: First the polynomial coeffs, then the binarity parameters
    #p0,p1,p2,p3,p4, fb, DeltaG, sigMS, sigBS = theta
    fb, DeltaG, sigMS, sigBS = theta[-4:] 
    polycoeffs_ms            = np.array(theta[:-4][::-1])
    polycoeffs_bs            = polycoeffs_ms.copy()
    polycoeffs_bs[-1]        = polycoeffs_ms[-1] + DeltaG
    assert polycoeffs_ms[-1] != polycoeffs_bs[-1]
    # Define the functional form of the MS and the BS
    mainseq   = np.poly1d(polycoeffs_ms)
    binseq    = np.poly1d(polycoeffs_bs)
    # Compute the effective sigma_x (we need the orthogonal component in each )
    derivative= np.polyder(mainseq)
    sigma2_eff= ( (derivative(x) * xerr)**2. + yerr**2. )/(1. + (derivative(x))**2. ) 
    # Compute the likelihood - as a product of the individual data point probabilities
    summand_1 = (1-fb) * np.exp( - (y - mainseq(x))**2. / (2.*(sigma2_eff + sigMS**2.))) / np.sqrt(sigma2_eff + sigMS**2.)
    summand_2 =    fb  * np.exp( - (y -  binseq(x))**2. / (2.*(sigma2_eff + sigBS**2.))) / np.sqrt(sigma2_eff + sigBS**2.)
    logL      =  np.sum(np.log(summand_1 + summand_2))
    return logL

def lnlike_simple(theta, x, y, yerr):
    """
    Polynomial MS + parallel binary sequence (both with constant scatter)
    """
    # Unpack theta
    polycoeffs = theta[::-1]
    # Define the functional form of the MS and the BS
    mainseq   = np.poly1d(polycoeffs)
    # Compute the likelihood - as a product of the individual data point probabilities
    return np.sum(- (y - mainseq(x))**2. / (2.*(yerr**2.)))

def lnprior(theta):
    """
    Return flat priors in all parameters
    """
    # Unpack theta: First the polynomial coeffs, then the binarity parameters
    polycoeffs               = theta[:-4]
    fb, DeltaG, sigMS, sigBS = theta[-4:]
    if (abs(p) < 10 for p in polycoeffs) and \
    0 <   fb  < 0.9 and -0.9 < DeltaG < -0.5  and \
    0 < sigMS < 0.3 and  0   <  sigBS <  0.3 :
        return 0.0
    return -np.inf

def lnprior2(theta):
    """
    Return flat priors in fb + Gaussians in DeltaG, sigMS, sigBS
    """
    # Unpack theta: First the polynomial coeffs, then the binarity parameters
    polycoeffs               = theta[:-4]
    fb, DeltaG, sigMS, sigBS = theta[-4:]
    if 0 <  fb  < 0.9 and -1 < DeltaG < -0.4 and 0 < sigMS < 0.4 and 0 < sigBS < 0.4:
        return norm.pdf(DeltaG, -0.75, 0.05) * \
               norm.pdf(np.log10(sigMS), -1, 0.2) * \
               norm.pdf(np.log10(sigBS), -0.8, 0.2)
    return -np.inf

def lnprior3(theta):
    """
    Return flat priors in fb + Gaussians in DeltaG, sigMS, sigBS + Gaussians in polynomials
    """
    # Unpack theta: First the polynomial coeffs, then the binarity parameters
    p0, p1, p2, p3, p4, p5, p6 = theta[:-4]
    fb, DeltaG, sigMS, sigBS   = theta[-4:]
    x = np.arange(-0.5, 4.5, 0.5)
    if 0 <  fb  < 0.9 and -1 < DeltaG < -0.4 and 0 < sigMS < 0.4 and 0 < sigBS < 0.4:
        return norm.pdf(DeltaG, -0.75, 0.05) * \
               norm.pdf(np.log10(sigMS), -1, 0.2) * \
               norm.pdf(np.log10(sigBS), -0.8, 0.2) * \
               np.prod(norm.pdf(p1 + 2*p2*x + 3*p3*x**2 + 4*p4*x**3 + 5*p5*x**4 + 6*p6*x**5, -3, 2))
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def lnprob2(theta, x, y, xerr, yerr):
    lp = lnprior2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike2(theta, x, y, xerr, yerr)

def lnprob3(theta, x, y, xerr, yerr):
    lp = lnprior3(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike2(theta, x, y, xerr, yerr)

def maxlike(theta0, x, y, yerr):
    # Find the maximum likelihood values for the simple polynomial fit model.
    chi2 = lambda *args: -2 * lnlike_simple(*args)
    result = op.minimize(chi2, theta0[:-4], args=(x, y, yerr))
    print("Maximum likelihood result:", result["x"])
    # Refine our first guess for the polynomial fit
    theta0[:-4] = result["x"]
    return theta0

def run_mcmc(x, y, xerr, yerr, theta0, nwalkers=32, nsteps=8000, burnin=3000,
             plotdir="../im/mcmc/", clusname="test", save_results=False, plot_max_likelihood=False):
    # Set up the sampler.
    ndim = len(theta0)
    pos = [theta0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    print("Running MCMC...")
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=(x, y, xerr, yerr), pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, nsteps, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    acc_frac = np.mean(sampler.acceptance_fraction)
    print("Mean acceptance fraction: {0:.2f}".format(acc_frac))

    if acc_frac > 0.05:
        # Get the marginalised fit parameters
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        p0_m,p1_m,p2_m,p3_m,p4_m,p5_m,p6_m,fb_m,DG_m,sM_m,sB_m = map(lambda v: (np.round(v[1],3), 
                                                                        np.round(v[2]-v[1],3), 
                                                                        np.round(v[1]-v[0],3)), 
                                                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        # Save the last 100 samples
        np.savez(plotdir + "samples_mcmc_" + clusname + ".npz", samples = samples[:100])
        # MCMC iteration plot
        plt.clf()
        fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 3*ndim))
        poly_labels    = ["$p_0$", "$p_1$", "$p_2$", "$p_3$", "$p_4$", "$p_5$", "$p_6$", "$p_7$", "$p_8$"]
        mixture_labels = ["$f_b$","$\Delta G$", "$\sigma_{MS}$", "$\sigma_{BS}$"]
        labels         = poly_labels[:(ndim-4)] + mixture_labels
        for ii in np.arange(ndim):
            axes[ii].plot(sampler.chain[:, :, ii].T, color="k", alpha=0.4)
            axes[ii].set_ylabel(labels[ii])
        axes[ndim-1].set_xlabel("step number")
        fig.tight_layout(h_pad=0.0)
        fig.savefig(plotdir+"mcmc_chain_parameter_iteration_"+clusname+".png")
        plt.close()

        # Corner plot
        fig = corner.corner(samples, labels=labels, show_titles=True, title_kwargs={"fontsize": 15})
        fig.savefig(plotdir+"mcmc_fit_corner_"+clusname+".png")
        plt.suptitle(clusname.replace("_", " "), fontsize=25)
        plt.close()
        
        # CMD plot
        plt.figure(figsize=(8,8))
        plt.errorbar(x,y,xerr=xerr, yerr=yerr, ls='none', label=None, c="grey", zorder=0)
        plt.scatter(x,y, label=None, c="w")
        xarr = np.linspace(np.min(x),np.max(x),100)
        fb_simple=[]
        if plot_max_likelihood:
            # Plot the maximum likelihood result.
            polynom = np.poly1d(theta0[:-4][::-1])
            plt.plot(xarr, polynom(xarr), lw=2, c="grey", label="Max. likelihood fit")
        # Plot some MCMC samples onto the data.
        for p0,p1,p2,p3,p4,p5,p6,fb,DG,sM,sB in samples[np.random.randint(len(samples), size=30)]:
            ms = np.poly1d([p6,p5,p4,p3,p2,p1,p0])
            plt.plot(xarr, ms(xarr), color="b", alpha=0.2)
            bs = np.poly1d([p6,p5,p4,p3,p2,p1,p0+DG])
            plt.plot(xarr, bs(xarr), color="orange", alpha=0.2)
            binaries = (y < ms(x) - 3 * sM)
            fb_simple.append(np.sum(binaries)/len(x))
            plt.scatter(x[binaries],y[binaries], label=None, c="r", s=30, lw=0.3, alpha=.04)
        # Plot the best-parameter result.
        #polynom = np.poly1d([p6_m[0],p5_m[0],p4_m[0],p3_m[0],p2_m[0],p1_m[0],p0_m[0]])
        plt.plot(xarr, ms(xarr), c='blue', label="MCMC main sequence samples", alpha=0.2)
        plt.plot(xarr, bs(xarr), c='orange', label="MCMC binary sequence samples", alpha=0.2)
        plt.xlabel(r"$BP-RP$", fontsize=22)
        plt.ylabel(r"$G$", fontsize=22)
        plt.legend(loc="upper right", fontsize=19)
        # Annotate name of the cluster and binary fraction
        ax = plt.gca()
        plt.text(0.1, 0.2, clusname.replace("_", " "), horizontalalignment='left',
                 verticalalignment='center', transform=ax.transAxes, fontsize=22)
        plt.text(0.1, 0.1, r"$f_b={{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(fb_m[0],fb_m[1],fb_m[2]), 
                 horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=22)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(plotdir+"cmd_fit_mcmc_"+clusname+".png")
        plt.close()
        
        # CMD residual plot
        plt.figure(figsize=(4,8))
        ax = plt.subplot(111)
        # first of all, the base transformation of the data points is needed
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(-90)
        ax2 = ax.twinx() 
        edges = [-1.3,.3]
        kdearr = np.linspace(edges[0], edges[1],1000)
        kderes = np.zeros(1000)
        # Overplot the 30 realizations
        for p0,p1,p2,p3,p4,p5,p6,fb,DG,sM,sB in samples[np.random.randint(len(samples), size=30)]:
            ms = np.poly1d([p6,p5,p4,p3,p2,p1,p0])
            # Compute Delta G_i
            DGi  = (y - ms(x))[(y - ms(x)<edges[1]) & (y - ms(x)>edges[0])]
            vals, bins, patches = plt.hist(-DGi, density=True, histtype="step", bins="fd", lw=2.5,
                                           label="Gaia DR2 open clusters ($d < 1.5$ kpc)", alpha=.2, 
                                           color="grey", transform= rot + base)
            # Compute KDE
            kde = gaussian_kde(y - ms(x), bw_method=.1)
            kderes = kderes + kde.evaluate(kdearr)
        plt.plot(-kdearr, kderes/30, c="k", lw=3, alpha=1, transform= rot + base)
        # Overplot the 2 Gaussians #fb_m,DG_m,sM_m,sB_m
        plt.plot(-kdearr, (1-fb_m[0]) * norm.pdf(kdearr, loc=0., scale=sM_m[0]), c="b", lw=3, 
                 alpha=1, transform= rot + base)
        plt.plot(-kdearr,   fb_m[0]   * norm.pdf(kdearr, loc=DG_m[0], scale=sB_m[0]), c="orange", lw=3, 
                 alpha=1, transform= rot + base)
        # Compute Chi2 (Residual KDE - 2 Gaussian fit)
        fit_func = (1-fb_m[0]) * norm.pdf(kdearr, loc=0., scale=sM_m[0]) + fb_m[0] * norm.pdf(kdearr, loc=DG_m[0], scale=sB_m[0])
        chi2 = np.sum( (kderes - fit_func)**2. )
        ax.axhline(0.,      c="b", ls="dashed")
        ax.axhline(DG_m[0], c="orange", ls="dashed")
        # Beautify the plot
        ax.set_ylabel(r"$G-P(BP-RP)$  [mag]") 
        ax.set_xlabel(r"Density") 
        ax2.set_xlim(0, np.max([1.2*np.max(kderes/30), 5.])) 
        ax2.set_yticks([]) 
        ax.set_ylim(edges[1], edges[0]) 
        plt.text(0.25, 0.06, r"$\sigma_{{\rm MS}}={{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(sM_m[0],sM_m[1],sM_m[2]), 
                 horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=20)
        plt.text(0.25, 0.5, r"$\Delta G={{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(DG_m[0],DG_m[1],DG_m[2]), 
                 horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=20)
        plt.text(0.25, 0.75, r"$\sigma_{{\rm BS}}={{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(sB_m[0],sB_m[1],sB_m[2]), 
                 horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=20)
        plt.savefig(plotdir+"cmd_fit_residuals_"+clusname+".png")
        plt.tight_layout()
        plt.close()

        #  save / return the results as a one-row astropy table
        t = Table(names=('clus', 'nb_MS_members', 'MS_width', 'nsteps', 'burnin', 'acc_frac', 
                         'fb_50', 'fb_sigu', 'fb_sigl', 'DG_50', 'DG_sigu', 'DG_sigl', 
                         'sM_50', 'sM_sigu', 'sM_sigl', 'sB_50', 'sB_sigu', 'sB_sigl', 'chi2'), 
                  dtype=('S20', 'i4', 'f4', 'i4', 'i4', 'f4', 
                         'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
        t.add_row((clusname, len(x), np.max(x)-np.min(x), nsteps, burnin, np.round(acc_frac,2), 
                   fb_m[0], fb_m[1], fb_m[2], DG_m[0], DG_m[1], DG_m[2], 
                   sM_m[0], sM_m[1], sM_m[2], sB_m[0], sB_m[1], sB_m[2], chi2))        
        if save_results:
            t.write(plotdir + clusname + "_mcmc_res.fits")
        else:
            return t
    else:
        print("Not converged. You need to work on your code?")
        return None

def run_pipeline_tarricq(ocdir="Tarricq_selected_members_OCs/",
                 restabledir="data/mcmc_results_summary.fits",
                 plotdir="im/mcmc/", ii0=0):
    """
    Execute the MCMC model loop for all the files in ocdir.
    Save the tabulated results in restabledir and the plots in plotdir.
    """
    # Read (partial) results table if it exists
    if ii0 !=0:
        restable = Table.read(restabledir)
    # Read the list of files:
    files = sorted(os.listdir(ocdir))    
    # Loop it away
    for ii in np.arange(ii0, len(files)):
        t = Table.read(ocdir + files[ii], format="ascii")
        x    = t["bp_rp"]
        xerr = t["bp_rp_error"]
        y    = t["phot_g_mean_mag"]
        yerr = t["phot_g_mean_mag_error"]
        clusname = files[ii][:-27]

        print("No.", ii, ":", clusname, "(", len(t), "stars )")

        # Initial guess for ML fit
        theta0 = [0.6, 5.67, -0.66, -0.32, 0.09, 0., 0., 0.2, -0.7, 0.05, 0.05]
        Nfit   = len(theta0)
        # First find the maximum likelihood values for the simple fit model.
        theta0 = maxlike(theta0, x, y, yerr)
        # Now run MCMC
        result = run_mcmc(x, y, xerr, yerr, theta0, nwalkers=32, nsteps=8000, burnin=3000,
                          plotdir=plotdir, clusname=clusname, save_results=False)
        # Save the results
        if ii == 0:
            restable = result
        else:
            if result == None:
                pass
            else:
                restable = vstack([restable, result])
        restable.write(restabledir, overwrite=True)
    
def run_pipeline_cantat(ocdir="data/Cantat_selected_members_OCs_younger50Myr/",
                 restabledir="data/mcmc_results_cantat_summary.fits",
                 plotdir="im_mcmc/im_mcmc_cantat/", ii0=0):
    """
    Execute the MCMC model loop for all the files in ocdir.
    Save the tabulated results in restabledir and the plots in plotdir.
    """
    # Read (partial) results table if it exists
    if ii0 !=0:
        restable = Table.read(restabledir)
    # Read the list of files:
    files = sorted(os.listdir(ocdir))    
    # Loop it away
    for ii in np.arange(ii0, len(files)):
        t = Table.read(ocdir + files[ii], format="ascii")
        x    = t["BP-RP"]
        xerr = np.sqrt( (2.5 * np.log10(1 + 1./t["phot_bp_mean_flux_over_error"]))**2. + 
                        (2.5 * np.log10(1 + 1./t["phot_rp_mean_flux_over_error"]))**2. )
        y    = t["Gmag"]
        yerr = 2.5 * np.log10(1 + 1./t["phot_g_mean_flux_over_error"])
        clusname = files[ii][:-26]

        print("No.", ii, ":", clusname, "(", len(t), "stars )")

        # Initial guess for ML fit
        theta0 = [0.6, 5.67, -0.66, -0.32, 0.09, 0., 0., 0.2, -0.7, 0.05, 0.05]
        Nfit   = len(theta0)
        # First find the maximum likelihood values for the simple fit model.
        theta0 = maxlike(theta0, x, y, yerr)
        # Now run MCMC
        try:
            result = run_mcmc(x, y, xerr, yerr, theta0, nwalkers=32, nsteps=8000, burnin=3000,
                              plotdir=plotdir, clusname=clusname, save_results=False)
        except:
            pass
        # Save the results
        if result == None:
            pass
        else:
            if ii == 0:
                restable = result
            else:
                restable = vstack([restable, result])
            restable.write(restabledir, overwrite=True)
    
def run_pipeline_sims(ocdir="data/GOG_Simulations/Simulation_selected_members_OCs_0/",
                      restabledir="data/mcmc_results_summary_gausspriors_simulation_0.fits",
                      plotdir="im_mcmc/mcmc_gausspriors_simulation/", ii0=0):
    """
    Execute the MCMC model loop for all the files in ocdir.
    Save the tabulated results in restabledir and the plots in plotdir.
    """
    # Read (partial) results table if it exists
    if ii0 !=0:
        restable = Table.read(restabledir)
    # Read the list of files:
    files = sorted(os.listdir(ocdir))    
    # Loop it away
    for ii in np.arange(ii0, len(files)):
        t = Table.read(ocdir + files[ii], format="ascii")
        x    = t["BP-RP"]
        xerr = Table.Column([ t["sigBp"][ii]+t["sigRp"][ii] for ii in np.arange(0, len(t))])
        y    = t["G"]
        yerr = t["sigG"]
        clusname = files[ii][:-25]

        print("No.", ii, ":", clusname, "(", len(t), "stars )")

        # Initial guess for ML fit
        theta0 = [0.6, 5.67, -0.66, -0.32, 0.09, 0., 0., 0.2, -0.7, 0.05, 0.05]
        Nfit   = len(theta0)
        # First find the maximum likelihood values for the simple fit model.
        theta0 = maxlike(theta0, x, y, yerr)
        # Now run MCMC
        result = run_mcmc(x, y, xerr, yerr, theta0, nwalkers=32, nsteps=8000, burnin=3000,
                          plotdir=plotdir, clusname=clusname, save_results=False)
        # Save the results
        if ii == 0:
            restable = result
        else:
            if result == None:
                pass
            else:
                restable = vstack([restable, result])
        restable.write(restabledir, overwrite=True)

if __name__ == "__main__":
    # Run the pipeline for the Tarricq+2022 clusters
    run_pipeline_tarricq(ocdir="data/Tarricq_selected_members_OCs/",
                 restabledir="data/mcmc_results_tarricq_newpriors3.fits",
                 plotdir="im_mcmc/im_mcmc_tarricq_newpriors3/", ii0=0)
"""
if __name__ == "__main__":
    # Run the pipeline for the Cantat+2020 clusters
    run_pipeline_cantat(ocdir="data/Cantat_selected_members_OCs_younger50Myr/",
                 restabledir="data/mcmc_results_cantat_summary.fits",
                 plotdir="im_mcmc/im_mcmc_cantat/", ii0=0)
"""
