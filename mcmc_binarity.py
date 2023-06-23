import numpy as np
import os
from astropy.table import Table, vstack

from scipy import interpolate
from scipy.stats import gaussian_kde, norm, kstest
import scipy.optimize as op
import time
import emcee
import corner
from multiprocessing import Pool
import parsec20_tools, extinction

from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from astroML.plotting import setup_text_plots
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
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

def cmd_plot(clusname, x, y, xerr, yerr, samples, results, savefilename, color_background=False):
    # CMD plot
    plt.figure(figsize=(8,8))
    plt.errorbar(x,y,xerr=xerr, yerr=yerr, ls='none', label=None, c="grey", zorder=0)
    plt.scatter(x,y, label=None, c="w", edgecolor="k")
    xarr = np.linspace(np.min(x),np.max(x),100)
    fb_simple=[]
    # Plot some MCMC samples onto the data.
    for p0,p1,p2,p3,p4,p5,p6,fb,DG,sM,sB in samples[-30:]:
        ms = np.poly1d([p6,p5,p4,p3,p2,p1,p0])
        plt.plot(xarr, ms(xarr), color="b", alpha=0.2)
        bs = np.poly1d([p6,p5,p4,p3,p2,p1,p0+DG])
        plt.plot(xarr, bs(xarr), color="orange", alpha=0.2)
        binaries = (y < ms(x) - 3 * sM)
        fb_simple.append(np.sum(binaries)/len(x))
        plt.scatter(x[binaries],y[binaries], label=None, c="darkorange", s=30, lw=0.3, alpha=.04, edgecolor="k")
    plt.plot(xarr, ms(xarr), c='blue', label="MCMC main sequence samples", alpha=0.2)
    plt.plot(xarr, bs(xarr), c='orange', label="MCMC binary sequence samples", alpha=0.2)
    plt.xlabel(r"$BP-RP$", fontsize=22)
    plt.ylabel(r"$G$", fontsize=22)
    plt.legend(loc="upper right", fontsize=19)
    # Annotate name of the cluster and binary fraction
    ax = plt.gca()
    plt.text(0.1, 0.2, clusname.replace("_", " "), horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, fontsize=22)
    plt.text(0.1, 0.1, r"$f_b={{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(results["fb_50"][0],
                                                                             results["fb_sigu"][0],
                                                                             results["fb_sigl"][0]), 
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=22)
    ax.invert_yaxis()
    # If fit quality is bad, mark background 
    if color_background:
        ax.set_facecolor("mistyrose")
    plt.tight_layout()
    plt.savefig(savefilename)
    # Return the simplistic binary fraction statistics
    return [np.round(np.median(fb_simple),3), 
            np.round(np.percentile(fb_simple, 84)-np.median(fb_simple),3), 
            np.round(np.median(fb_simple)-np.percentile(fb_simple, 16),3)]
    
def cmd_residuals_plot(clusname, x, y, samples, results, savefilename, color_background=False):
    plt.figure(figsize=(4,8))
    ax = plt.subplot(111)
    # first of all, the base transformation of the data points is needed
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(-90)

    edges = [-1.3,.5]

    kdearr = np.linspace(edges[0], edges[1],1000)
    kderes = np.zeros(1000)

    for p0,p1,p2,p3,p4,p5,p6,fb,DG,sM,sB in samples[-30:]:
        ms = np.poly1d([p6,p5,p4,p3,p2,p1,p0])
        # Compute Delta G_i
        DGi  = (y - ms(x))[(y - ms(x)<edges[1]) & (y - ms(x)>edges[0])]
        vals, bins, patches = plt.hist(-DGi, density=True, histtype="step", bins="fd", lw=2.5,
                                       label="Gaia DR2 open clusters ($d < 1.5$ kpc)", alpha=.2, 
                                       color="grey", transform= rot + base)
        kde = gaussian_kde(y - ms(x), bw_method=.1)
        kderes = kderes + kde.evaluate(kdearr)
    plt.plot(-kdearr, kderes/30, c="k", lw=3, alpha=1, transform= rot + base)

    # Overplot the 2 Gaussians #fb_m,DG_m,sS_m,sB_m
    plt.plot(-kdearr, (1-results["fb_50"][0]) * norm.pdf(kdearr, loc=0., scale=results["sS_50"][0]), 
             c="b", lw=3, alpha=1, transform= rot + base)
    plt.plot(-kdearr,   results["fb_50"][0]   * norm.pdf(kdearr, loc=results["DG_50"][0], scale=results["sB_50"][0]), 
             c="orange", lw=3, alpha=1, transform= rot + base)

    ax.axhline(   0.,               c="b", ls="dashed")
    ax.axhline(results["DG_50"][0], c="orange", ls="dashed")
    
    # Compute 1-sided KS test
    cdf = (1-results["fb_50"][0]) * norm.cdf(kdearr, loc=0., scale=results["sS_50"][0]) + \
            results["fb_50"][0]   * norm.cdf(kdearr, loc=results["DG_50"][0], scale=results["sB_50"][0])
    ks_stat, pvalue = kstest(kde.resample(len(x)).flatten(), cdf)
    
    # Beautify the plot
    ax2 = ax.twinx() 
    ax2.set_ylabel(r"$G-P(BP-RP)$  [mag]") 
    ax.set_xlabel(r"Density") 
    ax2.set_xlim(0, np.max([1.2*np.max(kderes/30), 5.])) 
    ax2.set_yticks([]) 
    ax.set_ylim(edges[1], edges[0]) 

    if color_background:
        ax.set_facecolor("mistyrose")

    plt.text(0.25, 0.12, 
             r"$\sigma_{{\rm MS}}={{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(results["sS_50"][0],
                                                                                 results["sS_sigu"][0],
                                                                                 results["sS_sigl"][0]), 
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=20)
    plt.text(0.25, 0.5, 
             r"$\Delta G={{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(results["DG_50"][0],
                                                                        results["DG_sigu"][0],
                                                                        results["DG_sigl"][0]), 
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=20)
    plt.text(0.25, 0.75, 
             r"$\sigma_{{\rm BS}}={{{0:.2f}}}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(results["sB_50"][0],
                                                                                 results["sB_sigu"][0],
                                                                                 results["sB_sigl"][0]), 
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=20)
    plt.savefig(savefilename)
    return pvalue

def run_mcmc(x, y, xerr, yerr, theta0, nwalkers=32, nsteps=10000, burnin=5000,
             plotdir="../im/mcmc/", clusname="test", save_results=False, plot_max_likelihood=False,
             OC_selection_conditions=[10., 10., 10., 10.]):
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
        p0_m,p1_m,p2_m,p3_m,p4_m,p5_m,p6_m,fb_m,DG_m,sS_m,sB_m = map(lambda v: (np.round(v[1],3), 
                                                                        np.round(v[2]-v[1],3), 
                                                                        np.round(v[1]-v[0],3)), 
                                                             zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        # Check if the fits are "ok"
        ok_fit = ( abs(DG_m[0]+0.75) <= OC_selection_conditions[0] and 
                   sB_m[0]           <= OC_selection_conditions[1] and 
                   sS_m[0]           <= OC_selection_conditions[2] and 
                   fb_m[1] + fb_m[2] <= OC_selection_conditions[3] )
        # Save the last 100 samples
        np.savez(plotdir + "samples_mcmc_" + clusname + ".npz", samples = samples[-100:])
        
        # Turn results into a one-row astropy table
        t = Table(names=('clus', 'nb_MS_members', 'MS_width', 'nsteps', 'burnin', 'acc_frac', 
                         'fb_50', 'fb_sigu', 'fb_sigl', 'DG_50', 'DG_sigu', 'DG_sigl', 
                         'sS_50', 'sS_sigu', 'sS_sigl', 'sB_50', 'sB_sigu', 'sB_sigl', 
                         'KS_pvalue', 'fb_3sScut_50', 'fb_3sScut_sigu', 'fb_3sScut_sigl', 'ok_fit'), 
                  dtype=('S20', 'i4', 'f4', 'i4', 'i4', 'f4', 
                         'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 
                         'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 
                         'f4', 'f4', 'f4', 'f4', 'i4'))
        t.add_row((clusname, len(x), np.max(x)-np.min(x), nsteps, burnin, np.round(acc_frac,2), 
                   fb_m[0], fb_m[1], fb_m[2], DG_m[0], DG_m[1], DG_m[2], 
                   sS_m[0], sS_m[1], sS_m[2], sB_m[0], sB_m[1], sB_m[2], 
                   0.0, 0.0, 0.0, 0.0, ok_fit ))       

        # MCMC iteration plot
        plt.clf()
        fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 3*ndim))
        poly_labels    = ["$p_0$", "$p_1$", "$p_2$", "$p_3$", "$p_4$", "$p_5$", "$p_6$", "$p_7$", "$p_8$"]
        mixture_labels = ["$f_b$","$\Delta G$", "$\sigma_{SS}$", "$\sigma_{BS}$"]
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
        plt.suptitle(clusname.replace("_", " "), fontsize=25)
        fig.savefig(plotdir+"mcmc_fit_corner_"+clusname+".png")
        plt.close()
        
        # CMD plot
        fb_simple    = cmd_plot(clusname, x, y, xerr, yerr, samples, t, 
                                savefilename=plotdir+"cmd_fit_mcmc_"+clusname+".png", 
                                color_background=1-ok_fit)
        t['fb_3sScut_50'][0]   = fb_simple[0]
        t['fb_3sScut_sigu'][0] = fb_simple[1]
        t['fb_3sScut_sigl'][0] = fb_simple[2]
        plt.close()
        
        # CMD residual plot --> Save also the p-value of the KS test
        KS_pvalue = cmd_residuals_plot(clusname, x, y, samples, t, 
                                             savefilename=plotdir+"cmd_fit_residuals_"+clusname+".png", 
                                             color_background=1-ok_fit)
        t["KS_pvalue"][0] = KS_pvalue
        plt.close()
    
        # save / return the results
        if save_results:
            t.write(plotdir + clusname + "_mcmc_res.fits")
        else:
            return t
    else:
        print("Not converged. You need to work on your code?")
        return None

def run_pipeline(which = "tarricq", OC_selection_conditions=[0.05, 0.25, 0.2, 0.5], ii0=0):
    """
    Execute the MCMC model loop for all the files for either the "tarricq", "cantat", or "sim" OCs.
    Save the tabulated results in restabledir and the plots in plotdir.
    """
    if which == "tarricq":
        ocdir    = "data/Tarricq_selected_members_OCs/"
        plotdir  = "im_mcmc/im_mcmc_tarricq/"
        restabledir = "data/mcmc_results_tarricq_summary.fits"
    elif which == "cantat":
        ocdir    = "data/Cantat_selected_members_OCs_younger50Myr/"
        plotdir  = "im_mcmc/im_mcmc_cantat/"
        restabledir = "data/mcmc_results_cantat_summary.fits"
    elif which == "sims":
        ocdir    = "data/GOG_Simulations/Simulation_selected_members_OCs_9/"
        plotdir  = "im_mcmc/mcmc_gausspriors_simulation/"
        restabledir = "data/mcmc_results_summary_simulations_9.fits"
    elif which == "hunt":
        ocdir    = "data/"
        plotdir  = "im_mcmc/mcmc_gausspriors_hunt/"
        restabledir = "data/mcmc_results_hunt_summary.fits"
        isoc     = parsec20_tools.Isochrones()
    else:
        raise ValueError("Not a valid 'which' parameter")
    # Create output dir if necessary
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    # Read (partial) results table if it exists
    if ii0 !=0:
        restable = Table.read(restabledir)
        print(restable)
    if which == "hunt":
        # Read the summary table & membership table
        clusters = Table.read(ocdir+"Hunt2023_clusters.fits")
        members  = Table.read(ocdir+"Hunt2023_members.fits")
        files    = clusters["name"][ (clusters["n_stars_tidal"] >= 30) & (clusters["distance_50"] <= 2000)]
    else:
        # Read the list of files:
	    files = sorted(os.listdir(ocdir)) 
    # Loop it away
    for ii in np.arange(ii0, len(files)):
        if which == "hunt":
            t    = members[  members["name"]  == files[ii] ]
            clus = clusters[ clusters["name"] == files[ii] ][0]
            # deredden & select MS
            AG, ABP, ARP = extinction.A0_to_AG_ABP_ARP(clus["a_v_50"], 
                                                       t['phot_g_mean_mag'], t['phot_bp_mean_mag'],
                                                       t['phot_rp_mean_mag'])
            t['mg0']    =   t['phot_g_mean_mag'] - clus["distance_modulus_50"] - AG
            t['bp_rp0'] =   t['bp_rp'] - (ABP - ARP)
            mainseq = (2.9 * t['bp_rp0'] - 1.4 < t['mg0']) & \
                      (2.9 * t['bp_rp0'] + 3.4 > t['mg0']) & \
                      ( t['mg0'] > isoc.get_turnoff_G(clus["log_age_50"]) )
            t = t[ mainseq & (t["within_r_t"] > 0.9) ]
            # select CMD
            x    = t["bp_rp"]
            xerr = np.sqrt( (2.5 * np.log10(1 + t["phot_bp_mean_flux_error"]/t["phot_bp_mean_flux"]))**2. + 
				            (2.5 * np.log10(1 + t["phot_rp_mean_flux_error"]/t["phot_rp_mean_flux"]))**2. )
            y    = t["phot_g_mean_mag"]
            yerr = 2.5 * np.log10(1 + t["phot_g_mean_flux_error"]/t["phot_g_mean_flux"])
            clusname = files[ii]
        else:
	        # Read the membership table
            t = Table.read(ocdir + files[ii], format="ascii")
            if which == "tarricq":
                x    = t["bp_rp"]
                xerr = t["bp_rp_error"]
                y    = t["phot_g_mean_mag"]
                yerr = t["phot_g_mean_mag_error"]
                clusname = files[ii][:-27]
            elif which == "cantat":
                x    = t["BP-RP"]
                xerr = np.sqrt( (2.5 * np.log10(1 + 1./t["phot_bp_mean_flux_over_error"]))**2. + 
                                (2.5 * np.log10(1 + 1./t["phot_rp_mean_flux_over_error"]))**2. )
                y    = t["Gmag"]
                yerr = 2.5 * np.log10(1 + 1./t["phot_g_mean_flux_over_error"])
                clusname = files[ii][:-26]
            elif which == "sims":
                x    = t["BP-RP"]
                xerr = np.sqrt( t["sigBp"]**2. + t["sigRp"]**2. )
                y    = t["G"]
                yerr = t["sigG"]
                clusname = files[ii][:-25]

        print("No.", ii, ":", clusname, "(", len(t), "stars )")

        if len(t) >=20:
            # Initial guess for ML fit to the CMD
            theta0 = [0.6, 5.67, -0.66, -0.32, 0.09, 0., 0., 0.2, -0.7, 0.05, 0.05]
            Nfit   = len(theta0)
            # First find the maximum likelihood values for the simple fit model.
            theta0 = maxlike(theta0, x, y, yerr)
            # Now run MCMC
            result = run_mcmc(x, y, xerr, yerr, theta0, 
                              plotdir=plotdir, clusname=clusname, save_results=False,
                              OC_selection_conditions=OC_selection_conditions)
            # Save the results
            if result is None:
                pass
            else:
                if 'restable' not in locals():
                    restable = result
                print(restable)
                restable.write(restabledir, overwrite=True, format="fits")
            
if __name__ == "__main__":
    # Run the pipeline for all clusters
    run_pipeline(which = "hunt",  ii0=2240)
    #run_pipeline(which = "cantat",  ii0=0)
    #run_pipeline(which = "tarricq", ii0=0)
    #run_pipeline(which = "sims",    ii0=0)
    
