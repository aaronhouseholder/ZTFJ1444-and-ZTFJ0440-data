#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import emcee
import corner

# Constants
G = 6.67430e-11
M_sun = 1.98847e30
R_sun = 6.9634e8
M_jup = 1.89813e27
R_jup = 6.9911e7  
AU = 1.496e11  # meters
M_sun_to_M_jup = M_sun / M_jup
R_sun_to_R_jup = R_sun / R_jup
R_sun_to_AU = R_sun / AU


class MESAInterpolator:
    def __init__(self, filename):
        print(f"Loading MESA models from {filename}...")
        data = np.loadtxt(filename)
        
        self.mass = data[:, 0]
        self.age = data[:, 1]
        self.log_R = data[:, 2]
        self.log_Teff = data[:, 3]
        
        points = np.column_stack([np.log10(self.mass), np.log10(self.age)])
        self.R_interp = LinearNDInterpolator(points, self.log_R)
        
        print(f"  Loaded {len(self.mass)} data points")
        print(f"  Mass range: {self.mass.min():.4f} - {self.mass.max():.4f} M_sun")
    
    def get_radius(self, M_Msun, age_years):
        log_mass = np.log10(M_Msun)
        log_age = np.log10(age_years)
        log_R = self.R_interp(log_mass, log_age)
        if np.any(np.isnan(log_R)):
            return np.nan
        return 10**log_R


MESA = MESAInterpolator('BD_MESA_complete_set.data')


def mass_radius_relation_mdwarf(M_Msun):
    """MS M-dwarf relation: R = 0.85 * M^0.85 (valid for ZAMS stars)"""
    return 0.85 * M_Msun**0.85


def get_R1_radius(M1_Msun, age_years):
    """
    Get accretor radius using power law for ALL masses.
    R = 0.85 * M^0.85 (ZAMS M-dwarf relation)
    """
    return mass_radius_relation_mdwarf(M1_Msun)


def roche_lobe_radius(M1, M2, a):
    """Eggleton's Roche lobe approximation."""
    q = M2 / M1
    R_L = a * 0.49 * q**(2/3) / (0.6 * q**(2/3) + np.log(1 + q**(1/3)))
    return R_L


def orbital_elements(M1, M2, P_orb_min):
    """Calculate orbital separation in solar radii."""
    P_orb_sec = P_orb_min * 60.0
    M_total = (M1 + M2) * M_sun
    a_m = (G * M_total * P_orb_sec**2 / (4 * np.pi**2))**(1/3)
    return a_m / R_sun


def radial_velocity_K1(M1, M2, P_orb_min, inclination_deg):
    """Calculate RV semi-amplitude in km/s."""
    P_orb_sec = P_orb_min * 60.0
    i_rad = np.radians(inclination_deg)
    M1_kg = M1 * M_sun
    M2_kg = M2 * M_sun
    M_total = M1_kg + M2_kg
    K1_mps = (2 * np.pi * G / P_orb_sec)**(1/3) * M2_kg * np.sin(i_rad) / M_total**(2/3)
    return K1_mps / 1000.0


def log_prior(params, M1_min, M1_max, age_min_Gyr, age_max_Gyr):
    """Flat prior on [M1, M2, cos_i, age]."""
    M1, M2, cos_i, age_Gyr = params
    
    if not (M1_min <= M1 <= M1_max):
        return -np.inf
    if not (0.005 <= M2 <= 0.06):
        return -np.inf
    if not (0.0 <= cos_i <= 1.0):
        return -np.inf
    if not (age_min_Gyr <= age_Gyr <= age_max_Gyr):
        return -np.inf
    if M1 <= M2:
        return -np.inf
    
    age_years = age_Gyr * 1e9
    R2_Rsun = MESA.get_radius(M2, age_years)
    if np.isnan(R2_Rsun):
        return -np.inf
    
    return 0.0


def log_likelihood(params, P_orb_min, K1_obs, K1_err):
    """Log-likelihood with exponential Roche-filling penalty."""
    M1, M2, cos_i, age_Gyr = params
    
    inclination = np.degrees(np.arccos(cos_i))
    age_years = age_Gyr * 1e9
    
    R1_Rsun = get_R1_radius(M1, age_years)
    if np.isnan(R1_Rsun):
        return -np.inf
    
    R2_pred = MESA.get_radius(M2, age_years)
    if np.isnan(R2_pred):
        return -np.inf
    
    a = orbital_elements(M1, M2, P_orb_min)
    R_L2 = roche_lobe_radius(M1, M2, a)
    
    q_prime = M1 / M2
    R_L1 = a * 0.49 * q_prime**(2/3) / (0.6 * q_prime**(2/3) + np.log(1 + q_prime**(1/3)))
    
    fracr = np.abs(R2_pred - R_L2) / R_L2
    chi2_filling = np.exp(fracr / 0.01)
    lnlike_filling = -0.5 * chi2_filling
    
    if R1_Rsun >= R_L1:
        return -np.inf
    
    if K1_obs is not None and K1_err is not None:
        K1_pred = radial_velocity_K1(M1, M2, P_orb_min, inclination)
        chi2_K1 = ((K1_pred - K1_obs) / K1_err)**2
        lnlike_K1 = -0.5 * chi2_K1
    else:
        lnlike_K1 = 0.0
    
    return lnlike_filling + lnlike_K1


def log_probability(params, P_orb_min, K1_obs, K1_err, M1_min, M1_max, 
                    age_min_Gyr, age_max_Gyr):
    """Combined log-probability."""
    lnprior = log_prior(params, M1_min, M1_max, age_min_Gyr, age_max_Gyr)
    if not np.isfinite(lnprior):
        return -np.inf
    
    lnlike = log_likelihood(params, P_orb_min, K1_obs, K1_err)
    if not np.isfinite(lnlike):
        return -np.inf
    
    return lnprior + lnlike


def run_mcmc(P_orb_min, K1_obs=None, K1_err=None, M1_min=0.03, M1_max=0.25,
             age_min_Gyr=1.0, age_max_Gyr=10.0,
             nwalkers=32, nsteps=5000, burn_in=1000):
    ndim = 4
    
    M1_init = 0.5*(M1_min + M1_max) + 0.005 * np.random.randn(nwalkers)
    M2_init = 0.03 + 0.005 * np.random.randn(nwalkers)
    cos_i_init = np.random.uniform(0.5, 1.0, nwalkers)
    age_init = 5.0 + 2.0 * np.random.randn(nwalkers)
    
    M1_init = np.clip(M1_init, M1_min + 0.001, M1_max - 0.001)
    M2_init = np.clip(M2_init, 0.01, 0.04)
    cos_i_init = np.clip(cos_i_init, 0.0, 1.0)
    age_init = np.clip(age_init, age_min_Gyr, age_max_Gyr)
    
    pos = np.array([M1_init, M2_init, cos_i_init, age_init]).T
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(P_orb_min, K1_obs, K1_err, M1_min, M1_max, age_min_Gyr, age_max_Gyr)
    )
    
    print(f"Running MCMC: {nwalkers} walkers, {nsteps} steps")
    if K1_obs is not None:
        print(f"  K1 = {K1_obs:.1f} +/- {K1_err:.1f} km/s")
    
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    return sampler


def analyze_results(sampler, P_orb_min, burn_in=1000, system_name="System"):
    """Analyze results."""
    samples = sampler.get_chain(discard=burn_in, flat=True)
    
    M1_samples = samples[:, 0]
    M2_samples = samples[:, 1]
    cos_i_samples = samples[:, 2]
    age_Gyr = samples[:, 3]
    
    i_samples = np.degrees(np.arccos(cos_i_samples))
    M1_Mjup = M1_samples * M_sun_to_M_jup
    M2_Mjup = M2_samples * M_sun_to_M_jup
    
    R1_Rsun_all = np.array([get_R1_radius(m1, age*1e9) 
                            for m1, age in zip(M1_samples, age_Gyr)])
    R1_Rjup_all = R1_Rsun_all * R_sun_to_R_jup
    
    R2_list = []
    valid_idx = []
    for idx, (m2, age) in enumerate(zip(M2_samples, age_Gyr)):
        r2 = MESA.get_radius(m2, age * 1e9)
        if not np.isnan(r2):
            R2_list.append(r2)
            valid_idx.append(idx)
    
    valid_idx = np.array(valid_idx)
    R2_samples = np.array(R2_list)
    R2_Rjup = R2_samples * R_sun_to_R_jup
    
    a_Rsun_all = np.array([orbital_elements(m1, m2, P_orb_min) 
                           for m1, m2 in zip(M1_samples, M2_samples)])
    a_Rjup_all = a_Rsun_all * R_sun_to_R_jup
    a_AU_all = a_Rsun_all * R_sun_to_AU
    
    R_L1_all = np.array([roche_lobe_radius(m2, m1, a)  # R_L1 (accretor)
                         for m1, m2, a in zip(M1_samples, M2_samples, a_Rsun_all)])
    R_L2_all = np.array([roche_lobe_radius(m1, m2, a)  # R_L2 (donor)
                         for m1, m2, a in zip(M1_samples, M2_samples, a_Rsun_all)])
    
    R1_over_a = R1_Rsun_all / a_Rsun_all
    R2_over_a = R2_samples / a_Rsun_all[valid_idx]
    R1_over_RL1 = R1_Rsun_all / R_L1_all
    R2_over_RL2 = R2_samples / R_L2_all[valid_idx]
    
    rho_sun = 1.408
    rho1_samples = M1_samples[valid_idx] / R1_Rsun_all[valid_idx]**3 * rho_sun
    rho2_samples = M2_samples[valid_idx] / R2_samples**3 * rho_sun
    
    print(f"\n{system_name} - Results")

    for i, name in enumerate(['M1', 'M2', 'inclination', 'Age (Gyr)']):
        values = [M1_samples, M2_samples, i_samples, age_Gyr][i]
        med, lo, hi = np.percentile(values, [50, 16, 84])
        print(f"{name:12s}: {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")

    print(f"\nDerived quantities:")
    
    # ACCRETOR (M1, R1)
    print(f"\nACCRETOR:")
    med, lo, hi = np.percentile(M1_samples, [50, 16, 84])
    print(f"M1 (M_sun)   : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    med, lo, hi = np.percentile(M1_Mjup, [50, 16, 84])
    print(f"M1 (M_jup)   : {med:.2f} +{hi-med:.2f} -{med-lo:.2f}")
    
    med, lo, hi = np.percentile(R1_Rsun_all, [50, 16, 84])
    print(f"R1 (R_sun)   : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    med, lo, hi = np.percentile(R1_Rjup_all, [50, 16, 84])
    print(f"R1 (R_jup)   : {med:.3f} +{hi-med:.3f} -{med-lo:.3f}")
    
    med, lo, hi = np.percentile(rho1_samples, [50, 16, 84])
    print(f"rho1 (g/cm³) : {med:.2f} +{hi-med:.2f} -{med-lo:.2f}")
    
    # DONOR (M2, R2)
    print(f"\nDONOR:")
    med, lo, hi = np.percentile(M2_samples, [50, 16, 84])
    print(f"M2 (M_sun)   : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    med, lo, hi = np.percentile(M2_Mjup, [50, 16, 84])
    print(f"M2 (M_jup)   : {med:.2f} +{hi-med:.2f} -{med-lo:.2f}")
    
    med, lo, hi = np.percentile(R2_samples, [50, 16, 84])
    print(f"R2 (R_sun)   : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    med, lo, hi = np.percentile(R2_Rjup, [50, 16, 84])
    print(f"R2 (R_jup)   : {med:.3f} +{hi-med:.3f} -{med-lo:.3f}")
    
    med, lo, hi = np.percentile(rho2_samples, [50, 16, 84])
    print(f"rho2 (g/cm³) : {med:.2f} +{hi-med:.2f} -{med-lo:.2f}")
    
    # ORBITAL PARAMETERS
    print(f"\nORBITAL PARAMETERS:")
    med, lo, hi = np.percentile(a_Rsun_all, [50, 16, 84])
    print(f"a (R_sun)    : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    med, lo, hi = np.percentile(a_Rjup_all, [50, 16, 84])
    print(f"a (R_jup)    : {med:.3f} +{hi-med:.3f} -{med-lo:.3f}")
    med, lo, hi = np.percentile(a_AU_all, [50, 16, 84])
    print(f"a (AU)       : {med:.6f} +{hi-med:.6f} -{med-lo:.6f}")
    
    # ROCHE LOBE RADII
    print(f"\nROCHE LOBE RADII:")
    med, lo, hi = np.percentile(R_L1_all, [50, 16, 84])
    print(f"R_L1 (R_sun) : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    med, lo, hi = np.percentile(R_L1_all * R_sun_to_R_jup, [50, 16, 84])
    print(f"R_L1 (R_jup) : {med:.3f} +{hi-med:.3f} -{med-lo:.3f}")
    
    med, lo, hi = np.percentile(R_L2_all[valid_idx], [50, 16, 84])
    print(f"R_L2 (R_sun) : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    med, lo, hi = np.percentile(R_L2_all[valid_idx] * R_sun_to_R_jup, [50, 16, 84])
    print(f"R_L2 (R_jup) : {med:.3f} +{hi-med:.3f} -{med-lo:.3f}")
    
    # FILLING FACTORS
    print(f"\nFILLING FACTORS:")
    med, lo, hi = np.percentile(R1_over_a, [50, 16, 84])
    print(f"R1/a         : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    med, lo, hi = np.percentile(R2_over_a, [50, 16, 84])
    print(f"R2/a         : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    
    med, lo, hi = np.percentile(R1_over_RL1, [50, 16, 84])
    print(f"R1/R_L1      : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    med, lo, hi = np.percentile(R2_over_RL2, [50, 16, 84])
    print(f"R2/R_L2      : {med:.4f} +{hi-med:.4f} -{med-lo:.4f}")
    
    # DENSITY RATIO
    print(f"\nDENSITY RATIO:")
    rho_ratio = rho1_samples / rho2_samples
    med, lo, hi = np.percentile(rho_ratio, [50, 16, 84])
    print(f"rho1/rho2    : {med:.3f} +{hi-med:.3f} -{med-lo:.3f}")
    if med < 1.0:
        print(f"  NOTE: rho1 < rho2 (accretor less dense than donor)")
        print(f"        This is allowed - Roche geometry determines donor")
    
    # Corner plot with expanded parameters
    R1_aligned = R1_Rjup_all[valid_idx]
    M1_aligned = M1_samples[valid_idx]
    M2_aligned = M2_samples[valid_idx]
    i_aligned = i_samples[valid_idx]
    age_aligned = age_Gyr[valid_idx]
    a_aligned = a_Rjup_all[valid_idx]
    R1_over_RL1_aligned = R1_over_RL1[valid_idx]
    
    samples_display = np.column_stack([M1_aligned, M2_aligned, i_aligned, 
                                       age_aligned, R1_aligned, R2_Rjup, 
                                       a_aligned, R1_over_RL1_aligned, R2_over_RL2, rho_ratio])
    
    labels = ['$M_1$ ($M_\\odot$)', '$M_2$ ($M_\\odot$)', '$i$ (deg)', 
              'Age (Gyr)', '$R_1$ ($R_{Jup}$)', '$R_2$ ($R_{Jup}$)', 
              '$a$ ($R_{Jup}$)', '$R_1/R_{L1}$', '$R_2/R_{L2}$', r'$\rho_1/\rho_2$']
    
    fig = corner.corner(samples_display, labels=labels, quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_fmt='.3f')
    fig.suptitle(f'{system_name}', y=1.00)
    plt.savefig(f'{system_name.lower().replace(" ", "_")}_corner.png', dpi=150, bbox_inches='tight')
    print(f"\nCorner plot: {system_name.lower().replace(' ', '_')}_corner.png")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x_range = (0.0, 0.13)
    bins = np.linspace(x_range[0], x_range[1], 100)
    
    for data, color, label in [(M1_samples, 'r', '$M_1$'),
                                (M2_samples, 'orange', '$M_2$'),
                                (R1_Rsun_all, 'b', '$R_1$'),
                                (R2_samples, 'cyan', '$R_2$')]:
        counts, edges = np.histogram(data, bins=bins)
        counts_norm = counts / np.sum(counts) / (edges[1] - edges[0])
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts_norm, color=color, linewidth=2, label=label)
    
    ax.set_xlim(x_range)
    ax.set_xlabel('Mass or Radius (solar)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax.legend(fontsize=13, loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{system_name.lower().replace(" ", "_")}_combined.png', dpi=300, bbox_inches='tight')
    print(f"Combined plot: {system_name.lower().replace(' ', '_')}_combined.png")
    
    # 2x4 panel plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    m2_med = np.median(M2_Mjup)
    m2_lo = np.percentile(M2_Mjup, 16)
    m2_hi = np.percentile(M2_Mjup, 84)
    axes[0, 0].hist(M2_Mjup, bins=50, alpha=0.7, edgecolor='black', color='orange',
                    label=f'${m2_med:.1f}^{{+{m2_hi-m2_med:.1f}}}_{{-{m2_med-m2_lo:.1f}}}$ $M_{{\\rm Jup}}$')
    axes[0, 0].set_xlim(15, 45)
    axes[0, 0].set_xlabel('$M_2$ ($M_{Jup}$)', fontsize=13)
    axes[0, 0].set_ylabel('Counts', fontsize=13)
    axes[0, 0].set_title('Donor Mass', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11, loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    r2_med = np.median(R2_Rjup)
    r2_lo = np.percentile(R2_Rjup, 16)
    r2_hi = np.percentile(R2_Rjup, 84)
    axes[0, 1].hist(R2_Rjup, bins=50, alpha=0.7, edgecolor='black', color='green',
                    label=f'${r2_med:.3f}^{{+{r2_hi-r2_med:.3f}}}_{{-{r2_med-r2_lo:.3f}}}$ $R_{{\\rm Jup}}$')
    axes[0, 1].set_xlim(0.6, 1.4)
    axes[0, 1].set_xlabel('$R_2$ ($R_{Jup}$)', fontsize=13)
    axes[0, 1].set_ylabel('Counts', fontsize=13)
    axes[0, 1].set_title('Donor Radius', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11, loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    rho2_med = np.median(rho2_samples)
    rho2_lo = np.percentile(rho2_samples, 16)
    rho2_hi = np.percentile(rho2_samples, 84)
    axes[0, 2].hist(rho2_samples, bins=50, alpha=0.7, edgecolor='black', color='brown',
                    label=f'${rho2_med:.1f}^{{+{rho2_hi-rho2_med:.1f}}}_{{-{rho2_med-rho2_lo:.1f}}}$ g cm$^{{-3}}$')
    axes[0, 2].set_xlim(0, 50)
    axes[0, 2].set_xlabel(r'$\rho_2$ (g cm$^{-3}$)', fontsize=13)
    axes[0, 2].set_ylabel('Counts', fontsize=13)
    axes[0, 2].set_title('Donor Density', fontsize=13, fontweight='bold')
    axes[0, 2].legend(fontsize=11, loc='upper right')
    axes[0, 2].grid(True, alpha=0.3)
    
    i_med = np.median(i_samples)
    i_lo = np.percentile(i_samples, 16)
    i_hi = np.percentile(i_samples, 84)
    axes[0, 3].hist(i_samples, bins=50, alpha=0.7, edgecolor='black', color='cyan',
                    label=f'${i_med:.1f}^{{+{i_hi-i_med:.1f}}}_{{-{i_med-i_lo:.1f}}}$ deg')
    axes[0, 3].set_xlim(30, 90)
    axes[0, 3].set_xlabel('Inclination (deg)', fontsize=13)
    axes[0, 3].set_ylabel('Counts', fontsize=13)
    axes[0, 3].set_title('Orbital Inclination', fontsize=13, fontweight='bold')
    axes[0, 3].legend(fontsize=11, loc='upper left')
    axes[0, 3].grid(True, alpha=0.3)
    
    m1_med = np.median(M1_samples)
    m1_lo = np.percentile(M1_samples, 16)
    m1_hi = np.percentile(M1_samples, 84)
    axes[1, 0].hist(M1_samples, bins=50, alpha=0.7, edgecolor='black', color='steelblue',
                    label=f'${m1_med:.3f}^{{+{m1_hi-m1_med:.3f}}}_{{-{m1_med-m1_lo:.3f}}}$ $M_\\odot$')
    axes[1, 0].set_xlim(0.03, 0.25)
    axes[1, 0].set_xlabel('$M_1$ ($M_\\odot$)', fontsize=13)
    axes[1, 0].set_ylabel('Counts', fontsize=13)
    axes[1, 0].set_title('Accretor Mass', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11, loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    
    r1_med = np.median(R1_Rjup_all)
    r1_lo = np.percentile(R1_Rjup_all, 16)
    r1_hi = np.percentile(R1_Rjup_all, 84)
    axes[1, 1].hist(R1_Rjup_all, bins=50, alpha=0.7, edgecolor='black', color='blue',
                    label=f'${r1_med:.3f}^{{+{r1_hi-r1_med:.3f}}}_{{-{r1_med-r1_lo:.3f}}}$ $R_{{\\rm Jup}}$')
    axes[1, 1].set_xlim(0.6, 1.4)
    axes[1, 1].set_xlabel('$R_1$ ($R_{Jup}$)', fontsize=13)
    axes[1, 1].set_ylabel('Counts', fontsize=13)
    axes[1, 1].set_title('Accretor Radius', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11, loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    rho1_med = np.median(rho1_samples)
    rho1_lo = np.percentile(rho1_samples, 16)
    rho1_hi = np.percentile(rho1_samples, 84)
    axes[1, 2].hist(rho1_samples, bins=50, alpha=0.7, edgecolor='black', color='purple',
                    label=f'${rho1_med:.1f}^{{+{rho1_hi-rho1_med:.1f}}}_{{-{rho1_med-rho1_lo:.1f}}}$ g cm$^{{-3}}$')
    axes[1, 2].set_xlim(0, 120)
    axes[1, 2].set_xlabel(r'$\rho_1$ (g cm$^{-3}$)', fontsize=13)
    axes[1, 2].set_ylabel('Counts', fontsize=13)
    axes[1, 2].set_title('Accretor Density', fontsize=13, fontweight='bold')
    axes[1, 2].legend(fontsize=11, loc='upper right')
    axes[1, 2].grid(True, alpha=0.3)
    
    age_med = np.median(age_Gyr)
    age_lo = np.percentile(age_Gyr, 16)
    age_hi = np.percentile(age_Gyr, 84)
    axes[1, 3].hist(age_Gyr, bins=50, alpha=0.7, edgecolor='black', color='teal',
                    label=f'${age_med:.1f}^{{+{age_hi-age_med:.1f}}}_{{-{age_med-age_lo:.1f}}}$ Gyr')
    axes[1, 3].set_xlim(0, 10)
    axes[1, 3].set_xlabel('Age (Gyr)', fontsize=13)
    axes[1, 3].set_ylabel('Counts', fontsize=13)
    axes[1, 3].set_title('System Age', fontsize=13, fontweight='bold')
    axes[1, 3].legend(fontsize=11, loc='upper right')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{system_name.lower().replace(" ", "_")}_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Distribution plots: {system_name.lower().replace(' ', '_')}_distributions.png")
    
    # Save posteriors
    filename_base = system_name.lower().replace(" ", "_")
    
    # Save full samples with all derived quantities
    np.save(f'{filename_base}_posteriors.npy', samples_display)
    print(f"Saved posteriors: {filename_base}_posteriors.npy")
    
    # Save as text file with header
    header = "M1(Msun) M2(Msun) i(deg) Age(Gyr) R1(Rjup) R2(Rjup) a(Rjup) R1/RL1 R2/RL2 rho1/rho2"
    np.savetxt(f'{filename_base}_posteriors.txt', samples_display, 
               header=header, fmt='%.6f')
    print(f"Saved posteriors: {filename_base}_posteriors.txt")
    
    # Save summary statistics
    with open(f'{filename_base}_summary.txt', 'w') as f:
        f.write(f"Posterior Summary for {system_name}\n")
        f.write(f"\n")
        param_names = ['M1 (Msun)', 'M2 (Msun)', 'i (deg)', 'Age (Gyr)', 
                      'R1 (Rjup)', 'R2 (Rjup)', 'a (Rjup)', 
                      'R1/RL1', 'R2/RL2', 'rho1/rho2']
        for i, name in enumerate(param_names):
            med, lo, hi = np.percentile(samples_display[:, i], [50, 16, 84])
            f.write(f"{name:15s}: {med:.6f} +{hi-med:.6f} -{med-lo:.6f}\n")
    print(f"Saved summary: {filename_base}_summary.txt")
    
    return samples_display


if __name__ == "__main__":
    print("MCMC with MESA - Two Systems Analysis")

    # System 1: 86.65 min with K1 constraint
    print("\nZTF J0440+2325 (86.65 min)")
    
    sampler1 = run_mcmc(
        P_orb_min=86.65,
        K1_obs=24.7,
        K1_err=2.6,
        M1_min=0.075,
        M1_max=0.09,
        age_min_Gyr=1.0,
        age_max_Gyr=10.0,
        nwalkers=32,
        nsteps=5000,
        burn_in=1000
    )

    samples1 = analyze_results(sampler1, P_orb_min=86.65, burn_in=1000,
                               system_name="ZTF J0440 86.65 min")
    
    # System 2: 67.16 min without K1 constraint
    print("\nZTF J1444+4820 (67.16 min)")
    
    sampler2 = run_mcmc(
        P_orb_min=67.16,
        K1_obs=None,
        K1_err=None,
        M1_min=0.075,
        M1_max=0.2,
        age_min_Gyr=1.0,
        age_max_Gyr=10.0,
        nwalkers=32,
        nsteps=5000,
        burn_in=1000
    )
    
    samples2 = analyze_results(sampler2, P_orb_min=67.16, burn_in=1000, 
                               system_name="ZTF J1444 67.16 min")
    
    print("\nDone.")
