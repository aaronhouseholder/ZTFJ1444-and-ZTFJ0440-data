#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import emcee

from convergence import gelman_rubin

G = 6.67430e-11
M_sun = 1.98847e30
R_sun = 6.9634e8
M_jup = 1.89813e27
R_jup = 6.9911e7
AU = 1.496e11
M_sun_to_M_jup = M_sun / M_jup
R_sun_to_R_jup = R_sun / R_jup
R_sun_to_AU = R_sun / AU


class MESAInterpolator:
    def __init__(self, filename):
        data = np.loadtxt(filename)
        self.mass = data[:, 0]
        self.age = data[:, 1]
        self.log_R = data[:, 2]
        self.log_Teff = data[:, 3]
        points = np.column_stack([np.log10(self.mass), np.log10(self.age)])
        self.R_interp = LinearNDInterpolator(points, self.log_R)

    def get_radius(self, M_Msun, age_years):
        log_R = self.R_interp(np.log10(M_Msun), np.log10(age_years))
        if np.any(np.isnan(log_R)):
            return np.nan
        return 10**log_R


MESA = MESAInterpolator('BD_MESA_complete_set.data')


def mass_radius_relation_mdwarf(M_Msun):
    return 0.85 * M_Msun**0.85


def get_R1_radius(M1_Msun, age_years):
    return mass_radius_relation_mdwarf(M1_Msun)


def roche_lobe_radius(M1, M2, a):
    q = M2 / M1
    return a * 0.49 * q**(2/3) / (0.6 * q**(2/3) + np.log(1 + q**(1/3)))


def orbital_elements(M1, M2, P_orb_min):
    P_orb_sec = P_orb_min * 60.0
    M_total = (M1 + M2) * M_sun
    a_m = (G * M_total * P_orb_sec**2 / (4 * np.pi**2))**(1/3)
    return a_m / R_sun


def radial_velocity_K1(M1, M2, P_orb_min, inclination_deg):
    P_orb_sec = P_orb_min * 60.0
    i_rad = np.radians(inclination_deg)
    M1_kg = M1 * M_sun
    M2_kg = M2 * M_sun
    M_total = M1_kg + M2_kg
    K1_mps = (2 * np.pi * G / P_orb_sec)**(1/3) * M2_kg * np.sin(i_rad) / M_total**(2/3)
    return K1_mps / 1000.0


def log_prior(params, M1_min, M1_max, age_min_Gyr, age_max_Gyr):
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
    if np.isnan(MESA.get_radius(M2, age_Gyr * 1e9)):
        return -np.inf
    return 0.0


def log_likelihood(params, P_orb_min, K1_obs, K1_err):
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
    lnlike_filling = -0.5 * np.exp(fracr / 0.01)

    if R1_Rsun >= R_L1:
        return -np.inf

    if K1_obs is not None and K1_err is not None:
        K1_pred = radial_velocity_K1(M1, M2, P_orb_min, inclination)
        lnlike_K1 = -0.5 * ((K1_pred - K1_obs) / K1_err)**2
    else:
        lnlike_K1 = 0.0

    return lnlike_filling + lnlike_K1


def log_probability(params, P_orb_min, K1_obs, K1_err, M1_min, M1_max,
                    age_min_Gyr, age_max_Gyr):
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

    M1_init = np.clip(0.10 + 0.02 * np.random.randn(nwalkers), M1_min + 0.001, M1_max - 0.001)
    M2_init = np.clip(0.03 + 0.005 * np.random.randn(nwalkers), 0.01, 0.04)
    cos_i_init = np.random.uniform(0.5, 1.0, nwalkers)
    age_init = np.clip(5.0 + 2.0 * np.random.randn(nwalkers), age_min_Gyr, age_max_Gyr)

    pos = np.array([M1_init, M2_init, cos_i_init, age_init]).T

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(P_orb_min, K1_obs, K1_err, M1_min, M1_max, age_min_Gyr, age_max_Gyr)
    )
    sampler.run_mcmc(pos, nsteps, progress=True)
    return sampler


def analyze_results(sampler, P_orb_min, burn_in=1000, system_name="System"):
    chain = sampler.get_chain(discard=burn_in)
    R_hat = gelman_rubin(chain)
    converged = "converged" if np.all(R_hat < 1.1) else "NOT converged"
    print(f"R-hat: {R_hat.round(3)} [{converged}]")

    samples = sampler.get_chain(discard=burn_in, flat=True)
    M1_samples = samples[:, 0]
    M2_samples = samples[:, 1]
    cos_i_samples = samples[:, 2]
    age_Gyr = samples[:, 3]

    i_samples = np.degrees(np.arccos(cos_i_samples))

    R1_Rsun_all = np.array([get_R1_radius(m1, age*1e9)
                            for m1, age in zip(M1_samples, age_Gyr)])
    R1_Rjup_all = R1_Rsun_all * R_sun_to_R_jup

    R2_list, valid_idx = [], []
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

    R_L1_all = np.array([roche_lobe_radius(m2, m1, a)
                         for m1, m2, a in zip(M1_samples, M2_samples, a_Rsun_all)])
    R_L2_all = np.array([roche_lobe_radius(m1, m2, a)
                         for m1, m2, a in zip(M1_samples, M2_samples, a_Rsun_all)])

    R1_over_RL1 = R1_Rsun_all / R_L1_all
    R2_over_RL2 = R2_samples / R_L2_all[valid_idx]

    rho_sun = 1.408
    rho1_samples = M1_samples[valid_idx] / R1_Rsun_all[valid_idx]**3 * rho_sun
    rho2_samples = M2_samples[valid_idx] / R2_samples**3 * rho_sun
    rho_ratio = rho1_samples / rho2_samples

    # Build aligned arrays and save
    samples_display = np.column_stack([
        M1_samples[valid_idx], M2_samples[valid_idx], i_samples[valid_idx],
        age_Gyr[valid_idx], R1_Rjup_all[valid_idx], R2_Rjup,
        a_Rjup_all[valid_idx], R1_over_RL1[valid_idx], R2_over_RL2, rho_ratio
    ])

    filename_base = system_name.lower().replace(" ", "_")

    np.save(f'{filename_base}_posteriors.npy', samples_display)

    header = "M1(Msun) M2(Msun) i(deg) Age(Gyr) R1(Rjup) R2(Rjup) a(Rjup) R1/RL1 R2/RL2 rho1/rho2"
    np.savetxt(f'{filename_base}_posteriors.txt', samples_display,
               header=header, fmt='%.6f')

    param_names = ['M1 (Msun)', 'M2 (Msun)', 'i (deg)', 'Age (Gyr)',
                  'R1 (Rjup)', 'R2 (Rjup)', 'a (Rjup)',
                  'R1/RL1', 'R2/RL2', 'rho1/rho2']
    with open(f'{filename_base}_summary.txt', 'w') as f:
        for i, name in enumerate(param_names):
            med, lo, hi = np.percentile(samples_display[:, i], [50, 16, 84])
            f.write(f"{name:15s}: {med:.6f} +{hi-med:.6f} -{med-lo:.6f}\n")

    print(f"Saved {filename_base}_posteriors.npy/.txt, _summary.txt")
    return samples_display


if __name__ == "__main__":
    sampler1 = run_mcmc(P_orb_min=86.65, K1_obs=24.7, K1_err=2.6,
                        M1_min=0.075, M1_max=0.13)
    samples1 = analyze_results(sampler1, P_orb_min=86.65,
                               system_name="ZTF J0440 86.65 min")

    sampler2 = run_mcmc(P_orb_min=67.16, M1_min=0.075, M1_max=0.2)
    samples2 = analyze_results(sampler2, P_orb_min=67.16,
                               system_name="ZTF J1444 67.16 min")
