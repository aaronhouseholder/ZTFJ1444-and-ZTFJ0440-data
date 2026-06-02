# ZTF J0440+2325 and ZTF J1444+4820: Data and Code

Data products and analysis code for Householder et al. (2026), "Stars stably accreting from substellar objects," Nature Astronomy.

## Repository structure

```
J0440/
  ESI data/              Keck/ESI reduced spectra (.npz, 15 epochs)
  LRIS data/Blue/        Keck/LRIS blue-arm reduced spectra (.spec, 17 epochs)
  LRIS data/Red/         Keck/LRIS red-arm reduced spectra (.spec, 17 epochs)
  radial_velocities.txt  Na I doublet RVs from ESI and LRIS (gamma-subtracted)
  u1.dat, g1.dat, ...    HiPERCAM phase-folded light curves (u, g, r, i, z)

J1444/
  LRIS data/Blue/        Keck/LRIS blue-arm reduced spectra (.spec, 15 epochs)
  LRIS data/Red/         Keck/LRIS red-arm reduced spectra (.spec, 20 epochs)
  radial_velocities.txt  Na I doublet RVs from LRIS red-arm (gamma-subtracted)
  hipercam_band_*.txt    HiPERCAM phase-folded light curves (u, g, r, i, z)

massradius.py            MCMC for component masses and radii (Table 1)
roche_geometry_new.py    Ballistic stream trajectory simulation (Figure 3)
BD_MESA_complete_set.data  MESA brown dwarf evolutionary tracks
```

## Spectra format

LRIS `.spec` files have 8 columns: wavelength (air, angstroms), flux, sky flux, flux uncertainty, x pixel, y pixel, response, flag. ESI `.npz` files contain wavelength and flux arrays from merged echelle orders.

## Radial velocities

RV tables give systemic-subtracted velocities (dRV). Columns: dataset or frame number, MJD, BJD_TDB, orbital phase, dRV (km/s), and uncertainty (km/s). For J0440, both ESI and LRIS measurements are included.

## Running the code

The MCMC (`massradius.py`) requires `numpy`, `scipy`, `emcee`, and `matplotlib`. It reads `BD_MESA_complete_set.data` for brown dwarf radii. Run with `python3 massradius.py` to reproduce Table 1 posteriors.

The ballistic stream simulation (`roche_geometry_new.py`) requires `numpy`, `scipy`, and `matplotlib`. Run with `python3 roche_geometry_new.py` to reproduce Figure 3.

## External resources

BT-Settl atmosphere models used for spectral fitting are available from the [SVO Theoretical Spectra Server](https://svo2.cab.inta-csic.es/theory/newov2/index.php). The PyHammer spectral typing templates are available at [PyHammer](https://github.com/BU-hammerteam/PyHammer).

## Raw data

Raw Keck/ESI and Keck/LRIS data are publicly available through the [Keck Observatory Archive](https://koa.ipac.caltech.edu). ZTF photometry is available from the [ZTF public data release](https://www.ztf.caltech.edu). Swift/XRT observations are available through the Swift mission archive.
