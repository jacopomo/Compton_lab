# mc/physics/materials.py

import os
import numpy as np
from scipy.interpolate import CubicSpline

_THIS_DIR = os.path.dirname(__file__)
_MAT_DATA_DIR = os.path.join(_THIS_DIR, "material_data")

# cache for loaded materials
_MATERIAL_CACHE = {}

def _parse_material_file(path: str):
    """
    Parse material file with format:
    Material density: <rho>
    header lines...
    data rows: Energy    Incoher.    Photoel.    Tot. wo/ ...
    returns: dict with density, energy_array, sigma_compton, sigma_photoel
    """
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # find density line
    density = None
    for ln in lines[:10]:
        if "Material density" in ln:
            # extract the number after ':'
            try:
                density = float(ln.split(":")[1].strip())
            except Exception:
                raise ValueError(f"Could not parse density from line: {ln}")
            break
    if density is None:
        raise ValueError(f"Density line not found in {path}")

    # find the first data line (skip header rows). We'll detect the first line that starts with a number.
    data_lines = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # if it starts with a digit or a dot or a minus sign in scientific notation -> consider data
        if s[0].isdigit() or s[0] in "+-.":
            # ensure it has numeric columns separated by whitespace or tabs
            parts = s.split()
            # require at least 3 numeric columns (energy + two cross-sections)
            try:
                # try parsing the first three tokens as floats
                float(parts[0])
                float(parts[1])
                float(parts[2])
                data_lines.append(parts)
            except Exception:
                # not a numeric data line (probably header)
                continue

    if len(data_lines) == 0:
        raise ValueError(f"No numeric data lines found in {path}")

    arr = np.array([[float(x) for x in row] for row in data_lines], dtype=float)
    # Assume columns: energy, Compton, photoelectric, ...
    energy = arr[:, 0].copy() * 1000    # keV
    sigma_compton = arr[:, 1].copy()    # cm^2 / g
    sigma_photoel = arr[:, 2].copy()    # cm^2 / g

    # sort by energy ascending (some files may already be sorted)
    sort_idx = np.argsort(energy)
    energy = energy[sort_idx]
    sigma_compton = sigma_compton[sort_idx]
    sigma_photoel = sigma_photoel[sort_idx]

    return {
        "density": density,
        "energy": energy,
        "sigma_compton": sigma_compton,
        "sigma_photoel": sigma_photoel,
    }


def _make_loglog_spline(x, y, kind="cubic", allow_extrapolate=True):
    """
    Build an interpolator in log-log space.
    x, y must be positive; if zeros encountered in y they are replaced by small positive value.
    Returns a callable f(E_arr)->y_arr (same shape).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # ensure strictly positive y for log scale
    y_minpos = np.nanmax([np.min(y[y > 0]) if np.any(y > 0) else np.nan, 1e-300])
    y_safe = np.where(y > 0, y, y_minpos * 1e-6)

    lx = np.log(x)
    ly = np.log(y_safe)

    cs = CubicSpline(lx, ly, extrapolate=allow_extrapolate)
    def f(E):
        E = np.asarray(E, dtype=float)
        out = np.zeros_like(E, dtype=float)
        positive = E > 0.0
        if np.any(positive):
            logE = np.log(E[positive])
            out[positive] = np.exp(cs(logE))

        out[~positive] = 0.0
        return out
    return f

class Material:
    """
    Material loader and interpolators.
    Access methods:
      - material.sigma_compton(E) # returns cm^2/g, vectorized
      - material.sigma_photoel(E) # returns cm^2/g
      - material.sigma_total(E)   # returns cm^2/g (compton + photoel; could add coherent if present)
      - material.mfp_compton(E)   # returns mean free path for Compton (1 / Sigma_compton) in cm
      - material.mfp_pe(E)        # similar for photoelectric (1 / Sigma_pe)
    """

    def __init__(self, name: str):
        name_clean = name.strip()
        key = name_clean.lower()
        if key in _MATERIAL_CACHE:
            # return cached object fields
            cached = _MATERIAL_CACHE[key]
            # shallow copy of cached structure
            self.__dict__.update(cached.__dict__)
            return

        # find file
        filename = os.path.join(_MAT_DATA_DIR, f"{name_clean}.txt")
        if not os.path.exists(filename):
            # try case-insensitive search
            found = None
            for f in os.listdir(_MAT_DATA_DIR):
                if f.lower().startswith(name_clean.lower()):
                    found = os.path.join(_MAT_DATA_DIR, f)
                    break
            if found is None:
                raise FileNotFoundError(f"Material data file for '{name}' not found in {_MAT_DATA_DIR}")
            filename = found

        data = _parse_material_file(filename)
        self.name = name_clean
        self.density = float(data["density"])            # g/cm^3
        energy = data["energy"]
        sig_compt = data["sigma_compton"]
        sig_pe = data["sigma_photoel"]

        # build log-log splines for microscopic cross sections (cm^2/g)
        self._sigma_compton_fun = _make_loglog_spline(energy, sig_compt, kind="cubic", allow_extrapolate=True)
        self._sigma_pe_fun = _make_loglog_spline(energy, sig_pe, kind="cubic", allow_extrapolate=True)

        # convenience: total mass-attenuation (compton + pe)
        def sigma_total_fun(E):
            return self._sigma_compton_fun(E) + self._sigma_pe_fun(E)
        self._sigma_total_fun = sigma_total_fun

        # cache
        _MATERIAL_CACHE[key] = self

    # microscopic mass attenuation coefficients (cm^2 / g)
    def sigma_compton(self, E):
        """Return Compton (Compton) mass attenuation coefficient in cm^2/g for E (array or scalar)."""
        return self._sigma_compton_fun(E)

    def sigma_photoelectric(self, E):
        """Return photoelectric mass attenuation coefficient in cm^2/g for E (array or scalar)."""
        return self._sigma_pe_fun(E)

    def sigma_total(self, E):
        """Return total mass attenuation coefficient (compton + photoel) in cm^2/g."""
        return self._sigma_total_fun(E)

    # mean free paths [cm]
    def mfp_compton(self, E):
        """Return mean free path for Compton (cm). If Sigma==0 returns np.inf (or raises)."""
        Sigma = self.sigma_compton(E) * self.density
        with np.errstate(divide="ignore", invalid="ignore"):
            mfp = 1.0 / Sigma
        return mfp

    def mfp_pe(self, E):
        Sigma = self.sigma_photoelectric(E) * self.density
        with np.errstate(divide="ignore", invalid="ignore"):
            mfp = 1.0 / Sigma
        return mfp


# convenience loader
def load_material(name: str) -> Material:
    """
    Load (or get cached) Material by name (file <name>.txt in material_data).
    Example: mat = load_material('C') or load_material('NaI')
    """
    return Material(name)
