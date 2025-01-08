# gpu_spectrum.py
# -------------------------------------------------------------------
#  Example: Keep the constructor CPU-only (so external code sees
#  standard NumPy arrays), then do GPU logic inside the RT methods.
# -------------------------------------------------------------------

# Copyright (c) 2021-2023 Patricio Cubillos
# Pyrat Bay is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'Spectrum',
    'spectrum',
    'modulation',
    'intensity',
    'flux',
    'two_stream',
]

import numpy as np
import scipy.constants as sc
import scipy.special as ss
from scipy.interpolate import interp1d

# If you want to do GPU inside methods, we can import CuPy here:
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from .. import constants as pc
from .. import io
from .. import spectrum as ps
from .. import tools as pt
from ..lib import _trapz as t


class Spectrum():
    def __init__(self, inputs, log):
        """
        Make the wavenumber sample from user inputs.
        NOTE: We do NOT use CuPy here. We stay on CPU (NumPy).
        This avoids type errors with code that expects a NumPy array.
        """
        log.head('\nGenerating wavenumber array.')

        self.specfile  = inputs.specfile  # Transmission/Emission spectrum file
        self.intensity = None            # Intensity spectrum array
        self.clear     = None            # Clear modulation spectrum (patchy)
        self.cloudy    = None            # Cloudy modulation spectrum (patchy)
        self.starflux  = None            # Stellar flux
        self.wnosamp   = None
        self.wn        = None
        self.nwave     = None
        self.spectrum  = None

        # Quadrature set-up for emission, on CPU:
        if inputs.quadrature is not None:
            self.quadrature = inputs.quadrature
            # p_roots is CPU-based
            qnodes, qweights = ss.p_roots(self.quadrature)
            # Scale from [-1,1] to [0,1]:
            qnodes = 0.5*(qnodes + 1.0)
            self.raygrid = np.arccos(np.sqrt(qnodes))
            self.quadrature_mu = np.sqrt(qnodes)
            weights = 0.5 * np.pi * qweights
        else:
            # If custom angles:
            raygrid = inputs.raygrid * sc.degree  # CPU array
            if raygrid[0] != 0.0:
                log.error('First angle in raygrid must be 0.0 (normal)')
            if np.any(raygrid < 0) or np.any(raygrid > 0.5*np.pi):
                log.error('raygrid angles must lie between 0 and 90 deg')
            if np.any(np.ediff1d(raygrid) <= 0):
                log.error('raygrid angles must be monotonically increasing')

            self.raygrid = raygrid
            self.quadrature_mu = np.cos(raygrid)
            N = len(raygrid)
            bounds = np.linspace(0, 0.5*np.pi, N+1)
            bounds[1:-1] = 0.5*(raygrid[:-1] + raygrid[1:])
            weights = np.pi*(np.sin(bounds[1:])**2 - np.sin(bounds[:-1])**2)

        self.quadrature_weights = np.expand_dims(weights, axis=1)
        self.nangles = len(self.quadrature_mu)
        self.f_dilution = inputs.f_dilution

        # Radiative-transfer path:
        self._rt_path = inputs.rt_path

        if inputs.wlunits is None:
            self.wlunits = 'um'
        else:
            self.wlunits = inputs.wlunits
        wl_units = pt.u(self.wlunits)

        # Low wavenumber boundary logic:
        if inputs.wnlow is None and inputs.wlhigh is None:
            log.error('Undefined low wavenumber boundary.  Either set wnlow or wlhigh')
        if inputs.wnlow is not None and inputs.wlhigh is not None:
            log.warning(
                f'Both wnlow ({inputs.wnlow:.2e} cm-1) and wlhigh '
                f'({inputs.wlhigh:.2e} cm) were defined.  wlhigh will be ignored'
            )

        if inputs.wnlow is not None:
            self.wnlow  = inputs.wnlow
            self.wlhigh = 1.0/self.wnlow
        else:
            self.wlhigh = inputs.wlhigh
            self.wnlow  = 1.0/self.wlhigh

        # High wavenumber boundary logic:
        if inputs.wnhigh is None and inputs.wllow is None:
            log.error('Undefined high wavenumber boundary. Either set wnhigh or wllow')
        if inputs.wnhigh is not None and inputs.wllow is not None:
            log.warning(
                f'Both wnhigh ({inputs.wnhigh:.2e} cm-1) and wllow '
                f'({inputs.wllow:.2e} cm) were defined.  wllow will be ignored'
            )

        if inputs.wnhigh is not None:
            self.wnhigh = inputs.wnhigh
            self.wllow  = 1.0/self.wnhigh
        else:
            self.wllow  = inputs.wllow
            self.wnhigh = 1.0/self.wllow

        if self.wnlow > self.wnhigh:
            log.error(
                f'Wavenumber low boundary ({self.wnlow:.1f} cm-1) must be '
                f'larger than the high boundary ({self.wnhigh:.1f} cm-1)'
            )

        # Prepare to define wavenumber array:
        self.resolution = None
        self.wlstep     = None
        self.wnstep     = None

        # If cross-section tables exist, read them on CPU:
        if pt.isfile(inputs.extfile) == 1 and inputs.runmode != 'opacity':
            wn = io.read_opacity(inputs.extfile[0], extract='arrays')[3]  # CPU numpy array

            # Crop wavenumber to [wnlow, wnhigh] on CPU:
            mask = (wn >= self.wnlow) & (wn <= self.wnhigh)
            self.wn = wn[mask]
            self.nwave = len(self.wn)
            self.spectrum = np.zeros(self.nwave, dtype=np.float64)

            # Update boundaries
            if self.nwave > 0:
                if self.wnlow <= self.wn[0]:
                    self.wnlow = self.wn[0]
                if self.wnhigh >= self.wn[-1]:
                    self.wnhigh = self.wn[-1]

            # Guess sampling
            if self.nwave > 2:
                dwn = np.ediff1d(np.abs(self.wn))
                dwl = np.ediff1d(np.abs(1.0/self.wn))
                res = np.abs(self.wn[1:]/dwn)

                std_dwn = np.std(dwn/np.mean(dwn))
                std_dwl = np.std(dwl/np.mean(dwl))
                std_res = np.std(res /np.mean(res))

                if std_dwn < std_dwl and std_dwn < std_res:
                    self.wnstep = self.wn[1]-self.wn[0]
                    sampling_text = f'sampling rate = {self.wnstep:.2f} cm-1'
                elif std_dwl < std_dwn and std_dwl < std_res:
                    self.wlstep = np.abs(1/self.wn[0] - 1/self.wn[1])/pc.um
                    sampling_text = f'sampling rate = {self.wlstep:.6f} um'
                else:
                    g = self.wn[-2]/self.wn[-1]
                    self.resolution = round(0.5*(1+g)/(1-g), 5)
                    sampling_text = f'R = {self.resolution:.1f}'
            else:
                sampling_text = 'Too few points to guess sampling'

            log.msg(
                "Reading spectral sampling from extinction-coefficient table.  "
                f"Adopting array with {sampling_text}, "
                f"and {self.nwave} samples between "
                f"[{self.wnlow:.2f}, {self.wnhigh:.2f}] cm-1."
            )
            return

        # Otherwise define from user input:
        no_sampling = (
            inputs.wnstep is None and
            inputs.wlstep is None and
            inputs.resolution is None
        )
        if no_sampling:
            log.error('Undefined spectral sampling rate (set resolution, wnstep, or wlstep)')

        # Possibly set wnstep:
        if inputs.wnstep is not None:
            self.wnstep = inputs.wnstep

        # Highly composite numbers (for oversampling):
        hcn = np.array([
            1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840,
            1260, 1680, 2160, 2520, 5040, 7560, 10080, 15120, 20160, 25200,
            27720, 45360, 50400, 55440, 83160, 110880, 221760, 277200,
        ])

        if inputs.wnosamp is None:
            if self.wnstep is None:
                self.wnstep = 1.0
            # find the first hcn dividing wnstep with ratio <= 0.0004
            self.wnosamp = hcn[self.wnstep/hcn <= 0.0004][0]
        else:
            self.wnosamp = inputs.wnosamp

        self.resolution = inputs.resolution
        self.wlstep     = inputs.wlstep

        # Build self.wn on CPU:
        if self.resolution is not None:
            # constant-resolving-power sampling:
            wn_array = ps.constant_resolution_spectrum(
                self.wnlow, self.wnhigh, self.resolution
            )
            self.wn = np.asarray(wn_array)
            self.wlstep = None
        elif self.wlstep is not None:
            # constant wavelength sampling => convert to wavenumbers
            wl = np.arange(self.wllow, self.wlhigh, self.wlstep)
            # wavenumber = 1/wl, flipping if needed
            self.wn = 1.0/np.flip(wl)
            self.wnlow = float(self.wn[0])
            self.resolution = None
        else:
            # constant wavenumber sampling
            nwave = int((self.wnhigh - self.wnlow)/self.wnstep) + 1
            self.wn = self.wnlow + np.arange(nwave)*self.wnstep

        self.nwave = self.wn.size
        self.spectrum = np.zeros(self.nwave, dtype=np.float64)

        # Fine-sampled wavenumber array:
        if self.wnstep is None or self.wnosamp == 0:
            self.wnstep = 1.0   # fallback
            self.wnosamp = 1
        self.ownstep = self.wnstep/self.wnosamp
        if not np.isfinite(self.ownstep):
            self.ownstep = 1e-6
        self.onwave = int(np.ceil((self.wn[-1]-self.wnlow)/self.ownstep)) + 1
        self.own = self.wnlow + np.arange(self.onwave)*self.ownstep
        self.odivisors = pt.divisors(self.wnosamp)

        # Re-set final boundary
        if self.wn.size > 0 and self.wn[-1] != self.wnhigh:
            log.warning(
                f'Final wavenumber modified from {self.wnhigh:.4f} cm-1 (input) '
                f'to {self.wn[-1]:.4f} cm-1'
            )

        # Print info:
        log.msg(
            f'Initial wavenumber boundary:  {self.wnlow:.5e} cm-1  '
            f'({self.wlhigh/wl_units:.3e} {self.wlunits})\n'
            f'Final   wavenumber boundary:  {self.wnhigh:.5e} cm-1  '
            f'({self.wllow/wl_units:.3e} {self.wlunits})',
            indent=2,
        )
        if self.resolution is not None:
            msg = f'Spectral resolving power: {self.resolution:.1f}'
        elif self.wlstep is not None:
            wl_step = self.wlstep/wl_units
            msg = f'Wavelength sampling interval: {wl_step:.2g} {self.wlunits}'
        else:
            msg = f'Wavenumber sampling interval: {self.wnstep:.2g} cm-1'
        log.msg(
            f'{msg}\n'
            f'Wavenumber sample size:      {self.nwave:8d}\n'
            f'Wavenumber fine-sample size: {self.onwave:8d}\n',
            indent=2,
        )
        log.head('Wavenumber sampling done.')


    def __str__(self):
        fmt = {'float': '{: .3e}'.format}
        fw = pt.Formatted_Write()
        fw.write('Spectral information:')
        fw.write('Wavenumber internal units: cm-1')
        fw.write('Wavelength internal units: cm')
        fw.write('Wavelength display units (wlunits): {:s}', self.wlunits)
        fw.write(
            'Low wavenumber boundary (wnlow):   {:10.3f} cm-1  '
            '(wlhigh = {:6.2f} {})',
            self.wnlow, self.wlhigh/pt.u(self.wlunits), self.wlunits,
        )
        fw.write(
            'High wavenumber boundary (wnhigh): {:10.3f} cm-1  '
            '(wllow  = {:6.2f} {})',
            self.wnhigh, self.wllow/pt.u(self.wlunits), self.wlunits,
        )
        fw.write('Number of samples (nwave): {:d}', self.nwave)
        if self.resolution is None:
            fw.write('Sampling interval (wnstep): {:.3f} cm-1', self.wnstep)
        else:
            fw.write('Spectral resolving power (resolution): {:.1f}', self.resolution)

        fw.write(
            'Wavenumber array (wn, cm-1):\n    {}',
            self.wn,
            fmt={'float': '{: .3f}'.format},
        )
        fw.write('Oversampling factor (wnosamp): {:d}', self.wnosamp)

        fw.write(
            '\nGaussian quadrature cos(theta) angles (quadrature_mu):\n    {}',
            self.quadrature_mu,
            prec=3,
        )
        fw.write(
            'Gaussian quadrature weights (quadrature_weights):\n    {}',
            self.quadrature_weights.flatten(),
            prec=3,
        )
        if self.intensity is not None:
            fw.write('Intensity spectra (intensity, erg s-1 cm-2 sr-1 cm):')
            for intensity_row in self.intensity:
                fw.write('    {}', intensity_row, fmt=fmt, edge=3)

        # Print out final spectrum depending on RT path:
        if self._rt_path in pc.emission_rt:
            fw.write(
                'Emission spectrum (spectrum, erg s-1 cm-2 cm):\n    {}',
                self.spectrum, fmt=fmt, edge=3,
            )
        elif self._rt_path in pc.transmission_rt:
            fw.write(
                '\nTransmission spectrum, (Rp/Rs)**2 (spectrum):\n    {}',
                self.spectrum, fmt=fmt, edge=3,
            )
        return fw.text


#
#    M E T H O D S   (do GPU stuff here if desired)
#

def spectrum(pyrat):
    """
    Spectrum calculation driver.
    """
    pyrat.log.head('\nCalculate the planetary spectrum.')

    # Initialize the spectrum array (CPU-based by default):
    pyrat.spec.spectrum = np.zeros(pyrat.spec.nwave, dtype=np.float64)
    if pyrat.opacity.is_patchy:
        pyrat.spec.clear  = np.zeros(pyrat.spec.nwave, dtype=np.float64)
        pyrat.spec.cloudy = np.zeros(pyrat.spec.nwave, dtype=np.float64)

    # Dispatch to geometry-specific calls:
    if pyrat.od.rt_path in pc.transmission_rt:
        modulation(pyrat)
    elif pyrat.od.rt_path == 'emission':
        intensity(pyrat)
        flux(pyrat)
    elif pyrat.od.rt_path == 'emission_two_stream':
        two_stream(pyrat)

    # Possibly scale the emission flux:
    if pyrat.spec.f_dilution is not None and pyrat.od.rt_path in pc.emission_rt:
        pyrat.spec.spectrum *= pyrat.spec.f_dilution

    # Output to file:
    wn = pyrat.spec.wn  # still CPU
    io.write_spectrum(1.0/wn, pyrat.spec.spectrum, pyrat.spec.specfile,
                      'transit' if pyrat.od.rt_path in pc.transmission_rt else 'emission')
    if pyrat.spec.specfile is not None:
        specfile = f": '{pyrat.spec.specfile}'"
    else:
        specfile = ""
    pyrat.log.head(f"Computed spectrum{specfile}.", indent=2)
    pyrat.log.head('Done.')


def modulation(pyrat):
    """
    Calculate transmission spectrum for transit geometry.
    If desired, we can GPU-accelerate the integrand and trapz, etc.
    below is an example of how to do partial GPU usage.
    """
    # We can choose to do everything on CPU if we like:
    rtop = pyrat.atm.rtop
    radius = pyrat.atm.radius         # CPU array
    depth  = pyrat.od.depth           # CPU array
    nlayers = pyrat.od.ideep - rtop + 1

    # Suppose we do exponent on GPU:
    # 1) Convert arrays to GPU if available:
    if GPU_AVAILABLE:
        xp = cp
        radius_gpu = xp.asarray(radius)
        depth_gpu  = xp.asarray(depth)
        h_gpu = xp.ediff1d(radius_gpu[rtop:])
        integ_gpu = xp.exp(-depth_gpu[rtop:,:]) * xp.expand_dims(radius_gpu[rtop:], 1)
        # Move back to CPU for trapz2D:
        h_cpu    = h_gpu.get()
        integ_cpu= integ_gpu.get()
    else:
        # fallback CPU
        radius_np = radius
        depth_np  = depth
        h_cpu = np.ediff1d(radius_np[rtop:])
        integ_cpu = np.exp(-depth_np[rtop:,:])*np.expand_dims(radius_np[rtop:],1)

    # Possibly handle deck model on CPU:
    for model in pyrat.opacity.models:
        if model.name == 'deck' and model.itop > rtop:
            # Adjust h:
            h_cpu[model.itop-rtop-1] = model.rsurf - radius[model.itop-1]
            # Adjust integ by interpolation:
            f = interp1d(radius[rtop:], integ_cpu, axis=0)
            new_val = f(model.rsurf)
            integ_cpu[model.itop-rtop] = new_val
            break

    # 2) CPU-based 2D trapz:
    spectrum_val = t.trapz2D(integ_cpu, h_cpu, nlayers-1)

    # 3) Convert final to CPU-based result:
    rp = radius[rtop]
    pyrat.spec.spectrum = (rp*rp + 2*spectrum_val)/(pyrat.phy.rstar**2)

    # Patchy case:
    if pyrat.opacity.is_patchy:
        depth_clear = pyrat.od.depth_clear
        nlayers_clear = pyrat.od.ideep_clear - rtop + 1
        if GPU_AVAILABLE:
            depth_clear_gpu = xp.asarray(depth_clear)
            integ_clear_gpu = xp.exp(-depth_clear_gpu[rtop:,:]) * xp.expand_dims(radius_gpu[rtop:],1)
            h_clear_gpu = h_gpu.copy()
            # back to CPU
            h_clear_cpu = h_clear_gpu.get()
            integ_clear_cpu = integ_clear_gpu.get()
        else:
            integ_clear_cpu = np.exp(-depth_clear[rtop:,:]) * \
                              np.expand_dims(radius[rtop:],1)
            h_clear_cpu = h_cpu.copy()

        clear_val = t.trapz2D(integ_clear_cpu, h_clear_cpu, nlayers_clear-1)
        clear_spectrum = (rp*rp + 2*clear_val)/(pyrat.phy.rstar**2)

        pyrat.spec.clear  = clear_spectrum
        pyrat.spec.cloudy = pyrat.spec.spectrum
        fpatchy = pyrat.opacity.fpatchy
        pyrat.spec.spectrum = (pyrat.spec.cloudy*fpatchy
                               + pyrat.spec.clear*(1-fpatchy))


def intensity(pyrat):
    """
    Calculate the intensity spectrum (erg s-1 cm-2 sr-1 cm)
    for eclipse geometry.
    """
    # If we want, we can do partial GPU. For brevity, let's stay CPU-only here:
    spec = pyrat.spec
    pyrat.log.msg('Computing intensity spectrum (CPU).', indent=2)

    spec.intensity = np.zeros((spec.nangles, spec.nwave), dtype=np.float64)

    # Build blackbody array B on CPU:
    B = np.zeros((pyrat.atm.nlayers, spec.nwave), dtype=np.float64)
    ps.blackbody_wn_2D(spec.wn, pyrat.atm.temp, B, pyrat.od.ideep)

    # If there's a deck surface:
    for model in pyrat.opacity.models:
        if model.name == 'deck':
            B[model.itop] = ps.blackbody_wn(spec.wn, model.tsurf)

    pyrat.od.B = B  # store it in case needed

    # Now do plane-parallel integration. If t.intensity is CPU-based:
    spec.intensity = t.intensity(
        pyrat.od.depth, pyrat.od.ideep, B,
        spec.quadrature_mu, pyrat.atm.rtop,
    )


def flux(pyrat):
    """
    Calculate the hemisphere-integrated flux spectrum (erg s-1 cm-2 cm)
    for eclipse geometry.
    """
    spec = pyrat.spec
    # Weighted sum of intensities:
    w = spec.quadrature_weights  # shape [nangles, 1]
    # broadcast multiply [nangles, nwave] * [nangles,1], sum over nangles
    spec.spectrum[:] = np.sum(spec.intensity * w, axis=0)


def two_stream(pyrat):
    """
    Two-stream approximation.
    For brevity, we do CPU-based logic here.
    """
    pyrat.log.msg('Compute two-stream flux spectrum (CPU).', indent=2)
    spec = pyrat.spec
    phy = pyrat.phy
    nlayers = pyrat.atm.nlayers

    # Internal flux from interior temperature:
    spec.f_int = ps.blackbody_wn(spec.wn, pyrat.atm.tint)
    total_f_int = np.trapz(spec.f_int, spec.wn)
    if total_f_int > 0:
        scale = pc.sigma * (pyrat.atm.tint**4) / total_f_int
        spec.f_int *= scale

    # dtau0 = difference in depth:
    dtau0 = np.diff(pyrat.od.depth, axis=0)
    # trans = ...
    # SciPy exp1 => CPU
    trans = (1 - dtau0)*np.exp(-dtau0) + dtau0**2 * ss.exp1(dtau0)

    # Build blackbody B array, CPU:
    B = ps.blackbody_wn_2D(spec.wn, pyrat.atm.temp)
    # Bp
    Bp = np.diff(B, axis=0)/dtau0

    # flux_down/up:
    spec.flux_down = np.zeros((nlayers, spec.nwave), dtype=np.float64)
    spec.flux_up   = np.zeros((nlayers, spec.nwave), dtype=np.float64)

    # Possibly starflux:
    is_irradiation = (
        spec.starflux is not None
        and pyrat.atm.smaxis is not None
        and phy.rstar is not None
    )
    if is_irradiation:
        spec.flux_down[0] = (
            pyrat.atm.beta_irr * (phy.rstar/pyrat.atm.smaxis)**2
        ) * spec.starflux

    # Downward flux:
    for i in range(nlayers-1):
        spec.flux_down[i+1] = (
            trans[i]*spec.flux_down[i]
            + np.pi*B[i]*(1-trans[i])
            + np.pi*Bp[i]*(
                -2/3*(1-np.exp(-dtau0[i])) + dtau0[i]*(1-trans[i]/3)
            )
        )

    # Upward flux:
    spec.flux_up[-1] = spec.flux_down[-1] + spec.f_int
    for i in reversed(range(nlayers-1)):
        spec.flux_up[i] = (
            trans[i]*spec.flux_up[i+1]
            + np.pi*B[i+1]*(1-trans[i])
            + np.pi*Bp[i]*(
                2/3*(1-np.exp(-dtau0[i])) - dtau0[i]*(1-trans[i]/3)
            )
        )

    spec.spectrum = spec.flux_up[0]

