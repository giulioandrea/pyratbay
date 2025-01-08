# merged_spectrum_od_optimized.py
# -------------------------------------------------------------------
#  Single-file example demonstrating:
#  - Spectrum class & methods (CPU-based constructor).
#  - optical_depth function (GPU-based), minimizing data copies.
# -------------------------------------------------------------------
#  (C) 2021-2023 Patricio Cubillos under GPL-2.0 license

__all__ = [
    'Spectrum',
    'spectrum',
    'modulation',
    'intensity',
    'flux',
    'two_stream',
    'optical_depth',
]

import numpy as np
import scipy.constants as sc
import scipy.special as ss
from scipy.interpolate import interp1d
from time import perf_counter

# Attempt to import CuPy, else fallback
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

import pycuda.driver as drv
import pycuda.autoinit
from   pycuda.compiler import SourceModule

# Pyrat Bay internals
from .. import constants as pc
from .. import io
from .. import spectrum as ps
from .. import tools as pt
from ..lib import _trapz as t
from ..lib import cutils as cu
from .. import atmosphere as pa

# -------------------------------------------------------------------
#   S P E C T R U M   C L A S S
# -------------------------------------------------------------------
class Spectrum():
    def __init__(self, inputs, log):
        """
        Make the wavenumber sample from user inputs (CPU).
        Keeping constructor CPU-based to avoid type issues with external code.
        """
        log.head('\nGenerating wavenumber array.')
        self.specfile  = inputs.specfile
        self.intensity = None
        self.clear     = None
        self.cloudy    = None
        self.starflux  = None
        self.wnosamp   = None
        self.wn        = None
        self.nwave     = None
        self.spectrum  = None

        # Quadrature set-up (CPU):
        if inputs.quadrature is not None:
            self.quadrature = inputs.quadrature
            qnodes, qweights = ss.p_roots(self.quadrature)
            qnodes = 0.5*(qnodes + 1.0)
            self.raygrid = np.arccos(np.sqrt(qnodes))
            self.quadrature_mu = np.sqrt(qnodes)
            weights = 0.5*np.pi*qweights
        else:
            # Custom angles on CPU
            raygrid = inputs.raygrid * sc.degree
            if raygrid[0] != 0.0:
                log.error('First angle in raygrid must be 0.0 (normal)')
            if np.any(raygrid < 0) or np.any(raygrid > 0.5*np.pi):
                log.error('raygrid angles must lie between 0 and 90 deg')
            if np.any(np.ediff1d(raygrid) <= 0):
                log.error('raygrid must be monotonically increasing')

            self.raygrid = raygrid
            self.quadrature_mu = np.cos(raygrid)
            N = len(raygrid)
            bounds = np.linspace(0, 0.5*np.pi, N+1)
            bounds[1:-1] = 0.5*(raygrid[:-1] + raygrid[1:])
            weights = np.pi*(np.sin(bounds[1:])**2 - np.sin(bounds[:-1])**2)

        self.quadrature_weights = np.expand_dims(weights, axis=1)
        self.nangles = len(self.quadrature_mu)
        self.f_dilution = inputs.f_dilution

        self._rt_path = inputs.rt_path

        if inputs.wlunits is None:
            self.wlunits = 'um'
        else:
            self.wlunits = inputs.wlunits
        wl_units = pt.u(self.wlunits)

        # Low wn boundary
        if inputs.wnlow is None and inputs.wlhigh is None:
            log.error('Undefined low wavenumber boundary (set wnlow or wlhigh).')
        if inputs.wnlow is not None and inputs.wlhigh is not None:
            log.warning(
                f'Both wnlow ({inputs.wnlow:.2e} cm-1) and wlhigh '
                f'({inputs.wlhigh:.2e} cm) were defined. wlhigh ignored.'
            )
        if inputs.wnlow is not None:
            self.wnlow  = inputs.wnlow
            self.wlhigh = 1.0/self.wnlow
        else:
            self.wlhigh = inputs.wlhigh
            self.wnlow  = 1.0/self.wlhigh

        # High wn boundary
        if inputs.wnhigh is None and inputs.wllow is None:
            log.error('Undefined high wavenumber boundary (wnhigh or wllow).')
        if inputs.wnhigh is not None and inputs.wllow is not None:
            log.warning(
                f'Both wnhigh ({inputs.wnhigh:.2e} cm-1) and wllow '
                f'({inputs.wllow:.2e} cm) were defined. wllow ignored.'
            )
        if inputs.wnhigh is not None:
            self.wnhigh = inputs.wnhigh
            self.wllow  = 1.0/self.wnhigh
        else:
            self.wllow  = inputs.wllow
            self.wnhigh = 1.0/self.wllow

        if self.wnlow > self.wnhigh:
            log.error('wnlow > wnhigh not allowed.')

        self.resolution = None
        self.wlstep     = None
        self.wnstep     = None

        # Possibly read cross-section tables:
        if pt.isfile(inputs.extfile) == 1 and inputs.runmode != 'opacity':
            wn = io.read_opacity(inputs.extfile[0], extract='arrays')[3]  # CPU array
            mask = (wn >= self.wnlow) & (wn <= self.wnhigh)
            self.wn = wn[mask]
            self.nwave = len(self.wn)
            self.spectrum = np.zeros(self.nwave, dtype=np.float64)

            if self.nwave > 2:
                dwn = np.ediff1d(np.abs(self.wn))
                dwl = np.ediff1d(np.abs(1.0/self.wn))
                res = np.abs(self.wn[1:]/dwn)
                std_dwn = np.std(dwn/np.mean(dwn))
                std_dwl = np.std(dwl/np.mean(dwl))
                std_res = np.std(res / np.mean(res))
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
                "Reading sampling from extinction table. "
                f"{sampling_text}, {self.nwave} points between "
                f"[{self.wnlow:.2f}, {self.wnhigh:.2f}] cm-1."
            )
            return

        # Or from user input:
        no_sampling = (
            inputs.wnstep is None and
            inputs.wlstep is None and
            inputs.resolution is None
        )
        if no_sampling:
            log.error('Undefined spectral sampling (set resolution, wnstep, or wlstep).')

        if inputs.wnstep is not None:
            self.wnstep = inputs.wnstep

        hcn = np.array([
            1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840,
            1260, 1680, 2160, 2520, 5040, 7560, 10080, 15120, 20160, 25200,
            27720, 45360, 50400, 55440, 83160, 110880, 221760, 277200,
        ])
        if inputs.wnosamp is None:
            if self.wnstep is None:
                self.wnstep = 1.0
            # find first hcn dividing wnstep with ratio <= 0.0004
            self.wnosamp = hcn[self.wnstep/hcn <= 0.0004][0]
        else:
            self.wnosamp = inputs.wnosamp

        self.resolution = inputs.resolution
        self.wlstep     = inputs.wlstep

        if self.resolution is not None:
            wn_array = ps.constant_resolution_spectrum(
                self.wnlow, self.wnhigh, self.resolution
            )
            self.wn = np.asarray(wn_array)
            self.wlstep = None
        elif self.wlstep is not None:
            wl = np.arange(self.wllow, self.wlhigh, self.wlstep)
            self.wn = 1.0/np.flip(wl)
            self.wnlow = float(self.wn[0])
            self.resolution = None
        else:
            nwave = int((self.wnhigh - self.wnlow)/self.wnstep) + 1
            self.wn = self.wnlow + np.arange(nwave)*self.wnstep

        self.nwave = self.wn.size
        self.spectrum = np.zeros(self.nwave, dtype=np.float64)

        if self.wnstep is None or self.wnosamp == 0:
            self.wnstep = 1.0
            self.wnosamp = 1
        self.ownstep = self.wnstep/self.wnosamp
        if not np.isfinite(self.ownstep):
            self.ownstep = 1e-6
        self.onwave = int(np.ceil((self.wn[-1]-self.wnlow)/self.ownstep)) + 1
        self.own = self.wnlow + np.arange(self.onwave)*self.ownstep
        self.odivisors = pt.divisors(self.wnosamp)

        if self.wn.size > 0 and self.wn[-1] != self.wnhigh:
            log.warning(
                f'Final wavenumber from {self.wnhigh:.4f} to {self.wn[-1]:.4f} cm-1'
            )

        log.msg(
            f'Initial wn boundary: {self.wnlow:.5e} cm-1  '
            f'({self.wlhigh/wl_units:.3e} {self.wlunits})\n'
            f'Final wn boundary:   {self.wnhigh:.5e} cm-1  '
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
            f'Wavenumber sample size: {self.nwave}\n'
            f'Fine-sample size:       {self.onwave}',
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
            'Low wavenumber boundary (wnlow):   {:10.3f} cm-1',
            self.wnlow
        )
        fw.write(
            'High wavenumber boundary (wnhigh): {:10.3f} cm-1',
            self.wnhigh
        )
        fw.write('Number of samples (nwave): {:d}', self.nwave)
        if self.resolution is None:
            fw.write('Sampling interval (wnstep): {:.3f} cm-1', self.wnstep)
        else:
            fw.write('Spectral resolving power: {:.1f}', self.resolution)
        fw.write(
            'Wavenumber array (wn):\n    {}',
            self.wn,
            fmt={'float': '{: .3f}'.format},
        )
        fw.write('Oversampling factor (wnosamp): {:d}', self.wnosamp)
        fw.write(
            '\nGaussian quadrature cos(theta) angles:\n    {}',
            self.quadrature_mu,
            prec=3,
        )
        fw.write(
            'Gaussian quadrature weights:\n    {}',
            self.quadrature_weights.flatten(),
            prec=3,
        )
        if self.intensity is not None:
            fw.write('Intensity spectra (erg s-1 cm-2 sr-1 cm):')
            for row in self.intensity:
                fw.write('    {}', row, fmt=fmt, edge=3)
        # Emission or Transit:
        if self._rt_path in pc.emission_rt:
            fw.write(
                'Emission spectrum (erg s-1 cm-2 cm):\n    {}',
                self.spectrum, fmt=fmt, edge=3,
            )
        elif self._rt_path in pc.transmission_rt:
            fw.write(
                '\nTransmission spectrum (Rp/Rs)**2:\n    {}',
                self.spectrum, fmt=fmt, edge=3,
            )
        return fw.text

# -------------------------------------------------------------------
#   M E T H O D S   (Spectrum-related)
# -------------------------------------------------------------------
def spectrum(pyrat):
    pyrat.log.head('\nCalculate planetary spectrum.')
    pyrat.spec.spectrum = np.zeros(pyrat.spec.nwave, dtype=np.float64)
    if pyrat.opacity.is_patchy:
        pyrat.spec.clear  = np.zeros(pyrat.spec.nwave, dtype=np.float64)
        pyrat.spec.cloudy = np.zeros(pyrat.spec.nwave, dtype=np.float64)

    if pyrat.od.rt_path in pc.transmission_rt:
        modulation(pyrat)
    elif pyrat.od.rt_path == 'emission':
        intensity(pyrat)
        flux(pyrat)
    elif pyrat.od.rt_path == 'emission_two_stream':
        two_stream(pyrat)

    if (pyrat.spec.f_dilution is not None) and (pyrat.od.rt_path in pc.emission_rt):
        pyrat.spec.spectrum *= pyrat.spec.f_dilution

    wn = pyrat.spec.wn
    kind = 'transit' if pyrat.od.rt_path in pc.transmission_rt else 'emission'
    io.write_spectrum(1.0/wn, pyrat.spec.spectrum, pyrat.spec.specfile, kind)
    if pyrat.spec.specfile:
        specfile = f": '{pyrat.spec.specfile}'"
    else:
        specfile = ""
    pyrat.log.head(f"Computed {kind} spectrum{specfile}.", indent=2)
    pyrat.log.head('Done.')


def modulation(pyrat):
    """
    Transmission geometry. Example partial GPU usage.
    """
    rtop = pyrat.atm.rtop
    radius = pyrat.atm.radius  # CPU
    depth  = pyrat.od.depth    # CPU
    nlayers = pyrat.od.ideep - rtop + 1

    if GPU_AVAILABLE:
        xp = cp
        radius_gpu = xp.asarray(radius)
        depth_gpu  = xp.asarray(depth)
        h_gpu      = xp.ediff1d(radius_gpu[rtop:])
        integ_gpu  = xp.exp(-depth_gpu[rtop:, :]) * xp.expand_dims(radius_gpu[rtop:], 1)
        h_cpu      = h_gpu.get()
        integ_cpu  = integ_gpu.get()
    else:
        h_cpu = np.ediff1d(radius[rtop:])
        integ_cpu = np.exp(-depth[rtop:,:])*np.expand_dims(radius[rtop:],1)

    # deck model fix
    for model in pyrat.opacity.models:
        if model.name == 'deck' and model.itop > rtop:
            h_cpu[model.itop-rtop-1] = model.rsurf - radius[model.itop-1]
            f = interp1d(radius[rtop:], integ_cpu, axis=0)
            new_val = f(model.rsurf)
            integ_cpu[model.itop-rtop] = new_val
            break

    # trapz on CPU
    spectrum_val = t.trapz2D(integ_cpu, h_cpu, nlayers-1)
    rp = radius[rtop]
    pyrat.spec.spectrum = (rp*rp + 2*spectrum_val)/(pyrat.phy.rstar**2)

    if pyrat.opacity.is_patchy:
        depth_clear = pyrat.od.depth_clear
        nlayers_clear = pyrat.od.ideep_clear - rtop + 1
        if GPU_AVAILABLE:
            xp = cp
            depth_clear_gpu = xp.asarray(depth_clear)
            integ_clear_gpu = xp.exp(-depth_clear_gpu[rtop:,:]) \
                              * xp.expand_dims(radius_gpu[rtop:],1)
            h_clear_gpu = h_gpu.copy()
            h_clear_cpu = h_clear_gpu.get()
            integ_clear_cpu = integ_clear_gpu.get()
        else:
            h_clear_cpu = h_cpu.copy()
            integ_clear_cpu = np.exp(-depth_clear[rtop:,:]) \
                              * np.expand_dims(radius[rtop:],1)

        clear_val = t.trapz2D(integ_clear_cpu, h_clear_cpu, nlayers_clear-1)
        clear_spectrum = (rp*rp + 2*clear_val)/(pyrat.phy.rstar**2)

        pyrat.spec.clear  = clear_spectrum
        pyrat.spec.cloudy = pyrat.spec.spectrum
        fpatchy = pyrat.opacity.fpatchy
        pyrat.spec.spectrum = (pyrat.spec.cloudy*fpatchy
                               + pyrat.spec.clear*(1-fpatchy))


def intensity(pyrat):
    """
    Eclipse geometry intensity (CPU example).
    """
    spec = pyrat.spec
    pyrat.log.msg('Computing intensity spectrum (CPU).', indent=2)
    spec.intensity = np.zeros((spec.nangles, spec.nwave), dtype=np.float64)

    B = np.zeros((pyrat.atm.nlayers, spec.nwave), dtype=np.float64)
    ps.blackbody_wn_2D(spec.wn, pyrat.atm.temp, B, pyrat.od.ideep)

    for model in pyrat.opacity.models:
        if model.name == 'deck':
            B[model.itop] = ps.blackbody_wn(spec.wn, model.tsurf)

    pyrat.od.B = B
    spec.intensity = t.intensity(
        pyrat.od.depth, pyrat.od.ideep, B,
        spec.quadrature_mu, pyrat.atm.rtop,
    )


def flux(pyrat):
    """
    Hemisphere-integrated flux (CPU).
    """
    spec = pyrat.spec
    w = spec.quadrature_weights
    spec.spectrum[:] = np.sum(spec.intensity * w, axis=0)


def two_stream(pyrat):
    """
    Two-stream approximation (CPU).
    """
    pyrat.log.msg('Compute two-stream flux spectrum (CPU).', indent=2)
    spec = pyrat.spec
    phy  = pyrat.phy
    nlayers = pyrat.atm.nlayers

    spec.f_int = ps.blackbody_wn(spec.wn, pyrat.atm.tint)
    total_f_int = np.trapz(spec.f_int, spec.wn)
    if total_f_int > 0:
        scale = pc.sigma*(pyrat.atm.tint**4)/total_f_int
        spec.f_int *= scale

    dtau0 = np.diff(pyrat.od.depth, axis=0)
    trans = (1 - dtau0)*np.exp(-dtau0) + dtau0**2*ss.exp1(dtau0)
    B = ps.blackbody_wn_2D(spec.wn, pyrat.atm.temp)
    Bp = np.diff(B, axis=0)/dtau0

    spec.flux_down = np.zeros((nlayers, spec.nwave), dtype=np.float64)
    spec.flux_up   = np.zeros((nlayers, spec.nwave), dtype=np.float64)

    is_irr = (
        spec.starflux is not None
        and pyrat.atm.smaxis is not None
        and phy.rstar is not None
    )
    if is_irr:
        spec.flux_down[0] = (
            pyrat.atm.beta_irr * (phy.rstar/pyrat.atm.smaxis)**2
        ) * spec.starflux

    for i in range(nlayers-1):
        spec.flux_down[i+1] = (
            trans[i]*spec.flux_down[i]
            + np.pi*B[i]*(1-trans[i])
            + np.pi*Bp[i]*(
                -2/3*(1-np.exp(-dtau0[i])) + dtau0[i]*(1-trans[i]/3)
            )
        )
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


# -------------------------------------------------------------------
#   O P T I C A L   D E P T H   (optimized to reduce copies)
# -------------------------------------------------------------------
def optical_depth(pyrat):
    """
    Calculate the optical depth in transmission geometry or emission,
    with fewer CPU<->GPU copies.
    """
    start_od = perf_counter()
    od = pyrat.od
    nwave = pyrat.spec.nwave
    nlayers = pyrat.atm.nlayers
    rtop = pyrat.atm.rtop
    f_patchy = pyrat.opacity.fpatchy

    pyrat.log.head('\nBegin optical-depth calculation.')

    # Allocate arrays (CPU or GPU).
    # We'll do the entire extinction on GPU if available:
    if GPU_AVAILABLE:
        xp = cp
        od.ec       = xp.empty((nlayers, nwave), dtype=xp.float64)
        od.depth    = xp.zeros((nlayers, nwave), dtype=xp.float64)
        if f_patchy is not None:
            od.ec_clear   = xp.empty((nlayers, nwave), dtype=xp.float64)
            od.depth_clear= xp.zeros((nlayers, nwave), dtype=xp.float64)
    else:
        xp = np
        od.ec       = np.empty((nlayers, nwave), dtype=np.float64)
        od.depth    = np.zeros((nlayers, nwave), dtype=np.float64)
        if f_patchy is not None:
            od.ec_clear    = np.empty((nlayers, nwave), dtype=np.float64)
            od.depth_clear = np.zeros((nlayers, nwave), dtype=np.float64)

    # Ray path:
    if od.rt_path in pc.emission_rt:
        # radius is CPU array
        if GPU_AVAILABLE:
            od.raypath = -cu.ediff(np.asanyarray(pyrat.atm.radius))  # returns CPU array
        else:
            od.raypath = -cu.ediff(pyrat.atm.radius)
    elif od.rt_path in pc.transmission_rt:
        od.raypath = pa.transit_path(pyrat.atm.radius, rtop)  # CPU

    # Sum contributions into od.ec:
    # Copy from CPU-based pyrat.opacity.ec to GPU if needed
    if GPU_AVAILABLE:
        od.ec[rtop:] = cp.asarray(pyrat.opacity.ec[rtop:])
        if f_patchy is not None:
            od.ec_clear[rtop:] = cp.asarray(pyrat.opacity.ec[rtop:])
            # add cloud
            od.ec[rtop:] += cp.asarray(pyrat.opacity.ec_cloud[rtop:])
    else:
        od.ec[rtop:] = pyrat.opacity.ec[rtop:]
        if f_patchy is not None:
            od.ec_clear[rtop:] = od.ec[rtop:].copy()
            od.ec[rtop:] += pyrat.opacity.ec_cloud[rtop:]

    rbottom = nlayers
    for model in pyrat.opacity.models:
        if model.name == 'deck':
            rbottom = model.itop + 1
            break

    # If emission, we can skip the PyCUDA kernel approach:
    if od.rt_path in pc.emission_rt:
        # Just set od.ideep if needed, or do CPU-based approach
        od.ideep = np.tile(nlayers-1, nwave)
        pass

    elif od.rt_path in pc.transmission_rt:
        # We'll do a single CPU->GPU flatten pass for ec, intervals, etc.
        # Then one kernel call, and one GPU->CPU pass for depth (if we want depth on CPU).
        # But we already have ec on GPU if GPU_AVAILABLE.

        od.ideep = np.full(nwave, -1, dtype=np.int32)

        # Build intervals (padded_arrays) from raypath on CPU:
        # raypath is CPU array shape: [nlayers-1 or so, ...]
        # shape must be (rbottom-rtop, something). We'll pad them if needed
        # for the kernel logic:
        if (rbottom - rtop) <= 0:
            # no layers?
            pass
        else:
            max_len = len(od.raypath[-1])  # last sub-array length
            padded_arrays_cpu = np.array([
                np.pad(row, (0, max_len - len(row)), 'constant')
                for row in od.raypath[rtop:rbottom]
            ], dtype=np.float64)

            # Flatten data & allocate tau
            if GPU_AVAILABLE:
                data_flat  = od.ec.reshape(-1)       # ec on GPU => we need it as CPU for PyCUDA, so .get()
                data_flat_cpu = data_flat.get()
                intervals_cpu = padded_arrays_cpu.ravel()
                od_depth_flat_cpu = np.zeros_like(data_flat_cpu)
            else:
                data_flat_cpu = od.ec.reshape(-1)
                intervals_cpu = padded_arrays_cpu.ravel()
                od_depth_flat_cpu = np.zeros_like(data_flat_cpu)

            mod = SourceModule(r"""
            __global__ void parallelize_tau_gloria_for(
                double *tau, 
                int *ideep, 
                double *intervals, 
                double taumax, 
                double *data, 
                int nwave,
                int nr,
                int rtop
            ){
              int jx = blockIdx.x * blockDim.x + threadIdx.x;
              if (jx < nwave){
                int i, r;
                ideep[jx] = -1;
                for(r = 0; r < nr; r++){
                  double Integral = 0.0;
                  // init tau=0
                  tau[jx + nwave*r] = 0.0;
                  if(ideep[jx]<0){
                    for(i=0; i<=r; i++){
                      Integral += intervals[i + r*(nr-1)]
                                * (data[jx+(i+1)*nwave] + data[jx + i*nwave]);
                    }
                    tau[jx + nwave*r] = Integral;
                    if(Integral > taumax){
                      ideep[jx] = r;
                    }
                  }
                }
              }
            }""")

            # Allocate on GPU with PyCUDA driver API:
            data_gpu      = drv.mem_alloc(data_flat_cpu.nbytes)
            intervals_gpu = drv.mem_alloc(intervals_cpu.nbytes)
            ideep_gpu     = drv.mem_alloc(od.ideep.nbytes)
            tau_gpu       = drv.mem_alloc(od_depth_flat_cpu.nbytes)

            # Copy CPU->GPU
            drv.memcpy_htod(data_gpu,      data_flat_cpu)
            drv.memcpy_htod(intervals_gpu, intervals_cpu)
            drv.memcpy_htod(ideep_gpu,     od.ideep)

            # Kernel launch
            N_Threads_Blocco = 256
            N_blocks = (nwave + N_Threads_Blocco -1)//N_Threads_Blocco
            kernel = mod.get_function("parallelize_tau_gloria_for")
            kernel(
                tau_gpu,
                ideep_gpu,
                intervals_gpu,
                np.double(od.maxdepth),
                data_gpu,
                np.int32(nwave),
                np.int32(nlayers),
                np.int32(rtop),
                block=(N_Threads_Blocco,1,1),
                grid=(N_blocks,1,1),
            )
            drv.Context.synchronize()

            # Copy GPU->CPU
            drv.memcpy_dtoh(od_depth_flat_cpu, tau_gpu)
            drv.memcpy_dtoh(od.ideep, ideep_gpu)

            # free GPU
            data_gpu.free()
            intervals_gpu.free()
            ideep_gpu.free()
            tau_gpu.free()

            # reshape od_depth back
            od_depth_cpu = od_depth_flat_cpu.reshape(nlayers, nwave)

            # If we want to keep `od.depth` on GPU, we do:
            if GPU_AVAILABLE:
                od.depth = cp.asarray(od_depth_cpu)
            else:
                od.depth = od_depth_cpu

            # correct negative ideep
            od.ideep[od.ideep<0] = rbottom-1

        # If patchy:
        if f_patchy is not None:
            # kept the GPU version of patchy calculations
            od.ideep_clear = np.full(nwave, -1, dtype=np.int32)
            for r in range(rtop, rbottom):
                # standard CPU approach from trapz or partial approach
                od.depth_clear[r] = t.optdepth(
                    xp.asnumpy(od.ec_clear[rtop:r+1]) if GPU_AVAILABLE else od.ec_clear[rtop:r+1],
                    od.raypath[r],
                    od.maxdepth,
                    od.ideep_clear,
                    r
                )

    end_od = perf_counter()
    od.time_od = end_od - start_od
    pyrat.log.head(f"Optical depth done. Total time: {od.time_od:.3f} s.")

