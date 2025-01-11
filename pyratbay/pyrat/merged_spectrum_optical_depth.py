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
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from time import perf_counter
from .. import constants as pc
from .. import io
from .. import spectrum as ps
from .. import tools as pt
from ..lib import _trapz as t
import pycuda.gpuarray as gpuarray



class Spectrum():
    def __init__(self, inputs, log):
        """
        Make the wavenumber sample from user inputs.
        """
        log.head('\nGenerating wavenumber array.')

        self.specfile = inputs.specfile # Transmission/Emission spectrum file
        self.intensity = None  # Intensity spectrum array
        self.clear     = None  # Clear modulation spectrum for patchy model
        self.cloudy    = None  # Cloudy modulation spectrum for patchy model
        self.starflux  = None  # Stellar flux spectrum
        self.wnosamp = None

        # Gaussian-quadrature flux integration over hemisphere (for emission)
        if inputs.quadrature is not None:
            self.quadrature = inputs.quadrature
            qnodes, qweights = ss.p_roots(self.quadrature)
            qnodes = 0.5*(qnodes + 1.0)
            self.raygrid = np.arccos(np.sqrt(qnodes))
            self.quadrature_mu = np.sqrt(qnodes)
            weights = 0.5 * np.pi * qweights
        else:
            # Custom defined mu-angles:
            raygrid = inputs.raygrid * sc.degree
            if raygrid[0] != 0:
                log.error('First angle in raygrid must be 0.0 (normal)')
            if np.any(raygrid < 0) or np.any(raygrid > 0.5*np.pi):
                log.error('raygrid angles must lie between 0 and 90 deg')
            if np.any(np.ediff1d(raygrid) <= 0):
                log.error('raygrid angles must be monotonically increasing')
            self.quadrature_mu = np.cos(raygrid)
            # Weights are the projected area for each mu range:
            bounds = np.linspace(0, 0.5*np.pi, len(raygrid)+1)
            bounds[1:-1] = 0.5 * (raygrid[:-1] + raygrid[1:])
            weights = np.pi * (np.sin(bounds[1:])**2 - np.sin(bounds[:-1])**2)

        self.quadrature_weights = np.expand_dims(weights, axis=1)
        self.nangles = len(self.quadrature_mu)
        self.f_dilution = inputs.f_dilution

        # Radiative-transfer path:
        self._rt_path = inputs.rt_path

        if inputs.wlunits is None:
            self.wlunits = 'um'
        else:
            self.wlunits = inputs.wlunits
        wl_units = pt.u(inputs.wlunits)

        # Low wavenumber boundary:
        if inputs.wnlow is None and inputs.wlhigh is None:
            log.error(
                'Undefined low wavenumber boundary.  Either set wnlow or wlhigh'
            )
        if inputs.wnlow is not None and inputs.wlhigh is not None:
            log.warning(
                f'Both wnlow ({self.wnlow:.2e} cm-1) and wlhigh '
                 '({self.wlhigh:.2e} cm) were defined.  wlhigh will be ignored'
            )

        if inputs.wnlow is not None:
            self.wnlow = inputs.wnlow
            self.wlhigh = 1.0 / self.wnlow
        else:
            self.wlhigh = inputs.wlhigh
            self.wnlow = 1.0 / self.wlhigh


        # High wavenumber boundary:
        if inputs.wnhigh is None and inputs.wllow is None:
            log.error(
                'Undefined high wavenumber boundary. Either set wnhigh or wllow'
            )
        if inputs.wnhigh is not None and inputs.wllow is not None:
            log.warning(
                f'Both wnhigh ({self.wnhigh:.2e} cm-1) and wllow '
                 '({self.wllow:.2e} cm) were defined.  wllow will be ignored'
            )

        if inputs.wnhigh is not None:
            self.wnhigh = inputs.wnhigh
            self.wllow = 1.0 / self.wnhigh
        else:
            self.wllow = inputs.wllow
            self.wnhigh = 1.0 / self.wllow

        # Consistency check (wnlow < wnhigh):
        if self.wnlow > self.wnhigh:
            log.error(
                f'Wavenumber low boundary ({self.wnlow:.1f} cm-1) must be '
                f'larger than the high boundary ({self.wnhigh:.1f} cm-1)'
            )

        self.resolution = None
        self.wlstep = None

        # If there are cross-section tables, take sampling from there:
        if pt.isfile(inputs.extfile) == 1 and inputs.runmode != 'opacity':
            wn = io.read_opacity(inputs.extfile[0], extract='arrays')[3]

            # Update wavenumber sampling:
            self.wn = wn[(wn >= self.wnlow) & (wn <= self.wnhigh)]
            self.nwave = len(self.wn)
            self.spectrum = np.zeros(self.nwave, np.double)

            if self.wnlow <= self.wn[0]:
                self.wnlow = self.wn[0]
            if self.wnhigh >= self.wn[-1]:
                self.wnhigh = self.wn[-1]

            # Guess sampling by looking at std of sampling rates:
            dwn = np.ediff1d(np.abs(self.wn))
            dwl = np.ediff1d(np.abs(1.0/self.wn))
            res = np.abs(self.wn[1:]/dwn)

            std_dwn = np.std(dwn/np.mean(dwn))
            std_dwl = np.std(dwl/np.mean(dwl))
            std_res = np.std(res/np.mean(res))
            if std_dwn < std_dwl and std_dwn < std_res:
                self.wnstep = self.wn[1] - self.wn[0]
                sampling_text = f'sampling rate = {self.wnstep:.2f} cm-1'
            elif std_dwl < std_dwn and std_dwl < std_res:
                self.wlstep = np.abs(1/self.wn[0] - 1/self.wn[1]) / pc.um
                sampling_text = f'sampling rate = {self.wlstep:.6f} um'
            else:
                g = self.wn[-2]/self.wn[-1]
                # Assuming no one would care for a R with more than 5 decimals:
                self.resolution = np.round(0.5*(1+g)/(1-g), decimals=5)
                #self.wnhigh = 2 * ex.wn[-1]/(1+g)
                sampling_text = f'R = {self.resolution:.1f}'

            log.msg(
                "Reading spectral sampling from extinction-coefficient "
                f"table.  Adopting array with {sampling_text}, "
                f"and {self.nwave} samples between "
                f"[{self.wnlow:.2f}, {self.wnhigh:.2f}] cm-1."
            )
            return


        # At least one sampling mode must be defined:
        undefined_sampling_rate = (
            inputs.wnstep is None and
            inputs.wlstep is None and
            inputs.resolution is None
        )
        if undefined_sampling_rate:
            log.error(
                'Undefined spectral sampling rate, either set resolution, '
                'wnstep, or wlstep'
            )

        # highly composite numbers
        hcn = np.array([
            1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840,
            1260, 1680, 2160, 2520, 5040, 7560, 10080, 15120, 20160, 25200,
            27720, 45360, 50400, 55440, 83160, 110880, 221760, 277200,
        ])

        if inputs.wnstep is not None:
            self.wnstep = inputs.wnstep
        # Default to a wavenumber supersampling ~0.0004 (R ~2e7 at 1.0 um)
        if inputs.wnosamp is None:
            if inputs.wnstep is None:
                self.wnstep = 1.0
            self.wnosamp = hcn[self.wnstep/hcn <= 0.0004][0]
        else:
            self.wnosamp = inputs.wnosamp

        self.resolution = inputs.resolution
        self.wlstep = inputs.wlstep

        if inputs.resolution is not None:
            # Constant-resolving power wavenumber sampling:
            self.wn = ps.constant_resolution_spectrum(
                self.wnlow, self.wnhigh, self.resolution,
            )
            self.wlstep = None
        elif self.wlstep is not None:
            # Constant-sampling rate wavelength sampling:
            wl = np.arange(self.wllow, self.wlhigh, self.wlstep)
            self.wn = 1.0/np.flip(wl)
            self.wnlow = self.wn[0]
            self.resolution = None
        else:
            # Constant-sampling rate wavenumber sampling:
            nwave = int((self.wnhigh-self.wnlow)/self.wnstep) + 1
            self.wn = self.wnlow + np.arange(nwave) * self.wnstep
        self.nwave = len(self.wn)

        # Fine-sampled wavenumber array:
        self.ownstep = self.wnstep / self.wnosamp
        self.onwave = int(np.ceil((self.wn[-1]-self.wnlow)/self.ownstep)) + 1
        self.own = self.wnlow + np.arange(self.onwave) * self.ownstep
        self.spectrum = np.zeros(self.nwave, np.double)

        # Get list of divisors:
        self.odivisors = pt.divisors(self.wnosamp)

        # Re-set final boundary (stay inside given boundaries):
        if self.wn[-1] != self.wnhigh:
            log.warning(
                f'Final wavenumber modified from {self.wnhigh:.4f} cm-1 (input)'
                f'\n                            to {self.wn[-1]:.4f} cm-1'
            )

        # Screen output:
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
            wl_step = self.wlstep / wl_units
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
            fw.write(
                'Spectral resolving power (resolution): {:.1f}',
                self.resolution,
            )
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
            for intensity in self.intensity:
                fw.write('    {}', intensity, fmt=fmt, edge=3)
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


def spectrum(pyrat):
    """
    Spectrum calculation driver.
    """
    pyrat.log.head('\nCalculate the planetary spectrum.')

    # Initialize the spectrum array:
    pyrat.spec.spectrum = np.empty(pyrat.spec.nwave, np.double)
    if pyrat.opacity.is_patchy:
        pyrat.spec.clear  = np.empty(pyrat.spec.nwave, np.double)
        pyrat.spec.cloudy = np.empty(pyrat.spec.nwave, np.double)

    # Call respective function depending on the RT/geometry:
    if pyrat.od.rt_path in pc.transmission_rt:
        modulation(pyrat)

    elif pyrat.od.rt_path == 'emission':
        intensity(pyrat)
        flux(pyrat)

    elif pyrat.od.rt_path == 'emission_two_stream':
        two_stream(pyrat)

    if pyrat.spec.f_dilution is not None and pyrat.od.rt_path in pc.emission_rt:
        pyrat.spec.spectrum *= pyrat.spec.f_dilution

    # Print spectrum to file:
    if pyrat.od.rt_path in pc.transmission_rt:
        spec_type = 'transit'
    elif pyrat.od.rt_path in pc.emission_rt:
        spec_type = 'emission'

    io.write_spectrum(
        1.0/pyrat.spec.wn, pyrat.spec.spectrum, pyrat.spec.specfile, spec_type)
    if pyrat.spec.specfile is not None:
        specfile = f": '{pyrat.spec.specfile}'"
    else:
        specfile = ""
    pyrat.log.head(f"Computed {spec_type} spectrum{specfile}.", indent=2)
    pyrat.log.head('Done.')

try:
    from . import spectrum_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

def compute_combined_gpu(pyrat):
    """
    Optimized combined computation of optical depth and spectrum using GPU.
    """
    start_time = perf_counter()
    
    # Extract needed variables
    od = pyrat.od
    nwave = pyrat.spec.nwave
    nlayers = pyrat.atm.nlayers
    rtop = pyrat.atm.rtop
    radius = pyrat.atm.radius
    rstar_squared = pyrat.phy.rstar**2
    
    # Pre-calculate arrays on CPU to minimize GPU memory operations
    h = np.ascontiguousarray(np.ediff1d(radius[rtop:]), dtype=np.float64)
    radius_arr = np.ascontiguousarray(radius, dtype=np.float64)
    
    # Use shared memory for frequently accessed data
    cuda_code = """
    #include <math.h>
    
    __global__ void compute_combined_kernel(
        double *__restrict__ tau,
        int *__restrict__ ideep,
        const double *__restrict__ intervals,
        const double taumax,
        const double *__restrict__ data,
        const double *__restrict__ radius,
        const double *__restrict__ h,
        double *__restrict__ spectrum,
        const int nwave,
        const int nr,
        const int rtop,
        const double rstar_squared,
        const int max_length
    ) {
        // Shared memory for frequently accessed data
        extern __shared__ double shared_data[];
        double *shared_radius = shared_data;
        double *shared_h = &shared_data[nr];
        
        // Load data into shared memory
        int tid = threadIdx.x;
        for (int i = tid; i < nr && i < blockDim.x; i += blockDim.x) {
            if (i < nr) {
                shared_radius[i] = radius[i];
                if (i < nr-rtop) {
                    shared_h[i] = h[i];
                }
            }
        }
        __syncthreads();
        
        int jx = blockIdx.x * blockDim.x + threadIdx.x;
        if (jx < nwave) {
            // Register cache for frequently used values
            const double rtop_radius_sq = shared_radius[rtop] * shared_radius[rtop];
            double integral = 0.0;
            int max_layer = -1;
            
            // Optical depth and spectrum calculation combined loop
            #pragma unroll 4
            for (int r = 0; r < nr - rtop; r++) {
                if (max_layer < 0) {
                    double tau_val = 0.0;
                    
                    // Vectorized reduction for integral calculation
                    #pragma unroll 4
                    for (int i = 0; i <= r; i++) {
                        tau_val += intervals[i + r * max_length] * 
                                 (data[jx + (i + rtop + 1) * nwave] + 
                                  data[jx + (i + rtop) * nwave]);
                    }
                    
                    tau[jx + nwave * r] = tau_val;
                    
                    // Early exit if tau exceeds max depth
                    if (tau_val > taumax) {
                        max_layer = r;
                        ideep[jx] = rtop + r;
                        break;
                    }
                    
                    // Spectrum calculation
                    double exp_term = exp(-tau_val);
                    double r_term = shared_radius[r + rtop];
                    integral += exp_term * r_term * shared_h[r];
                }
            }
            
            if (max_layer < 0) {
                max_layer = nr - rtop - 1;
                ideep[jx] = nr - 1;
            }
            
            // Final spectrum calculation
            spectrum[jx] = (rtop_radius_sq + 2.0 * integral) / rstar_squared * 100.0;
        }
    }
    """
    
    # Compile kernel with optimization flags
    mod = SourceModule(cuda_code, options=['-O3', '--use_fast_math'])
    
    # Prepare arrays
    data_flatten = np.ascontiguousarray(od.ec, dtype=np.float64)
    
    # Handle inhomogeneous raypath array
    max_length = max(len(path) for path in od.raypath[rtop:nlayers])
    intervals = np.zeros((nlayers-rtop, max_length), dtype=np.float64)
    for i, path in enumerate(od.raypath[rtop:nlayers]):
        intervals[i, :len(path)] = path
    intervals = np.ascontiguousarray(intervals.flatten(), dtype=np.float64)
    
    ideep = np.array(np.tile(-1, nwave), dtype=np.int32)
    spectrum = np.zeros(nwave, dtype=np.float64)
    
    try:
        # Allocate GPU memory using gpuarray for better memory management
        data_gpu = gpuarray.to_gpu(data_flatten)
        intervals_gpu = gpuarray.to_gpu(intervals)
        radius_gpu = gpuarray.to_gpu(radius_arr)
        h_gpu = gpuarray.to_gpu(h)
        ideep_gpu = drv.mem_alloc(ideep.nbytes)
        spectrum_gpu = drv.mem_alloc(spectrum.nbytes)
        tau_gpu = drv.mem_alloc(data_flatten.nbytes)
        
        # Calculate optimal thread/block configuration
        max_threads = drv.Context.get_device().get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
        threads_per_block = min(256, max_threads)
        blocks_per_grid = (nwave + threads_per_block - 1) // threads_per_block
        
        # Calculate shared memory size
        shared_mem_size = (nlayers + (nlayers-rtop)) * np.dtype(np.float64).itemsize
        
        # Launch kernel
        kernel = mod.get_function("compute_combined_kernel")
        kernel(
            tau_gpu,
            ideep_gpu,
            intervals_gpu,
            np.float64(od.maxdepth),
            data_gpu,
            radius_gpu,
            h_gpu,
            spectrum_gpu,
            np.int32(nwave),
            np.int32(nlayers),
            np.int32(rtop),
            np.float64(rstar_squared),
            np.int32(max_length),
            block=(threads_per_block, 1, 1),
            grid=(blocks_per_grid, 1, 1),
            shared=shared_mem_size
        )
        
        # Copy results back
        drv.memcpy_dtoh(od.depth, tau_gpu)
        drv.memcpy_dtoh(ideep, ideep_gpu)
        drv.memcpy_dtoh(spectrum, spectrum_gpu)
        
        # Clean up
        data_gpu.gpudata.free()
        intervals_gpu.gpudata.free()
        radius_gpu.gpudata.free()
        h_gpu.gpudata.free()
        ideep_gpu.free()
        spectrum_gpu.free()
        tau_gpu.free()
        
        return spectrum, od.depth, ideep
        
    except Exception as e:
        # Cleanup in case of error
        try:
            data_gpu.gpudata.free()
            intervals_gpu.gpudata.free()
            radius_gpu.gpudata.free()
            h_gpu.gpudata.free()
            ideep_gpu.free()
            spectrum_gpu.free()
            tau_gpu.free()
        except:
            pass
        raise e

def modulation(pyrat):
    """
    Optimized transmission spectrum calculation for transit geometry.
    Uses GPU when available, falls back to optimized CPU implementation.
    """
    try:
        # Try GPU computation first
        spectrum, depth, ideep = compute_combined_gpu(pyrat)
        pyrat.spec.spectrum = spectrum
        pyrat.od.depth = depth
        pyrat.od.ideep = ideep
        
    except Exception as e:
        # Fall back to CPU if GPU fails
        pyrat.log.warning(f"GPU computation failed, falling back to CPU: {e}")
        
        rtop = pyrat.atm.rtop
        radius = pyrat.atm.radius
        depth = pyrat.od.depth
        
        # Optimized CPU implementation
        h = np.ediff1d(radius[rtop:])
        radius_expanded = np.expand_dims(radius[rtop:], 1)
        depth_slice = depth[rtop:,:]
        
        # Vectorized exponential calculation
        exp_terms = np.exp(-depth_slice)
        integ = exp_terms * radius_expanded
        
        # Efficient integration
        nlayers = pyrat.od.ideep - rtop + 1
        spectrum = t.trapz2D(integ, h, nlayers-1)
        pyrat.spec.spectrum = (radius[rtop]**2 + 2*spectrum) / pyrat.phy.rstar**2
        
        if pyrat.opacity.is_patchy:
            depth_clear = pyrat.od.depth_clear
            h_clear = h
            
            # Vectorized calculation for patchy clouds
            exp_terms_clear = np.exp(-depth_clear[rtop:,:])
            integ_clear = exp_terms_clear * radius_expanded
            
            nlayers = pyrat.od.ideep_clear - rtop + 1
            pyrat.spec.clear = t.trapz2D(integ_clear, h_clear, nlayers-1)
            pyrat.spec.clear = (
                (radius[rtop]**2 + 2*pyrat.spec.clear) / pyrat.phy.rstar**2
            )
            pyrat.spec.cloudy = pyrat.spec.spectrum
            
            # Vectorized mixing calculation
            f_patchy = pyrat.opacity.fpatchy
            pyrat.spec.spectrum = (
                pyrat.spec.cloudy * f_patchy +
                pyrat.spec.clear * (1-f_patchy)
            )
    
    # Original CPU implementation follows...

def intensity(pyrat):
    """Calculate the intensity spectrum for eclipse geometry."""
    spec = pyrat.spec
    pyrat.log.msg('Computing intensity spectrum.', indent=2)

    # Allocate intensity array
    spec.intensity = np.empty((spec.nangles, spec.nwave), np.double)

    # Calculate the Planck Emission
    pyrat.od.B = np.zeros((pyrat.atm.nlayers, spec.nwave), np.double)
    ps.blackbody_wn_2D(spec.wn, pyrat.atm.temp, pyrat.od.B, pyrat.od.ideep)

    for model in pyrat.opacity.models:
        if model.name == 'deck':
            pyrat.od.B[model.itop] = ps.blackbody_wn(pyrat.spec.wn, model.tsurf)

    # Check if we can use CUDA
    if CUDA_AVAILABLE:
        try:
            spec.intensity = spectrum_cuda.compute_intensity_cuda(
                pyrat.od.depth,
                pyrat.od.B,
                spec.quadrature_mu,
                pyrat.atm.rtop
            )
            return
        except Exception as e:
            pyrat.log.warning(f"CUDA computation failed, falling back to CPU: {e}")


def intensity(pyrat):
    """
    Calculate the intensity spectrum (erg s-1 cm-2 sr-1 cm) for
    eclipse geometry.
    """
    spec = pyrat.spec
    pyrat.log.msg('Computing intensity spectrum.', indent=2)

    # Allocate intensity array:
    spec.intensity = np.empty((spec.nangles, spec.nwave), np.double)

    # Calculate the Planck Emission:
    pyrat.od.B = np.zeros((pyrat.atm.nlayers, spec.nwave), np.double)
    ps.blackbody_wn_2D(spec.wn, pyrat.atm.temp, pyrat.od.B, pyrat.od.ideep)

    for model in pyrat.opacity.models:
        if model.name == 'deck':
            pyrat.od.B[model.itop] = ps.blackbody_wn(pyrat.spec.wn, model.tsurf)

    # Plane-parallel radiative-transfer intensity integration:
    spec.intensity = t.intensity(
        pyrat.od.depth, pyrat.od.ideep, pyrat.od.B, spec.quadrature_mu,
        pyrat.atm.rtop,
    )

def flux(pyrat):
    """
    Calculate the hemisphere-integrated flux spectrum (erg s-1 cm-2 cm)
    for eclipse geometry.
    """
    # Weight-sum the intensities to get the flux:
    pyrat.spec.spectrum[:] = np.sum(
        pyrat.spec.intensity * pyrat.spec.quadrature_weights,
        axis=0,
    )


def two_stream(pyrat):
    """
    Two-stream approximation radiative transfer
    following Heng et al. (2014)

    This function defines downward (flux_down) and uppward fluxes
    (flux_up) into pyrat.spec, and sets the emission spectrum as the
    uppward flux at the top of the atmosphere (flux_up[0]):

    flux_up: 2D float ndarray
        Upward flux spectrum through each layer under the two-stream
        approximation (erg s-1 cm-2 cm).
    flux_down: 2D float ndarray
        Downward flux spectrum through each layer under the two-stream
        approximation (erg s-1 cm-2 cm).
    """
    pyrat.log.msg('Compute two-stream flux spectrum.', indent=2)
    spec = pyrat.spec
    phy = pyrat.phy
    nlayers = pyrat.atm.nlayers

    # Set internal net bolometric flux to sigma*Tint**4:
    spec.f_int = ps.blackbody_wn(spec.wn, pyrat.atm.tint)
    total_f_int = np.trapz(spec.f_int, spec.wn)
    if total_f_int > 0:
        spec.f_int *= pc.sigma * pyrat.atm.tint**4 / total_f_int

    # Diffusivity factor (Eq. B5 of Heng et al. 2014):
    dtau0 = np.diff(pyrat.od.depth, n=1, axis=0)
    trans = (1-dtau0)*np.exp(-dtau0) + dtau0**2 * ss.exp1(dtau0)

    B = pyrat.od.B = ps.blackbody_wn_2D(spec.wn, pyrat.atm.temp)
    Bp = np.diff(pyrat.od.B, n=1, axis=0) / dtau0

    # Diffuse approximation to compute downward and upward fluxes:
    spec.flux_down = np.zeros((nlayers, spec.nwave))
    spec.flux_up = np.zeros((nlayers, spec.nwave))

    is_irradiation = (
        spec.starflux is not None
        and pyrat.atm.smaxis is not None
        and phy.rstar is not None
    )
    # Top boundary condition:
    if is_irradiation:
        spec.flux_down[0] = \
            pyrat.atm.beta_irr * (phy.rstar/pyrat.atm.smaxis)**2 * spec.starflux
    # Eqs. (B6) of Heng et al. (2014):
    for i in range(nlayers-1):
        spec.flux_down[i+1] = (
            trans[i] * spec.flux_down[i]
            + np.pi * B[i] * (1-trans[i])
            + np.pi * Bp[i] * (
                  -2/3 * (1-np.exp(-dtau0[i])) + dtau0[i]*(1-trans[i]/3))
        )

    spec.flux_up[nlayers-1] = spec.flux_down[nlayers-1] + spec.f_int
    for i in reversed(range(nlayers-1)):
        spec.flux_up[i] = (
            trans[i] * spec.flux_up[i+1]
            + np.pi * B[i+1] * (1-trans[i])
            + np.pi * Bp[i] * (
                  2/3 * (1-np.exp(-dtau0[i])) - dtau0[i]*(1-trans[i]/3))
        )

    spec.spectrum = spec.flux_up[0]
