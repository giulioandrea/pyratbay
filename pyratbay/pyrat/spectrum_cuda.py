# spectrum_cuda.py
import ctypes
import numpy as np
import os

# Load the CUDA library
def load_cuda_lib():
    # Determine the library file extension based on the OS
    if os.name == 'posix':
        lib_ext = '.so'
    elif os.name == 'nt':
        lib_ext = '.dll'
    else:
        raise OSError("Unsupported operating system")

    # Construct library path relative to this file
    lib_path = os.path.join(
        os.path.dirname(__file__),
        'lib',
        f'spectrum_cuda{lib_ext}'
    )
    
    try:
        cuda_lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise OSError(f"Failed to load CUDA library: {e}")

    # Define argument types for modulation function
    cuda_lib.launch_modulation_kernel.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64),  # radius
        np.ctypeslib.ndpointer(dtype=np.float64),  # depth
        np.ctypeslib.ndpointer(dtype=np.float64),  # h
        np.ctypeslib.ndpointer(dtype=np.float64),  # spectrum
        ctypes.c_int,  # nwave
        ctypes.c_int,  # nlayers
        ctypes.c_int,  # rtop
        ctypes.c_double,  # rstar_squared
    ]

    # Define argument types for intensity function
    cuda_lib.launch_intensity_kernel.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64),  # depth
        np.ctypeslib.ndpointer(dtype=np.float64),  # B
        np.ctypeslib.ndpointer(dtype=np.float64),  # quadrature_mu
        np.ctypeslib.ndpointer(dtype=np.float64),  # intensity
        ctypes.c_int,  # nwave
        ctypes.c_int,  # nlayers
        ctypes.c_int,  # nangles
        ctypes.c_int,  # rtop
    ]

    return cuda_lib

# Global library instance
_cuda_lib = None

def get_cuda_lib():
    global _cuda_lib
    if _cuda_lib is None:
        _cuda_lib = load_cuda_lib()
    return _cuda_lib

def compute_modulation_cuda(radius, depth, h, rtop, rstar_squared):
    """
    Compute transmission spectrum modulation using CUDA.
    
    Parameters
    ----------
    radius : ndarray
        Radius array
    depth : ndarray
        Optical depth array
    h : ndarray
        Layer thickness array
    rtop : int
        Top atmospheric layer index
    rstar_squared : float
        Square of stellar radius
        
    Returns
    -------
    ndarray
        Computed spectrum
    """
    lib = get_cuda_lib()
    
    # Ensure arrays are contiguous and in correct format
    radius = np.ascontiguousarray(radius, dtype=np.float64)
    depth = np.ascontiguousarray(depth, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    
    nwave = depth.shape[1]
    nlayers = depth.shape[0]
    
    # Prepare output array
    spectrum = np.zeros(nwave, dtype=np.float64)
    
    # Call CUDA function
    lib.launch_modulation_kernel(
        radius,
        depth,
        h,
        spectrum,
        nwave,
        nlayers,
        rtop,
        rstar_squared
    )
    
    return spectrum

def compute_intensity_cuda(depth, B, quadrature_mu, rtop):
    """
    Compute intensity spectrum using CUDA.
    
    Parameters
    ----------
    depth : ndarray
        Optical depth array
    B : ndarray
        Planck function array
    quadrature_mu : ndarray
        Gaussian quadrature angles
    rtop : int
        Top atmospheric layer index
        
    Returns
    -------
    ndarray
        Computed intensity spectrum
    """
    lib = get_cuda_lib()
    
    # Ensure arrays are contiguous and in correct format
    depth = np.ascontiguousarray(depth, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    quadrature_mu = np.ascontiguousarray(quadrature_mu, dtype=np.float64)
    
    nwave = depth.shape[1]
    nlayers = depth.shape[0]
    nangles = len(quadrature_mu)
    
    # Prepare output array
    intensity = np.zeros((nangles, nwave), dtype=np.float64)
    
    # Call CUDA function
    lib.launch_intensity_kernel(
        depth,
        B,
        quadrature_mu,
        intensity,
        nwave,
        nlayers,
        nangles,
        rtop
    )
    
    return intensity
