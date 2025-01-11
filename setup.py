# Copyright (c) 2021-2024 Patricio Cubillos
# Pyrat Bay is open-source software under the GPL-2.0 license (see LICENSE)

import os
import re
import subprocess
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
from numpy import get_include

# Source directories
srcdir = 'src_c/'  # C-code source folder
incdir = 'src_c/include/'  # Include folder with header files

# Detect CUDA installation
def find_cuda():
    if os.name == 'nt':  # Windows
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path is None:
            # Try standard installation path
            if os.path.exists('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'):
                for ver in ['v12.0', 'v11.0', 'v10.0']:
                    if os.path.exists(f'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{ver}'):
                        cuda_path = f'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\{ver}'
                        break
    else:  # Linux/Mac
        cuda_path = '/usr/local/cuda'
        if not os.path.exists(cuda_path):
            cuda_path = '/usr/cuda'
            if not os.path.exists(cuda_path):
                try:
                    nvcc_path = subprocess.check_output(['which', 'nvcc']).decode().strip()
                    cuda_path = os.path.dirname(os.path.dirname(nvcc_path))
                except:
                    return None
    
    if cuda_path is None:
        return None
    
    return {
        'home': cuda_path,
        'nvcc': os.path.join(cuda_path, 'bin', 'nvcc'),
        'include': os.path.join(cuda_path, 'include'),
        'lib64': os.path.join(cuda_path, 'lib64'),
    }

# Custom build command that handles both C and CUDA extensions
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # First try to find CUDA
        cuda = find_cuda()
        
        # Build regular C extensions
        build_ext.build_extensions(self)
        
        # If CUDA is available, build CUDA extension
        if cuda is not None:
            try:
                # Create lib directory if it doesn't exist
                lib_dir = os.path.join(self.build_lib, 'pyratbay', 'lib')
                os.makedirs(lib_dir, exist_ok=True)
                
                # Compile CUDA code
                self.compiler.spawn([
                    cuda['nvcc'],
                    '-O3',
                    '--shared',
                    '--compiler-options', "'-fPIC'",
                    '-o', os.path.join(lib_dir, 'spectrum_cuda.so'),
                    os.path.join(srcdir, 'spectrum_cuda.cu'),
                ])
            except Exception as e:
                print(f"CUDA compilation failed: {e}")
                print("Continuing without CUDA support")

# Get list of C source files
cfiles = os.listdir(srcdir)
cfiles = list(filter(lambda x: re.search('.+[.]c$', x), cfiles))
cfiles = list(filter(lambda x: not re.search('[.#].+[.]c$', x), cfiles))

# Setup include directories and compilation flags
inc = [get_include(), incdir]
eca = ['-ffast-math']
ela = []

# Create Extension instances for C files
extensions = [
    Extension(
        'pyratbay.lib.' + cfile.rstrip('.c'),
        sources=[f'{srcdir}{cfile}'],
        include_dirs=inc,
        extra_compile_args=eca,
        extra_link_args=ela,
    )
    for cfile in cfiles
]

# Main setup configuration
setup(
    name='pyratbay',
    ext_modules=extensions,
    include_dirs=inc,
    cmdclass={
        'build_ext': CustomBuildExt,
    },
)
