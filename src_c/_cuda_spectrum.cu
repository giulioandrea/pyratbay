#include <cuda_runtime.h>

// Helper functions for GPU calculations
__device__ double cuda_exp(double x) {
    return exp(x);
}

// Kernel for optical depth calculation
__global__ void compute_optical_depth_kernel(
    double* tau,
    int* ideep,
    const double* intervals,
    const double taumax,
    const double* data,
    const int nwave,
    const int nr,
    const int rtop
) {
    int jx = blockIdx.x * blockDim.x + threadIdx.x;
    if (jx < nwave) {
        int i, r;
        ideep[jx] = -1;
        
        for (r = 0; r < nr; r++) {
            tau[jx + nwave * r] = 0.0;
            if (ideep[jx] < 0) {
                double integral = 0.0;
                
                for (i = 0; i <= r; i++) {
                    integral += intervals[i + r * (nr-1)] * 
                              (data[jx + (i + 1) * nwave] + data[jx + i * nwave]);
                }
                
                tau[jx + nwave * r] = integral;
                
                if (tau[jx + r * nwave] > taumax) {
                    ideep[jx] = r;
                }
            }
        }
    }
}

// Combined kernel for spectrum calculation using optical depth
__global__ void compute_spectrum_kernel(
    const double* depth,
    const double* radius,
    const double* raypath,
    double* spectrum,
    const int nwave,
    const int nlayers,
    const int rtop,
    const double rstar_squared
) {
    int wave_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (wave_idx < nwave) {
        double integral = 0.0;
        
        // Calculate spectrum using optical depth
        for (int i = rtop; i < nlayers - 1; i++) {
            double exp_term = cuda_exp(-depth[i * nwave + wave_idx]);
            double path_term = raypath[i - rtop];
            integral += exp_term * radius[i] * path_term;
        }
        
        spectrum[wave_idx] = (radius[rtop] * radius[rtop] + 2.0 * integral) / rstar_squared;
    }
}

// Host wrapper function for combined calculation
extern "C" void launch_combined_calculation(
    // Optical depth parameters
    double* tau_h,
    int* ideep_h,
    const double* intervals_h,
    const double taumax,
    const double* data_h,
    // Spectrum parameters
    const double* radius_h,
    const double* raypath_h,
    double* spectrum_h,
    // Common parameters
    const int nwave,
    const int nlayers,
    const int rtop
) {
    // Allocate device memory
    double *tau_d, *intervals_d, *data_d, *radius_d, *raypath_d, *spectrum_d;
    int *ideep_d;
    
    cudaMalloc(&tau_d, nlayers * nwave * sizeof(double));
    cudaMalloc(&ideep_d, nwave * sizeof(int));
    cudaMalloc(&intervals_d, nlayers * (nlayers-1) * sizeof(double));
    cudaMalloc(&data_d, nlayers * nwave * sizeof(double));
    cudaMalloc(&radius_d, nlayers * sizeof(double));
    cudaMalloc(&raypath_d, (nlayers-rtop) * sizeof(double));
    cudaMalloc(&spectrum_d, nwave * sizeof(double));
    
    // Copy input data to device
    cudaMemcpy(intervals_d, intervals_h, nlayers * (nlayers-1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, data_h, nlayers * nwave * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(radius_d, radius_h, nlayers * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(raypath_d, raypath_h, (nlayers-rtop) * sizeof(double), cudaMemcpyHostToDevice);
    
    // Configure kernel parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (nwave + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch optical depth kernel
    compute_optical_depth_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        tau_d,
        ideep_d,
        intervals_d,
        taumax,
        data_d,
        nwave,
        nlayers,
        rtop
    );
    
    // Launch spectrum kernel
    compute_spectrum_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        tau_d,
        radius_d,
        raypath_d,
        spectrum_d,
        nwave,
        nlayers,
        rtop,
        radius_h[rtop] * radius_h[rtop]  // rstar_squared
    );
    
    // Copy results back to host
    cudaMemcpy(tau_h, tau_d, nlayers * nwave * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ideep_h, ideep_d, nwave * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(spectrum_h, spectrum_d, nwave * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(tau_d);
    cudaFree(ideep_d);
    cudaFree(intervals_d);
    cudaFree(data_d);
    cudaFree(radius_d);
    cudaFree(raypath_d);
    cudaFree(spectrum_d);
}