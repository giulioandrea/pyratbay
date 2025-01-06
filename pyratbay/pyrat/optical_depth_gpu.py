def optical_depth(pyrat):
    """
    Calculate the optical depth using GPU acceleration when available.
    """
    # Initialize all timing variables
    od = pyrat.od
    start_od = perf_counter()
    
    # Initialize GPU timing variables with default values
    od.time_copy_to_GPU = 0
    od.time_programming_GPU = 0
    od.time_copy_from_GPU = 0
    
    nwave = pyrat.spec.nwave
    nlayers = pyrat.atm.nlayers
    rtop = pyrat.atm.rtop
    f_patchy = pyrat.opacity.fpatchy

    pyrat.log.head('\nBegin optical-depth calculation.')

    # Initialize arrays
    od.ec = np.empty((nlayers, nwave))
    od.depth = np.zeros((nlayers, nwave))
    if f_patchy is not None:
        od.ec_clear = np.empty((nlayers, nwave))
        od.depth_clear = np.zeros((nlayers, nwave))

    # Calculate ray path
    if pyrat.od.rt_path in pc.emission_rt:
        pyrat.od.raypath = -cu.ediff(pyrat.atm.radius)
    elif pyrat.od.rt_path in pc.transmission_rt:
        pyrat.od.raypath = pa.transit_path(pyrat.atm.radius, pyrat.atm.rtop)

    # Sum extinction contributions
    od.ec[rtop:] = pyrat.opacity.ec[rtop:]

    if f_patchy is not None:
        od.ec_clear[rtop:] = np.copy(od.ec[rtop:])
        od.ec[rtop:] += pyrat.opacity.ec_cloud[rtop:]

    rbottom = nlayers
    for model in pyrat.opacity.models:
        if model.name == 'deck':
            rbottom = model.itop + 1
            break

    if od.rt_path in pc.transmission_rt:
        # GPU implementation for transmission
        try:
            start_copy_GPU = perf_counter()
            
            max_length = len(od.raypath[-1])
            padded_arrays = np.array([np.pad(arr, (0, max_length - len(arr)), mode='constant') 
                                    for arr in od.raypath[rtop:rbottom]])
            data_flatten = od.ec.flatten()
            od_depth_flatten = od.depth.flatten()

            # GPU memory allocation
            data_gpu = drv.mem_alloc(data_flatten.nbytes)
            intervals_gpu = drv.mem_alloc(padded_arrays.nbytes)
            ideep_gpu = drv.mem_alloc(od.ideep.nbytes)
            tau_gpu = drv.mem_alloc(od_depth_flatten.nbytes)

            # Copy to GPU
            drv.memcpy_htod(data_gpu, data_flatten)
            drv.memcpy_htod(intervals_gpu, padded_arrays)
            
            end_copy_GPU = perf_counter()
            od.time_copy_to_GPU = end_copy_GPU - start_copy_GPU

            # Kernel execution
            N_Threads_Blocco = 1024
            N_blocks = int((nwave - 1)/N_Threads_Blocco + 1)
            
            mod = SourceModule("""""")
            kernel = mod.get_function("parallelize_tau_gloria_for")
            kernel(
                tau_gpu, ideep_gpu, intervals_gpu,
                np.double(od.maxdepth),
                data_gpu,
                np.int32(nwave),
                np.int32(nlayers),
                np.int32(rtop),
                block=(N_Threads_Blocco,1,1),
                grid=(N_blocks,1,1)
            )
            
            end_GPU = perf_counter()
            od.time_programming_GPU = end_GPU - end_copy_GPU

            # Synchronize and copy back
            drv.Context.synchronize()
            drv.memcpy_dtoh(od.depth, tau_gpu)
            drv.memcpy_dtoh(od.ideep, ideep_gpu)
            
            end_copy_from_GPU = perf_counter()
            od.time_copy_from_GPU = end_copy_from_GPU - end_GPU

            # Cleanup
            data_gpu.free()
            intervals_gpu.free()
            ideep_gpu.free()
            tau_gpu.free()

        except Exception as e:
            pyrat.log.warning(f"GPU calculation failed: {str(e)}, falling back to CPU")
            # CPU fallback calculation
            od.ideep = np.array(np.tile(-1, nwave), dtype=np.intc)
            for r in range(rtop, rbottom):
                if od.ideep[i] < 0:
                    integral = 0.0
                    for j in range(r + 1):
                        integral += od.raypath[r][j] * (od.ec[j+1,i] + od.ec[j,i])
                    od.depth[r,i] = integral
                    
                    if od.depth[r,i] > od.maxdepth:
                        od.ideep[i] = rtop + r
            od.ideep[od.ideep < 0] = rbottom - 1

    elif od.rt_path in pc.emission_rt:
        # Emission path calculation
        od.ideep = np.tile(nlayers-1, nwave)
        maxdepth = np.inf if od.rt_path=='emission_two_stream' else od.maxdepth
        for i in range(nwave):
            od.ideep[i] = rtop - 1 + t.cumtrapz(
                od.depth[rtop:,i],
                od.ec[rtop:,i],
                od.raypath[rtop:rbottom],
                maxdepth)

    # Handle patchy clouds if present
    if f_patchy is not None:
        rbottom = nlayers
        od.ideep_clear = np.array(np.tile(-1, nwave), dtype=np.intc)
        for r in range(rtop, rbottom):
            od.depth_clear[r] = t.optdepth(
                od.ec_clear[rtop:r+1], od.raypath[r], od.maxdepth,
                od.ideep_clear, r)
        od.ideep_clear[od.ideep_clear < 0] = r

    od.time_od = perf_counter() - start_od
    pyrat.log.head(f'Optical depth done. TIME[s]: {od.time_od}')
