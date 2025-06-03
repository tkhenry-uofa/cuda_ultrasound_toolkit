#pragma once

#include <cuda_runtime.h>

#include "beamformer_constants.cuh"
#include "kernels/beamformer_kernels.cuh"
#include "../defs.h"
#include "../cuda_beamformer_parameters.h"

class Beamformer
{
public:
    Beamformer() = default;
    Beamformer(const Beamformer&) = delete;
    Beamformer& operator=(const Beamformer&) = delete;
    Beamformer(Beamformer&&) = delete;
    Beamformer& operator=(Beamformer&&) = delete;
    ~Beamformer() 
    { 
        if (_d_beamformer_hadamard) {
            cudaFree(_d_beamformer_hadamard);
            _d_beamformer_hadamard = nullptr;
        }
    }

    bool setup_beamformer(const CudaBeamformerParameters& bp);
    bool beamform(cuComplex* d_input, cuComplex* d_output, const CudaBeamformerParameters& bp);


private:

    float* _d_beamformer_hadamard = nullptr;

    bool _params_to_constants(const CudaBeamformerParameters& bp);
    bool _per_voxel_beamform(cuComplex* d_rf_buffer, cuComplex* d_volume);

    bf_kernels::BeamformerConstants _constants;      // Current beamformer constants
};