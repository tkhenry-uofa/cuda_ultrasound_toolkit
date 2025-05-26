#pragma once
#include <map>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>


#include "cuda_beamformer_parameters.h"
#include "defs.h"


class CudaSession {
public:
	CudaSession();
	~CudaSession();


    bool update_parameters(const CudaBeamformerParameters& new_params);



    struct CudaBuffers 
    {
        void* d_raw = nullptr;
        float* d_converted = nullptr;
        float* d_decoded = nullptr;
        cuComplex* d_complex = nullptr;
        cuComplex* d_volume = nullptr;
    };

private:
	CudaBeamformerParameters* _beamformer_params = nullptr;

    CudaBuffers _cuda_buffers;
    std::map<uint, cudaGraphicsResource_t> _ogl_buffers;

    cublasHandle_t _cublas_handle = nullptr;
    cufftHandle _forward_plan;
    cufftHandle _inverse_plan;

    float* _d_hadamard = nullptr;


};