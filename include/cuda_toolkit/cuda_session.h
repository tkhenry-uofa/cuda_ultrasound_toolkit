#pragma once
#include <map>
#include <cstdlib>
#include <span>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
 

#include "cuda_beamformer_parameters.h"
#include "defs.h"


class CudaSession {
public:
	CudaSession() =  default;
    CudaSession(const CudaSession&) = delete;
    CudaSession& operator=(const CudaSession&) = delete;
    CudaSession(CudaSession&&) = delete;
    CudaSession& operator=(CudaSession&&) = delete;
	~CudaSession() { deinit(); }


    bool init(float2 rf_daw_dim, float3 dec_data_dim, InputDataType data_type);
    bool deinit();

    bool register_ogl_buffers(const uint* rf_data_ssbos, uint rf_buffer_count, uint raw_data_ssbo);
    
    bool set_channel_mapping(const i16* channel_mapping);
    bool set_match_filter(const float* match_filter, uint length);


    bool beamform(std::span<u8> input_data, const CudaBeamformerParameters* params, std::span<u8> volume);
    struct CudaBuffers 
    {
        InputDataType data_type = InputDataType::INVALID_TYPE; 
        size_t raw_data_size = 0;
        size_t decoded_data_size = 0;
        size_t output_data_size = 0;
        void* d_raw = nullptr;
        float* d_converted = nullptr;
        cuComplex* d_decoded = nullptr;
        cuComplex* d_hilbert = nullptr;
        cuComplex* d_volume = nullptr;
    };

private:

    bool _unregister_ogl_buffers();
    bool _setup_device_buffers();
    bool _cleanup_device_buffers();

    bool _init; 

    float2 _rf_raw_dim;
    float3 _dec_data_dim;
	
    CudaBuffers _device_buffers;
    std::map<uint, cudaGraphicsResource_t> _ogl_raw_buffers;
    std::map<uint, cudaGraphicsResource_t> _ogl_rf_buffers;
    cufftHandle _forward_plan;
    cufftHandle _inverse_plan;

    float* _d_hadamard = nullptr;


    CudaBeamformerParameters _beamformer_params;


};