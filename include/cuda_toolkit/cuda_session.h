#pragma once
#include <map>
#include <cstdlib>
#include <span>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
 
#include "data_conversion/data_converter.h"
#include "hadamard/hadamard_decoder.h"
#include "rf_ffts/hilbert_handler.h"

#include "cuda_beamformer_parameters.h"
#include "defs.h"


class RfProcessor {
public:

    
    
	RfProcessor();
    RfProcessor(const RfProcessor&) = delete;
    RfProcessor& operator=(const RfProcessor&) = delete;
    RfProcessor(RfProcessor&&) = delete;
    RfProcessor& operator=(RfProcessor&&) = delete;
	~RfProcessor() { deinit(); }


    bool init(uint2 rf_daw_dim, uint3 dec_data_dim);
    bool deinit();

    bool register_ogl_buffers(std::span<const uint> rf_data_ssbos, uint raw_data_ssbo);
    bool unregister_ogl_buffers();
    
    bool set_channel_mapping(std::span<const int16_t> channel_mapping);
    bool set_match_filter(std::span<const float> match_filter);

    bool ogl_convert_decode(size_t input_buffer_offset, uint output_buffer_idx);
    bool ogl_hilbert(uint intput_buffer_idx, uint output_buffer_idx);

    bool beamform(std::span<u8> input_data, const CudaBeamformerParameters* params, std::span<u8> volume);
    
private:

    struct CudaBuffers 
    {
        size_t decoded_data_size = 0;
        float* d_converted = nullptr;
        float* d_decoded = nullptr;
        cuComplex* d_hilbert = nullptr;
        cuComplex* d_volume = nullptr;
    };

    bool _map_ogl_buffer(void** d_ptr, cudaGraphicsResource_t ogl_resource);
    bool _unmap_ogl_buffer(cudaGraphicsResource_t ogl_resource)
    {
        CUDA_RETURN_IF_ERROR(cudaGraphicsUnmapResources(1, &ogl_resource));
        return true;
    }

    bool _setup_decode_buffers();
    bool _cleanup_device_buffers();

    bool _dims_changed(uint2 rf_raw_dim, uint3 dec_data_dim) const
    {
        return _rf_raw_dim.x != rf_raw_dim.x || 
               _rf_raw_dim.y != rf_raw_dim.y ||
               _dec_data_dim.x != dec_data_dim.x ||
               _dec_data_dim.y != dec_data_dim.y ||
               _dec_data_dim.z != dec_data_dim.z;

    };

    size_t _decoded_data_count() { return (size_t)(_dec_data_dim.x) * _dec_data_dim.y * _dec_data_dim.z;}

    bool _init; 

    std::unique_ptr<data_conversion::DataConverter> _data_converter;
    std::unique_ptr<rf_fft::HilbertHandler> _hilbert_handler;
    std::unique_ptr<decoding::HadamardDecoder> _hadamard_decoder;

    uint2 _rf_raw_dim;
    uint3 _dec_data_dim;

    std::pair<uint, cudaGraphicsResource_t> _ogl_raw_buffer; // Original input buffer, could be any type

    std::vector<std::pair<uint, cudaGraphicsResource_t>> _ogl_rf_buffers;  // Pipeline buffers, all are float2 (complex)
	
    CudaBuffers _device_buffers;

    CudaBeamformerParameters _beamformer_params;
};