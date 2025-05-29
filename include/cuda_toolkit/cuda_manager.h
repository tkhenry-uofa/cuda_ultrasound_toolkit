#pragma once
#include <map>
#include <cstdlib>
#include <span>
#include <memory>
#include <array>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
 
#include "data_conversion/data_converter.h"
#include "hadamard/hadamard_decoder.h"
#include "rf_ffts/hilbert_handler.h"
#include "beamformer/beamformer.h"

#include "cuda_beamformer_parameters.h"
#include "defs.h"


class CudaManager {
public:

	CudaManager();
    CudaManager(const CudaManager&) = delete;
    CudaManager& operator=(const CudaManager&) = delete;
    CudaManager(CudaManager&&) = delete;
    CudaManager& operator=(CudaManager&&) = delete;
	~CudaManager() { deinit(); }

    bool init(uint2 rf_raw_dim, uint3 dec_data_dim, bool beamformer);
    bool deinit();

    bool set_channel_mapping(std::span<const int16_t> channel_mapping);
    bool set_match_filter(std::span<const float> match_filter);

	bool i16_convert_decode_strided(i16* d_input, cuComplex* d_output);
    
    bool hilbert_transform_strided(float* d_input, cuComplex* d_output);

    
    bool beamform(void* d_input, cuComplex* d_volume, 
                  const CudaBeamformerParameters& bp);

private:
    bool _i16_convert_decode(i16* d_input, float* d_output);

    bool _setup_decode_buffers();

    bool _cleanup_decode_buffers();

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
    std::unique_ptr<beamform::Beamformer> _beamformer;

    uint2 _rf_raw_dim;
    uint3 _dec_data_dim;

    struct {
        float* d_converted;
        float* d_decoded;
        size_t size;
    } _decode_buffers;

    cuComplex* _beamformer_rf_buffer;
};