#ifndef CUDA_TRANSFER_HH
#define CUDA_TRANSFER_HH

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include <parameter_defs.h>

#if defined(_WIN32)
    #define LIB_FN __declspec(dllexport)
#else
    #define LIB_FN
#endif


LIB_FN void beamform_i16( const int16_t* data, CudaBeamformerParameters bp, float* output);

LIB_FN void beamform_f32( const float* data, CudaBeamformerParameters bp, float* output);


#ifdef __cplusplus
}   // extern "C"  
#endif
#endif // !CUDA_TRANSFER_HH