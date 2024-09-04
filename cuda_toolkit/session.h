#ifndef SESSION_H
#define SESSION_H

#include "defs.h"
#include "sparse_matrix.h"

struct CudaSession
{
    bool init = false;
    bool rx_cols = false;
    uint2 input_dims;
    uint3 decoded_dims;

    cublasHandle_t cublas_handle = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;
    cufftHandle forward_plan;
    cufftHandle inverse_plan;
    cufftHandle strided_plan;

    int16_t* d_input = nullptr;
    float* d_converted = nullptr;
    float* d_decoded = nullptr;
    cuComplex* d_spectrum = nullptr;
    cufftComplex* d_complex = nullptr;
    float* d_hadamard = nullptr;

    SparseMatrix* hilbert_sparse = nullptr;
    cusparseDnMatDescr_t d_spectrum_dense;
    cusparseDnMatDescr_t d_decoded_dense;


    BufferMapping raw_data_ssbo;
    BufferMapping* rf_data_ssbos = nullptr;
    uint rf_buffer_count;

    VolumeConfiguration volume_configuration;

    uint* channel_mapping = nullptr;

    float pulse_delay;
    float element_pitch;
};

extern CudaSession Session;

#endif // !SESSION_H
