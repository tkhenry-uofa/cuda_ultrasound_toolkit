#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../defs.h"


namespace decoding
{
    class HadamardDecoder
    {
    public:
        HadamardDecoder() : _d_hadamard(nullptr), _hadamard_size(0), _readi_ordering(ReadiOrdering::HADAMARD) {};
        ~HadamardDecoder() 
        { 
            _cleanup_hadamard(); 
            if (_cublas_handle) {
                cublasDestroy(_cublas_handle);
            }
        }

        HadamardDecoder(const HadamardDecoder&) = delete;
        HadamardDecoder& operator=(const HadamardDecoder&) = delete;
        HadamardDecoder(HadamardDecoder&&) = delete;
        HadamardDecoder& operator=(HadamardDecoder&&) = delete;

        bool decode(float* d_input, float* d_output, uint3 decoded_dims);

		bool test_convert_and_decode(void* d_input, cuComplex* d_output, InputDataTypes input_type, uint2 input_dims, uint3 output_dims, short* d_channel_mapping);

        static bool generate_hadamard(float* d_hadamard, uint row_count, ReadiOrdering readi_ordering = ReadiOrdering::HADAMARD, u8* d_test = nullptr);

        bool set_hadamard(uint row_count, ReadiOrdering readi_ordering = ReadiOrdering::HADAMARD);

        const float* get_hadamard() const { return _d_hadamard; }

    private:

        void _cleanup_hadamard() 
		{
            CUDA_NULL_FREE(_d_hadamard);
			CUDA_NULL_FREE(_compact_hadamard_test);
			_readi_ordering = ReadiOrdering::HADAMARD;
			_hadamard_size = 0;
        }

        bool _create_cublas_handle() {
            if (!_cublas_handle) {
                cublasStatus_t status = cublasCreate(&_cublas_handle);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "Failed to create cublas handle" << std::endl;
                    return false;
                }
            }
            return true;
        }

        static void _sort_walsh(float* hadamard, uint row_count);


        ReadiOrdering _readi_ordering;
        float* _d_hadamard;
        uint _hadamard_size = 0;

		u8* _compact_hadamard_test = nullptr;

        cublasHandle_t _cublas_handle = nullptr;
    };
}