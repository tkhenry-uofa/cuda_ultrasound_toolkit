#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../cuda_beamformer_parameters.h"
#include "../defs.h"


namespace decoding
{
    class HadamardDecoder
    {
    public:
        HadamardDecoder() : _d_hadamard(nullptr), _hadamard_size(0), _readi_ordering(ReadiOrdering::HADAMARD) 
        {
            cublasStatus_t status = cublasCreate(&_cublas_handle);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                std::cerr << "Failed to create cublas handle" << std::endl;
            }
        };
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
        bool generate_hadamard(uint size, ReadiOrdering readi_ordering = ReadiOrdering::HADAMARD);

    private:

        bool _cleanup_hadamard() {
            if (_d_hadamard)
            {
                cudaFree(_d_hadamard);
                _d_hadamard = nullptr;
                _hadamard_size = 0;
                return true;
            }
            return false;
        }

        ReadiOrdering _readi_ordering;
        float* _d_hadamard;
        uint _hadamard_size = 0;

        cublasHandle_t _cublas_handle = nullptr;
    };
}