#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../defs.h";

namespace decoding
{
    class HadamardDecoder
    {
    public:
        HadamardDecoder();
        ~HadamardDecoder();




    private:
        float* _d_hadamard;
    };
}