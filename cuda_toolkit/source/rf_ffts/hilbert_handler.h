#pragma once

#include <cufft.h>
#include <span>

#include "../defs.h"

namespace rf_fft 
{

    class HilbertHandler
    {
    public:
        HilbertHandler() :  _forward_packed_plan(0), 
                            _forward_strided_plan(0), 
                            _inverse_plan(0), 
                            _fft_dims(0, 0), 
                            _d_filter(nullptr) {}

        HilbertHandler(const HilbertHandler&) = delete;
        HilbertHandler& operator=(const HilbertHandler&) = delete;
        HilbertHandler(HilbertHandler&&) = delete;
        HilbertHandler& operator=(HilbertHandler&&) = delete;
        ~HilbertHandler() { _cleanup_plans(); };

        // cufft only supports signals with int32 sized lengths.
        bool plan_ffts(int2 fft_dims);

        bool load_filter(std::span<const float> match_filter);

        // Standard real to complex hilbert transform
        bool packed_hilbert_and_filter(float* d_input, cuComplex* d_output);

        // Real to complex hilbert transform with real input already in complex format
        bool strided_hilbert_and_filter(cuComplex* d_input, cuComplex* d_output);
        
    private:

        void _cleanup_plans();
        void _cleanup_filter();

        bool _filter_and_scale(cuComplex* d_data);

        cufftHandle _forward_packed_plan;
        cufftHandle _forward_strided_plan;
        cufftHandle _inverse_plan;

        int2 _fft_dims;

        cuComplex* _d_filter;
    };
}
