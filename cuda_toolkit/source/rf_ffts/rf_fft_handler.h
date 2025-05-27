#pragma once

#include <cufft.h>
#include <span>

#include "../defs.h"

namespace rf_fft 
{

        class rfFftHandler
        {
        public:
            rfFftHandler() = default;
            rfFftHandler(const rfFftHandler&) = delete;
            rfFftHandler& operator=(const rfFftHandler&) = delete;
            rfFftHandler(rfFftHandler&&) = delete;
            rfFftHandler& operator=(rfFftHandler&&) = delete;
            ~rfFftHandler() {};

            bool plan_ffts(uint2 fft_dims);

            bool load_filter(std::span<const float> match_filter);
            bool hilbert_and_filter(const cuComplex* d_input, cuComplex* d_output, size_t input_offset, size_t output_offset, size_t sample_count, size_t channel_count);

        private:

            cufftHandle _forward_packed_plan;
            cufftHandle _forward_strided_plan;
            cufftHandle _inverse_plan;

            uint2 fft_dims;

            float* _d_match_filter;
            size_t _match_filter_length = 0;
        }
}
