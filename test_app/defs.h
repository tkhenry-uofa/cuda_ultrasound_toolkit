#ifndef DEFS_H
#define DEFS_H

#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <cufft.h>

typedef unsigned int uint;
typedef int16_t i16;

#define TOTAL_TOBE_CHANNELS 256

#ifdef _DEBUG
    #include <assert.h>
    #define ASSERT(x) assert(x);
#else
    #define ASSERT(x)
#endif // _DEBUG

namespace defs
{
	static const std::string rf_data_name = "rx_scans";

    static const std::string beamformer_params_name = "bp";
    static const std::string channel_mapping_name = "channel_mapping";
    static const std::string decoded_dims_name = "dec_data_dim";
    static const std::string raw_dims_name = "rf_raw_dim";
    static const std::string channel_offset_name = "channel_offset";

    static const std::string center_freq_name = "center_frequency";
    static const std::string sample_freq_name = "sample_frequency";
    static const std::string speed_of_sound_name = "speed_of_sound";
    static const std::string pitch_name = "pitch";
    static const std::string time_offset_name = "time_offset";
    static const std::string focus_name = "focus";

	struct ComplexF {
		float re = 0.0f;
		float im = 0.0f;
	};

    // Annotating Dim order
    typedef union
    {
        struct
        {
            uint sample_count;
            uint channel_count;
            uint tx_count;
        };
        struct uint3;

    } RfDataDims;
}


#endif // !DEFS_H

