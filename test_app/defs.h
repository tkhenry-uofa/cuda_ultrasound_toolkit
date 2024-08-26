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
    // VSX DATA
	static const std::string rf_data_name = "rf_scans";

    static const std::string beamformer_params_name = "bp";
    static const std::string channel_mapping_name = "channel_mapping";
    static const std::string decoded_dims_name = "dec_data_dim";
    static const std::string raw_dims_name = "rf_raw_dim";
    static const std::string channel_offset_name = "channel_offset";

    static const std::string center_freq_name = "center_frequency";
    static const std::string sample_freq_name = "sampling_frequency";
    static const std::string speed_of_sound_name = "speed_of_sound";
    static const std::string pitch_name = "pitch";
    static const std::string time_offset_name = "time_offset";
    static const std::string focus_name = "focus";
    static const std::string xdc_min_name = "xdc_min_xy";
    static const std::string xdc_max_name = "xdc_max_xy";

    namespace sims
    {
        // Field II simulation data
        static const std::string rf_data_name = "rx_scans";
        static const std::string tx_config_name = "tx_config";

        static const std::string f0_name = "f0";
        static const std::string fs_name = "fs";

        static const std::string row_count_name = "rows";
        static const std::string col_count_name = "cols";
        static const std::string width_name = "width";
        static const std::string pitch_name = "pitch";

        static const std::string x_min_name = "x_min";
        static const std::string x_max_name = "x_max";
        static const std::string y_min_name = "y_min";
        static const std::string y_max_name = "y_max";

        static const std::string tx_count_name = "no_transmits";
        static const std::string focus_name = "src";
        static const std::string pulse_delay_name = "pulse_delay";
    }

    

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

