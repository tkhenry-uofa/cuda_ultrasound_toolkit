#ifndef DEFS_H
#define DEFS_H

#include <windows.h>

#include <iostream>

#include <stdexcept>
#include <stdio.h>
#include <string>

#include "parameter_defs.h"


typedef void* Handle;

typedef unsigned int uint;

typedef char      c8;
typedef uint8_t   u8;
typedef int16_t   i16;
typedef uint16_t  u16;
typedef int32_t   i32;
typedef uint32_t  u32;
typedef int64_t   i64;
typedef uint64_t  u64;
typedef uint32_t  b32;
typedef float     f32;
typedef double    f64;
typedef ptrdiff_t size;
typedef ptrdiff_t iptr;

#define TOTAL_TOBE_CHANNELS 256

#ifdef _DEBUG
#include <assert.h>
#define ASSERT(x) assert(x);
#else
#define ASSERT(x)
#endif // _DEBUG

typedef union {
    struct { i32 x, y; };
    struct { i32 w, h; };
    i32 E[2];
} iv2;

typedef union {
    struct { i32 x, y, z; };
    struct { i32 w, h, d; };
    iv2 xy;
    i32 E[3];
} iv3;

typedef union {
    struct { u32 x, y; };
    struct { u32 w, h; };
    u32 E[2];
} uv2;

typedef union {
    struct { u32 x, y, z; };
    struct { u32 w, h, d; };
    uv2 xy;
    u32 E[3];
} uv3;

typedef union {
    struct { u32 x, y, z, w; };
    struct { uv3 xyz; u32 _w; };
    u32 E[4];
} uv4;

typedef union {
    struct { f32 x, y; };
    struct { f32 w, h; };
    f32 E[2];
} v2;

typedef union {
    struct { f32 x, y, z; };
    struct { f32 w, h, d; };
    f32 E[3];
} v3;

typedef union {
    struct { f32 x, y, z, w; };
    struct { f32 r, g, b, a; };
    struct { v3 xyz; f32 _1; };
    struct { f32 _2; v3 yzw; };
    struct { v2 xy, zw; };
    f32 E[4];
} v4;


enum compute_shaders {
    CS_CUDA_DECODE = 0,
    CS_CUDA_HILBERT = 1,
    CS_DAS = 2,
    CS_DEMOD = 3,
    CS_HADAMARD = 4,
    CS_MIN_MAX = 5,
    CS_SUM = 6,
    CS_LAST
};


typedef enum {
    DAS_ID_FORCES = 0,
    DAS_ID_UFORCES = 1,
    DAS_ID_HERCULES = 2,
    DAS_ID_RCA_VLS = 3,
    DAS_ID_RCA_TPW = 4
} TransmitModes;


#define DAS_ID_UFORCES  0
#define DAS_ID_HERCULES 1

#define MEGABYTE (1024ULL * 1024ULL)
#define GIGABYTE (1024ULL * 1024ULL * 1024ULL)


#define ERROR_MSG(err_code) \
    ([](DWORD code) { \
        LPSTR messageBuffer = nullptr; \
        size_t size = FormatMessageA( \
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, \
            NULL, \
            code, \
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), \
            (LPSTR)&messageBuffer, \
            0, \
            NULL \
        ); \
        if (size == 0) return std::string("Unknown error code: ") + std::to_string(code); \
        std::string message(messageBuffer, size); \
        LocalFree(messageBuffer); \
        return message; \
    })(err_code)


#define MAX_BEAMFORMED_SAVED_FRAMES 16
#define MAX_MULTI_XDC_COUNT         4
/* NOTE: This struct follows the OpenGL std140 layout. DO NOT modify unless you have
 * read and understood the rules, particulary with regards to _member alignment_ */
typedef struct {
    i16 channel_mapping[256];   /* Transducer Channel to Verasonics Channel */
    u16 uforces_channels[256];  /* Channels used for virtual UFORCES elements */
    f32 focal_depths[256];      /* [m] Focal Depths for each transmit of a RCA imaging scheme*/
    f32 transmit_angles[256];   /* [radians] Transmit Angles for each transmit of a RCA imaging scheme*/
    f32 xdc_transform[16];      /* IMPORTANT: column major order */
    u32 dec_data_dim[4];           /* Samples * Channels * Acquisitions; last element ignored */
    u32 output_points[4];          /* Width * Height * Depth * (Frame Average Count) */
    f32  output_min_coordinate[4];  /* [m] Back-Top-Left corner of output region (w ignored) */
    f32  output_max_coordinate[4];  /* [m] Front-Bottom-Right corner of output region (w ignored)*/
    f32 xdc_element_pitch[2];   /* [m] Transducer Element Pitch {row, col} */
    u32 rf_raw_dim[2];             /* Raw Data Dimensions */
    i32 transmit_mode;          /* Method/Orientation of Transmit */
    u32 decode;                 /* Decode or just reshape data */
    f32 speed_of_sound;         /* [m/s] */
    f32 sampling_frequency;     /* [Hz]  */
    f32 center_frequency;       /* [Hz]  */
    f32 time_offset;            /* pulse length correction time [s]   */
    f32 off_axis_pos;           /* [m] Position on screen normal to beamform in 2D HERCULES */
    i32 beamform_plane;         /* Plane to Beamform in 2D HERCULES */
    f32 f_number;               /* F# (set to 0 to disable) */
    u32 das_shader_id;
    u32 readi_group_id;			/* Which readi group this data is from*/
    u32 readi_group_size;		/* Size of readi transmit group */
	u32 data_type;          /* 0: i16, 1: f32 */
    u8 mixes_count;
    u8 mixes_offset;
	u8 mixes_rows[128]; 
    f32 match_filter[1024];
    u16 filter_length;
    f32 _pad[3];
} BeamformerParameters;

typedef struct {
    BeamformerParameters raw;
    enum compute_shaders compute_stages[16];
    u32                  compute_stages_count;
    b32                  upload;
    b32                  export_next_frame;
    c8                   export_pipe_name[1024];
} BeamformerParametersFull;

namespace defs
{
    // VSX DATA
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
    static const std::string focal_depth = "focal_depth";
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