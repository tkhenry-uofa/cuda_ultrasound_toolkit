#ifndef DEFS_H
#define DEFS_H

#include <stdexcept>
#include <stdio.h>
#include <string>
#include <cufft.h>

#define PIPE_INPUT_NAME "\\\\.\\pipe\\beamformer_data_fifo"
#define PIPE_OUTPUT_NAME "\\\\.\\pipe\\beamformer_output_fifo"
#define SMEM_NAME "Local\\ogl_beamformer_parameters"


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

typedef struct {
    iptr  file;
    char* name;
} Pipe;

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

#define DAS_ID_UFORCES  0
#define DAS_ID_HERCULES 1

#define MEGABYTE (1024ULL * 1024ULL)
#define GIGABYTE (1024ULL * 1024ULL * 1024ULL)


#define MAX_BEAMFORMED_SAVED_FRAMES 16
#define MAX_MULTI_XDC_COUNT         4
/* NOTE: This struct follows the OpenGL std140 layout. DO NOT modify unless you have
 * read and understood the rules, particulary with regards to _member alignment_ */
typedef struct {
    u16 channel_mapping[512];   /* Transducer Channel to Verasonics Channel */
    u32 uforces_channels[128];  /* Channels used for virtual UFORCES elements */
    f32 xdc_origin[4 * MAX_MULTI_XDC_COUNT];  /* [m] Corner of transducer being treated as origin */
    f32 xdc_corner1[4 * MAX_MULTI_XDC_COUNT]; /* [m] Corner of transducer along first axis */
    f32 xdc_corner2[4 * MAX_MULTI_XDC_COUNT]; /* [m] Corner of transducer along second axis */
    uv4 dec_data_dim;           /* Samples * Channels * Acquisitions; last element ignored */
    uv4 output_points;          /* Width * Height * Depth * (Frame Average Count) */
    v4  output_min_coordinate;  /* [m] Back-Top-Left corner of output region (w ignored) */
    v4  output_max_coordinate;  /* [m] Front-Bottom-Right corner of output region (w ignored)*/
    uv2 rf_raw_dim;             /* Raw Data Dimensions */
    u32 xdc_count;              /* Number of Transducer Arrays (4 max) */
    u32 channel_offset;         /* Offset into channel_mapping: 0 or 128 (rows or columns) */
    f32 speed_of_sound;         /* [m/s] */
    f32 sampling_frequency;     /* [Hz]  */
    f32 center_frequency;       /* [Hz]  */
    f32 focal_depth;            /* [m]   */
    f32 time_offset;            /* pulse length correction time [s]   */
    f32 off_axis_pos;           /* [m] Position on screen normal to beamform in 2D HERCULES */
    i32 beamform_plane;         /* Plane to Beamform in 2D HERCULES */
    f32 f_number;               /* F# (set to 0 to disable) */
    u32 das_shader_id;
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