/* See LICENSE for license details. */
#include <stddef.h>
#include <stdint.h>


/* See LICENSE for license details. */

typedef char      c8;
typedef uint8_t   u8;
typedef int16_t   i16;
typedef uint16_t  u16;
typedef int32_t   i32;
typedef uint32_t  u32;
typedef uint32_t  b32;
typedef uint64_t  u64;
typedef float     f32;
typedef double    f64;
typedef ptrdiff_t size;
typedef ptrdiff_t iptr;

typedef struct { f32 x, y; }       v2;
typedef struct { f32 x, y, z, w; } v4;
typedef struct { u32 x, y; }       uv2;
typedef struct { u32 x, y, z; }    uv3;
typedef struct { u32 x, y, z, w; } uv4;

typedef struct {
	u16 channel_mapping[256];   /* Transducer Channel to Verasonics Channel */
	u16 uforces_channels[256];  /* Channels used for virtual UFORCES elements */
	f32 focal_depths[256];      /* [m] Focal Depths for each transmit of a RCA imaging scheme*/
	f32 transmit_angles[256];   /* [radians] Transmit Angles for each transmit of a RCA imaging scheme*/
	f32 xdc_transform[16];      /* IMPORTANT: column major order */
	uv4 dec_data_dim;           /* Samples * Channels * Acquisitions; last element ignored */
	uv4 output_points;          /* Width * Height * Depth * (Frame Average Count) */
	v4  output_min_coordinate;  /* [m] Back-Top-Left corner of output region (w ignored) */
	v4  output_max_coordinate;  /* [m] Front-Bottom-Right corner of output region (w ignored)*/
	f32 xdc_element_pitch[2];   /* [m] Transducer Element Pitch {row, col} */
	uv2 rf_raw_dim;             /* Raw Data Dimensions */
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
	u32 data_type;
	u8 mixes_count;
	u8 mixes_offset;
	u8 mixes_rows[128];
	f32 match_filter[1024];
	u16 filter_length;
	f32 _pad[3];
} BeamformerParameters;

#define ARRAY_COUNT(a) (sizeof(a) / sizeof(*a))
typedef struct { size len; u8 *data; } s8;
#define s8(s) (s8){.len = ARRAY_COUNT(s) - 1, .data = (u8 *)s}

#if defined(_WIN32)
#define LIB_FN __declspec(dllexport)
#else
#define LIB_FN
#endif

LIB_FN b32 set_beamformer_parameters(char *shm_name, BeamformerParameters*);

/* NOTE: sends data and waits for (complex) beamformed data to be returned.
 * out_data: must be allocated by the caller as 2 f32s per output point. */
LIB_FN void beamform_i16(char *pipe_name, char *shm_name,
                                       i16 *data, uv2 data_dim,
                                       uv4 output_points, f32 *out_data);

LIB_FN void beamform_f32(char* pipe_name, char* shm_name,
	f32* data, uv2 data_dim,
	uv4 output_points, f32* out_data);
