typedef enum TxRxDirection
{
 TX_ROW_RX_ROW = 0,
 TX_ROW_RX_COL = 1,
 TX_COL_RX_ROW = 2,
 TX_COL_RX_COL = 3,
 INVALID = -1,
} TxRxDirection;
typedef enum BeamformPlane
{
 PLANE_XZ = 0,
 PLANE_YZ = 1,
 PLANE_XY = 2,
 PLANE_ARBITRARY = 3,
} BeamformPlane;
typedef enum SequenceId
{
 FORCES = 0,
 UFORCES = 1,
 HERCULES = 2,
 RCA_VLS = 3,
 RCA_TPW = 4,
 UHURCULES = 5,
 RACES = 6,
 EPIC_FORCES = 7,
 EPIC_UFORCES = 8,
 EPIC_UHERCULES = 9,
 FLASH = 10,
 MIXES_S = 100,
} SequenceId;
typedef enum DataType
{
    I16 = 0,
    F32 = 1,
    F32C = 2,
} DataType;
typedef enum ReadiOrdering
{
    HADAMARD = 0,
    WALSH = 1,
} ReadiOrdering;
typedef struct
{
 /*
	*	BP Head (Transducer and sequence information)
	*/
 float xdc_transform[16]; // 4x4 Orientation Matrix for the transducer, (column major order)
 float xdc_element_pitch[2]; // [m] Transducer Element Pitch {row, col}
 unsigned int rf_raw_dim[2]; // Raw Data Dimensions [samples * transmits + padding, total_channels (rows + cols)]
 unsigned int dec_data_dim[4]; // Expected dimensions after decoding [samples, rx_channels, transmits]; last element ignored
 bool decode; // Decode or just reshape data
 TxRxDirection transmit_mode; // TX and RX directions
 SequenceId das_shader_id; // Sequence type
 float time_offset; // pulse length correction time [s]
 /*
	*	BP UI (Beamforming settings)
	*/
 unsigned int output_points[4]; // [X, Y, Z, Frames]
 float output_min_coordinate[4]; // [m] Min XYZ positions, 4th value ignored
 float output_max_coordinate[4]; // [m] Max XYZ positions, 4th value ignored
 float sampling_frequency; // [Hz] Sampling frequency
 float center_frequency; // [Hz]  Frequency of the transmit, not the transducer reasonance
 float speed_of_sound; // [m/s] In the imaged volume
 float off_axis_pos; // Unused
 BeamformPlane beamform_plane; // Unused
 float f_number; // Dynamic receive apodization F# 
 bool interpolate; // Interpolate between samples during beamforming 
 bool coherency_weighting; // Apply coherency factor weighting to output data
 /*
	*	Large arrays seperate from the main BP
	*/
 short channel_mapping[256]; // Maps the ordering of the raw channel data to the physical channels
 short sparse_elements[256]; // Channels used for virtual UFORCES elements
 float focal_depths[256]; // [m] Focal Depths for each transmit
 float transmit_angles[256]; // [radians] Transmit Angles for each transmit
 /*
	*	Extra parameters (not part of the standard BP)
	*/
 unsigned int readi_group_count; // Number of READI groups in the scheme
 unsigned int readi_group_id; // Which READI group this represents
 ReadiOrdering readi_ordering; // Ordering of the READI groups
 unsigned int mixes_count; // Number of mixes crosses
 unsigned int mixes_offset; // Cross offset at the center of the array
 unsigned int mixes_rows[128]; // Cross row IDs (same for columns)
 unsigned int filter_length; // Length of the filter
 float rf_filter[1024]; // Time domain kernel of the filter (assumed to be sampled at fs)
 DataType data_type; // Type of the raw data being passed in
} CudaBeamformerParameters;
/*
* What test function we're calling from MATLAB
*/
typedef enum CudaCommand
{
 ERR = 0,
 ACK = 1,
 SUCCESS = 2,
 BEAMFORM_VOLUME = 3,
 SVD_FILTER = 4,
 NCC_MOTION_DETECT = 5,
} CudaCommand;
typedef struct CommandPipeMessage
{
 CudaCommand opcode;
 unsigned long long data_size;
 int frame_count;
} CommandPipeMessage;
typedef struct SVDParameters
{
 int frame_count;
 int filter_indicies[256]; // Singular values to remove
} SVDParameters;
typedef struct NCCMotionParameters
{
 int frame_count;
 unsigned int frame_dims[3]; // [X, Y, Z]
 int reference_frame;
} NCCMotionParameters;
typedef struct SharedMemoryParams
{
 CudaBeamformerParameters BeamformerParameters;
 SVDParameters SVDParameters;
 NCCMotionParameters NccMotionParameters;
} SharedMemoryParams;
__attribute__((dllexport)) void beamform_i16( const short* data, CudaBeamformerParameters bp, float* output);
__attribute__((dllexport)) void beamform_f32( const float* data, CudaBeamformerParameters bp, float* output);
