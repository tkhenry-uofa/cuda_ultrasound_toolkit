#ifndef PARAMETER_DEFS_H
#define PARAMETER_DEFS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../cuda_toolkit/cuda_beamformer_parameters.h"

#define COMMAND_PIPE_NAME "\\\\.\\pipe\\cuda_command"
#define DATA_SMEM_NAME "Local\\cuda_data"
#define PARAMETERS_SMEM_NAME "Local\\cuda_parameters"

#define MEGABYTE (1024 * 1024)
#define GIGABYTE (1024 * 1024 * 1024)

#define COMMAND_PIPE_SIZE MEGABYTE
#define DATA_SMEM_SIZE GIGABYTE

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
	CudaCommand opcode = ERR;
	long long data_size = 0;
	int frame_count = 0;
} CommandPipeMessage;

typedef struct SVDParameters
{
	int frame_count;
	int filter_indicies[256];	// Singular values to remove
} SVDParameters;

typedef struct NCCMotionParameters
{
	int frame_count;
	unsigned int  frame_dims[3];		// [X, Y, Z]
	int reference_frame;
} NCCMotionParameters;

typedef struct SharedMemoryParams
{
	CudaBeamformerParameters BeamformerParameters;
	SVDParameters SVDParameters;
	NCCMotionParameters NccMotionParameters;
} SharedMemoryParams;


#ifdef __cplusplus
}   // extern "C"  
#endif
#endif // !PARAMETER_DEFS_H

