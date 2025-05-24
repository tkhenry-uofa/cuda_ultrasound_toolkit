#ifndef PARAMETER_DEFS_H
#define PARAMETER_DEFS_H

#include "../cuda_toolkit/cuda_beamformer_parameters.h"

#define COMMAND_PIPE_NAME "\\\\.\\pipe\\cuda_command"
#define DATA_SMEM_NAME "Local\\cuda_data"
#define HEADER_SMEM_NAME "Local\\cuda_parameters"

#define DATA_SMEM_SIZE GIGABYTE

/*
* What test function we're calling from MATLAB
*/
typedef enum
{
	ERR = -1,
	ACK = 0,
	BEAMFORM_VOLUME = 1,
	SVD_FILTER = 2,
	NCC_MOTION_DETECT = 3,
} CudaCommand;

struct CommandPipeMessage
{
	CudaCommand opcode = ERR;
	long long data_size = 0;
	int frame_count = 0;
};

struct SVDParameters
{
	int frame_count;
	int filter_indicies[256];	// Singular values to remove
};

struct NCCMotionParameters
{
	int frame_count;
	unsigned int  frame_dims[3];		// [X, Y, Z]
	int reference_frame;
};

struct SharedMemoryParams
{
	CudaBeamformerParameters BeamformerParameters;
	SVDParameters SVDParameters;
	NCCMotionParameters NccMotionParameters;
};

#endif // !PARAMETER_DEFS_H

