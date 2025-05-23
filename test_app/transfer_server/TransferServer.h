#pragma once

#include "../defs.h"


#define PIPE_INPUT_NAME "\\\\.\\pipe\\beamformer_data_fifo"
#define PIPE_OUTPUT_NAME "\\\\.\\pipe\\beamformer_output_fifo"
#define SMEM_NAME "Local\\ogl_beamformer_parameters"

static class TransferServer
{
public:
	TransferServer();
	~TransferServer();

private:

	static constexpr const char* _command_pipe_name = "\\\\.\\pipe\\cuda_command";
	static constexpr const char* _data_pipe_name = "\\\\.\\pipe\\cuda_data";
	static constexpr const char* _smem_name = "Local\\cuda_parameters";
	


	Ha

};

