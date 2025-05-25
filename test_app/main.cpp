#include <iostream>
#include <string>
#include <chrono>

#include <cuda_toolkit_testing.h>

#include "defs.h"
#include "parser/mat_parser.h"

#include "transfer_server/matlab_transfer.h"
#include "transfer_server/TransferServer.h"


PipelineParams convert_params(BeamformerParametersFull* full_bp)
{
	PipelineParams params;
	BeamformerParameters bp = full_bp->raw;

	params.focus[0] = 0.0f;
	params.focus[1] = 0.0f;
	params.focus[2] = bp.focal_depths[0];

	params.pulse_delay = bp.time_offset;

	params.decoded_dims[0] = bp.dec_data_dim[0];
	params.decoded_dims[1] = bp.dec_data_dim[1];
	params.decoded_dims[2] = bp.dec_data_dim[2];

	params.raw_dims[0] = bp.rf_raw_dim[0];
	params.raw_dims[1] = bp.rf_raw_dim[1];

	params.vol_mins[0] = bp.output_min_coordinate[0];
	params.vol_mins[1] = bp.output_min_coordinate[1];
	params.vol_mins[2] = bp.output_min_coordinate[2];

	params.vol_maxes[0] = bp.output_max_coordinate[0];
	params.vol_maxes[1] = bp.output_max_coordinate[1];
	params.vol_maxes[2] = bp.output_max_coordinate[2];

	params.vol_counts[0] = bp.output_points[0];
	params.vol_counts[1] = bp.output_points[1];
	params.vol_counts[2] = bp.output_points[2];

	params.vol_resolutions[0] = (params.vol_maxes[0] - params.vol_mins[0]) / params.vol_counts[0];
	params.vol_resolutions[1] = (params.vol_maxes[1] - params.vol_mins[1]) / params.vol_counts[1];
	params.vol_resolutions[2] = (params.vol_maxes[2] - params.vol_mins[2]) / params.vol_counts[2];

	for (int i = 0; i < 256; i++)
	{
		params.channel_mapping[i] = bp.channel_mapping[i];
	}

	params.array_params.c = bp.speed_of_sound;
	params.array_params.center_freq = bp.center_frequency;
	params.array_params.sample_freq = bp.sampling_frequency;

	params.array_params.row_count = bp.dec_data_dim[2]; // Assuming square arrays for now
	params.array_params.col_count = bp.dec_data_dim[2];

	params.array_params.xdc_mins[0] = -bp.xdc_transform[12];
	params.array_params.xdc_mins[1] = -bp.xdc_transform[13];

	params.array_params.xdc_maxes[0] = bp.xdc_transform[12];
	params.array_params.xdc_maxes[1] = bp.xdc_transform[13];

	params.array_params.pitch[0] = bp.xdc_element_pitch[0];
	params.array_params.pitch[1] = bp.xdc_element_pitch[1];

	params.readi_group_id = bp.readi_group_id;
	params.readi_group_size = bp.readi_group_size;

	params.rf_data_type = (RfDataType)bp.data_type;
	params.f_number = bp.f_number;

	params.sequence = bp.das_shader_id;

	params.mixes_count = bp.mixes_count;
	params.mixes_offset = bp.mixes_offset;

	memcpy(params.mixes_rows, bp.mixes_rows, sizeof(bp.mixes_rows));

	params.filter_length = (uint)bp.filter_length;
	memcpy(params.match_filter, bp.match_filter, sizeof(bp.match_filter));

	return params;
}

bool readi_beamform()
{
	BeamformerParametersFull* full_bp = nullptr;
	Handle input_pipe = nullptr;
	Handle output_pipe = nullptr;

	std::cout << "Main: Creating smem and input pipe." << std::endl;
	bool result = matlab_transfer::create_smem(&full_bp);

	if (!result)
	{
		std::cout << "Main: Failed to create smem." << std::endl;
		return false;
	}

	void* data_buffer = (void*)malloc(INPUT_MAX_BUFFER);
	uint bytes_read = 0;
	uint timeout = 2 * 60 * 60 * 1000; // 2 hours (for long simulations)

	result = matlab_transfer::create_input_pipe(&input_pipe);

	int max_beamforms = 1000;
	// No state is carried over between iterations so this can handle multiple runs
	// All beamforming settings come from the state of the shared memory
	for (int g = 0; g < max_beamforms; g++)
	{
		std::cout << "Starting volume " << g + 1 << std::endl;
		if (!result)
		{
			std::cout << "Main: Failed to create input pipe." << std::endl;
			return false;
		}

		result = matlab_transfer::wait_for_data(input_pipe, data_buffer, &bytes_read, timeout);

		if (!result)
		{
			std::cout << "Error reading data from matlab." << std::endl;
			return false;
		}

		std::cout << "Restarting pipe" << std::endl;

		matlab_transfer::disconnect_pipe(input_pipe);
		matlab_transfer::close_pipe(input_pipe);
		input_pipe = nullptr;
		result = matlab_transfer::create_input_pipe(&input_pipe);

		std::cout << "Created input pipe, last error: " << matlab_transfer::last_error() << std::endl;

		if (!result)
		{
			std::cout << "Main: Failed to restart input pipe." << std::endl;
			return false;
		}

		// Now that we know matlab is up we can connect to the output pipe
		output_pipe = matlab_transfer::open_output_pipe(PIPE_OUTPUT_NAME);
		if (output_pipe == nullptr)
		{
			std::cout << "Error opening export pipe to matlab." << std::endl;
			return false;
		}

		// TODO: Unify structs and types so I don't have to deal with this 
		PipelineParams params = convert_params(full_bp);

		cuComplex* volume = nullptr;
		size_t output_size = (size_t)full_bp->raw.output_points[0] * full_bp->raw.output_points[1] * full_bp->raw.output_points[2] * sizeof(cuComplex);

		std::cout << "Starting pipeline " << g + 1 << std::endl;


		if (params.rf_data_type == RfDataType::INT_16) 
			readi_beamform_raw((i16*)data_buffer, params, &volume);
		else if (params.rf_data_type == RfDataType::FLOAT_32) 
			readi_beamform_fii((f32*)data_buffer, params, &volume);
		else
		{
			std::cout << "Invalid data type." << std::endl;
			return false;
		}

		matlab_transfer::write_to_pipe(output_pipe, volume, output_size);

		matlab_transfer::close_pipe(output_pipe);

		free(volume);

		std::cout << "Volume " << g + 1 << " done." << std::endl << std::endl;
	}

	free(data_buffer);

	return true;
}


bool handle_beamform_command(TransferServer* server, const CommandPipeMessage& command)
{
	const CudaBeamformerParameters* bp = &((server->get_parameters_smem())->BeamformerParameters);

	if (!bp)
	{
		std::cerr << "Error: Beamformer parameters not set." << std::endl;
		return false;
	}

	size_t data_size = command.data_size;
	if (data_size > DATA_SMEM_SIZE)
	{
		std::cerr << "Error: Data size too large." << std::endl;
		return false;
	}
	if (data_size == 0)
	{
		std::cerr << "Error: Data size is zero." << std::endl;
		return false;
	}

	const void* data_buffer = server->get_data_smem();
	if (!data_buffer)
	{
		std::cerr << "Error: Data buffer not set." << std::endl;
		return false;
	}

	return true;
}

void message_loop()
{

	TransferServer* server = nullptr;
	try
	{
		server = new TransferServer(COMMAND_PIPE_NAME, DATA_SMEM_NAME, PARAMETERS_SMEM_NAME);
	}
	catch(const std::runtime_error& e)
	{
		std::cerr << e.what() << '\n';
		throw e;
	}

	std::cout << "Server created." << std::endl;
	int commands_received = 0;
	while(true)
	{
		std::cout << "Waiting for command..." << std::endl;
		auto command_opt = server->wait_for_command();
		if(!command_opt.has_value())
		{
			std::cerr << "Error waiting for command." << std::endl;
			break;
		}
		commands_received++;
		std::cout << "Command received: " << commands_received << std::endl;

		CommandPipeMessage command = command_opt.value();

		switch(command.opcode)
		{
			case CudaCommand::BEAMFORM_VOLUME:
				std::cout << "Beamforming volume." << std::endl;

				if(!handle_beamform_command(server, command))
					throw std::runtime_error("Error handling beamform command");

				break;

			case CudaCommand::SVD_FILTER:
				std::cout << "SVD filter command." << std::endl;
				break;

			case CudaCommand::NCC_MOTION_DETECT:
				std::cout << "NCC motion detect command." << std::endl;
				break;

			case CudaCommand::ACK:
				std::cout << "Acknowledged command." << std::endl;
				break;

			case CudaCommand::ERR:
				std::cerr << "Error command received." << std::endl;
			break;

			default:
				std::cerr << "Unknown command opcode (" << command.opcode << ") received." << std::endl;
		}


		std::cout << "Command completed." << std::endl;
	}

	delete server;
}

int main()
{
	// bool result = false;
	// result = readi_beamform();

	message_loop();

	return 0;
}