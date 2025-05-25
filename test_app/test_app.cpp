#include "test_app.h"


TestApp::TestApp()
{
    _transfer_server = NULL;
    try
	{
		_transfer_server = new TransferServer(COMMAND_PIPE_NAME, DATA_SMEM_NAME, PARAMETERS_SMEM_NAME, DATA_SMEM_SIZE);
	}
	catch(const std::runtime_error& e)
	{
		std::cerr << e.what() << '\n';
		throw e;
	}
}

TestApp::~TestApp()
{
    cleanup();
}

void 
TestApp::cleanup()
{
    if (_transfer_server)
    {
        delete _transfer_server;
		_transfer_server = nullptr;
    }
}

void 
TestApp::run()
{
    _message_loop();
}

void
TestApp::_message_loop()
{
    int commands_received = 0;
	while(true)
	{
		std::cout << "Waiting for command..." << std::endl;
		auto command_opt = _transfer_server->wait_for_command();
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

				if(!_handle_beamform_command(command))
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
}


bool
TestApp::_handle_beamform_command(const CommandPipeMessage& command)
{
    const CudaBeamformerParameters* bp = &((_transfer_server->get_parameters_smem())->BeamformerParameters);

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

	const void* data_buffer = _transfer_server->get_data_smem();
	if (!data_buffer)
	{
		std::cerr << "Error: Data buffer not set." << std::endl;
		return false;
	}

	return true;
}
