#pragma once

#include <string>
#include "../defs.h"
class TransferServer
{
public:
	TransferServer( const char* command_pipe_name,
					const char* data_smem_name,
					const char* header_smem_name );
	~TransferServer();

	TransferServer( const TransferServer& ) = delete;
	TransferServer( TransferServer&& ) = delete;
	TransferServer& operator=( const TransferServer& ) = delete;
	TransferServer& operator=( TransferServer&& ) = delete;

	void* get_data_smem() const
	{
		return _data_smem;
	}

	SharedMemoryHeader* get_parameters_smem() const
	{
		return _parameters_smem;
	}

	CudaCommand get_command() const;

private:

	const char* _command_pipe_name;
	const char* _data_smem_name;
	const char* _params_smem_name;
	HANDLE _command_pipe_h;
	HANDLE _data_smem_h;
	HANDLE _params_smem_h;

	SharedMemoryHeader* _parameters_smem = nullptr;
	void* _data_smem = nullptr;

};

