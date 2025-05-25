#pragma once

#include <optional>
#include <string>
#include <cassert>

#include "../defs.h"
class TransferServer
{
public:
	TransferServer( const char* command_pipe_name,
					const char* data_smem_name,
					const char* header_smem_name,
					uint data_smem_size);
	~TransferServer();

	TransferServer( const TransferServer& ) = delete;
	TransferServer( TransferServer&& ) = delete;
	TransferServer& operator=( const TransferServer& ) = delete;
	TransferServer& operator=( TransferServer&& ) = delete;

	const void* get_data_smem() const
	{
		return _data_smem;
	}

	const SharedMemoryParams* get_parameters_smem() const
	{
		return _parameters_smem;
	}

	std::optional<CommandPipeMessage> wait_for_command();
	bool write_output( const void* data, size_t size );

	// Stubs
	bool respond_ack() {assert( false ); return false;};
	bool respond_error() {assert( false ); return false;};


private:

	bool _create_command_pipe();
	bool _cleanup_command_pipe();
	inline bool _restart_command_pipe() { return _cleanup_command_pipe() && _create_command_pipe(); }

	bool _create_data_smem();
	bool _create_params_smem();
	bool _cleanup_smem();

	const char* _command_pipe_name;
	const char* _data_smem_name;
	const char* _params_smem_name;
	HANDLE _command_pipe_h;
	HANDLE _data_smem_h;
	HANDLE _params_smem_h;

	uint _data_smem_size;

	SharedMemoryParams* _parameters_smem = nullptr;
	char* _data_smem = nullptr;

};

