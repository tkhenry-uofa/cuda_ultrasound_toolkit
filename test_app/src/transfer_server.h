#pragma once

#include <optional>
#include <span>
#include <string>
#include <cassert>
#include <cstddef>

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

	std::span<u8 const> get_data_smem() const
	{
		return _data_smem;
	}

	const SharedMemoryParams* get_parameters_smem() const
	{
		return _parameters_smem;
	}

	std::optional<CommandPipeMessage> wait_for_command();
	

	// Stubs
	bool respond_ack();
	bool respond_success(u32 output_size);
	bool respond_error();

	bool write_output_data(std::span<const u8> output_data);


private:

	bool _send_command_response( CommandPipeMessage message );
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

	const SharedMemoryParams* _parameters_smem = nullptr;

	uint _data_smem_size;
	u8* _data_smem_raw = nullptr;
	std::span<u8> _data_smem;

};