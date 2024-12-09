#include <Windows.h>

#include <iostream>

#include "matlab_transfer.h"


bool
matlab_transfer::disconnect_pipe(Handle pipe)
{
	BOOL result = DisconnectNamedPipe(pipe);

	return result;
}


bool
matlab_transfer::close_pipe(Handle pipe)
{
	BOOL result = CloseHandle(pipe);

	return result;
}

bool
matlab_transfer::_nack_response()
{
	return true;
}

uint
matlab_transfer::write_to_pipe(Handle pipe, void* data, uint len)
{
	DWORD bytes_written;
	BOOL result = WriteFile(pipe, data, len, &bytes_written, 0);

	if (!result)
	{
		std::cout << "Write error: " << GetLastError() << std::endl;
	}

	return (uint)bytes_written;
}



void*
matlab_transfer::_open_shared_memory_area(char* name, size cap)
{
	HANDLE h = CreateFileMappingA(INVALID_HANDLE_VALUE, 0, PAGE_READWRITE, 0, cap, name);
	if (h == NULL)
		return NULL;

	return MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, cap);
}


Handle
matlab_transfer::open_output_pipe(char* name)
{
	return CreateFileA(name, GENERIC_WRITE, 0, 0, OPEN_EXISTING, 0, 0);
}


uint 
matlab_transfer::_read_pipe(Handle pipe, void* buf, size len)
{
	DWORD total_read = 0;
	bool result = ReadFile(pipe, buf, len, &total_read, 0);
	return total_read;
}

bool
matlab_transfer::create_input_pipe(Handle* pipe)
{
	*pipe = CreateNamedPipeA(PIPE_INPUT_NAME, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE | PIPE_NOWAIT, 1, 0, 1 * MEGABYTE, 0, 0);

	return *pipe != nullptr;
}

int
matlab_transfer::last_error()
{
	return GetLastError();
}

bool 
matlab_transfer::create_smem(BeamformerParametersFull** bp_full)
{
	// Gives a pointer directly to the shared memory 
	*bp_full = (BeamformerParametersFull*)_open_shared_memory_area(SMEM_NAME, sizeof(BeamformerParametersFull));

	return *bp_full != nullptr;
}
bool 
matlab_transfer::wait_for_data(Handle pipe, void* data, uint* bytes_read, uint timeout)
{

	uint elapsed = 0;
	uint poll_period = 100; // ms

	bool pipe_ready = false;
	DWORD bytes_available = 0;

	std::cout << "Matlab Transfer: Waiting for data from matlab." << std::endl;

	while (elapsed <= timeout)
	{
		DWORD total_read = 0;
		bool result = ReadFile(pipe, data, INPUT_MAX_BUFFER, &total_read, 0);
		DWORD error = GetLastError();

		if (result && total_read > 0) 
		{
			*bytes_read = (uint)total_read;
			return true;
		}
		else if (!result) 
		{
			if (error != ERROR_NO_DATA && error != ERROR_PIPE_LISTENING)
			{
				std::cout << "Input pipe error: " << error << " Message: " << ERROR_MSG(error);
				return false;
			}
		}

		Sleep(poll_period);
		elapsed += poll_period;
	}

	std::cout << "Matlab Transfer: Timed out waiting for data from matlab." << std::endl;
	return false;
}


