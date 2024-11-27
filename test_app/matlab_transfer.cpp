#include <Windows.h>

#include <iostream>

#include "matlab_transfer.h"



bool
matlab_transfer::_nack_response()
{
	return true;
}

uint
matlab_transfer::_write_to_pipe(char* name, void* data, uint len)
{
	HANDLE pipe = CreateFileA(name, GENERIC_WRITE, 0, 0, OPEN_EXISTING, 0, 0);

	DWORD bytes_written;
	WriteFile(pipe, data, len, &bytes_written, 0);
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
matlab_transfer::_open_named_pipe(char* name)
{
	return (HANDLE)CreateNamedPipeA(name, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE, 1,
		0, 1 * MEGABYTE, 0, 0);
}

int 
matlab_transfer::_poll_pipe(Handle p)
{
	DWORD bytes_available = 0;
	return (int)(PeekNamedPipe(p, 0, 1 * MEGABYTE, 0, &bytes_available, 0) && bytes_available);
}

uint 
matlab_transfer::_read_pipe(Handle pipe, void* buf, size len)
{
	DWORD total_read = 0;
	bool result = ReadFile(pipe, buf, len, &total_read, 0);
	return total_read;
}

bool 
matlab_transfer::create_resources(BeamformerParametersFull** bp_full, Handle* input_pipe)
{
	// Gives a pointer directly to the shared memory 
	*bp_full = (BeamformerParametersFull*)_open_shared_memory_area(SMEM_NAME, sizeof(BeamformerParametersFull));
	
	// Open the pipe so that it can be written to
	*input_pipe = CreateNamedPipeA(PIPE_INPUT_NAME, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE, 1, 0, 1 * MEGABYTE, 0, 0);

	// TODO: Check individually and use GetLastError()

	if (*input_pipe == nullptr || *bp_full == nullptr)
	{
		std::cout << "Matlab Transfer: Failed to create pipe and shared memory." << std::endl;

		return false;
	}
	return true;
}
bool 
matlab_transfer::wait_for_data(Handle pipe, void** data, uint* bytes, uint timeout)
{

	uint elapsed = 0;
	uint poll_period = 100; // ms

	bool pipe_ready = false;
	DWORD bytes_available = 0;

	std::cout << "Matlab Transfer: Waiting for data from matlab." << std::endl;

	while (elapsed <= timeout)
	{
		pipe_ready = PeekNamedPipe(pipe, NULL, NULL, 0, &bytes_available, 0);

		if (pipe_ready && bytes_available > 0)
		{
			*data = malloc(bytes_available);
			*bytes = _read_pipe(pipe, *data, bytes_available);
			return true;
		}
		else
		{
			Sleep(poll_period);
			elapsed += poll_period;
		}
	}

	std::cout << "Matlab Transfer: Timed out waiting for data from matlab." << std::endl;
	return false;
}


