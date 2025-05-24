#include <stdbool.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "cuda_transfer_new.h"

#define mexErrMsgIdAndTxt  mexErrMsgIdAndTxt_800
#define mexWarnMsgIdAndTxt mexWarnMsgIdAndTxt_800
void mexErrMsgIdAndTxt(const char*, const char*, ...);
void mexWarnMsgIdAndTxt(const char*, const char*, ...);
#define error_tag "matlab_transfer:error"
#define error_msg(...)   mexErrMsgIdAndTxt(error_tag, __VA_ARGS__)
#define warning_msg(...) mexWarnMsgIdAndTxt(error_tag, __VA_ARGS__)

static volatile SharedMemoryParams* g_params = NULL;
static volatile void* g_data = NULL;
static HANDLE g_command_pipe = INVALID_HANDLE_VALUE;

void
cleanup_shared_resources()
{
    if (g_data)
    {
        UnmapViewOfFile(g_data);
        g_data = NULL;
    }

    if (g_params)
    {
        UnmapViewOfFile(g_params);
        g_params = NULL;
    }

    if (g_command_pipe != INVALID_HANDLE_VALUE)
    {
        CloseHandle(g_command_pipe);
        g_command_pipe = INVALID_HANDLE_VALUE;
    }
}

HANDLE 
open_command_pipe()
{
    HANDLE h = CreateFileA(COMMAND_PIPE_NAME, GENERIC_READ | GENERIC_WRITE,
                                 0, NULL, OPEN_EXISTING, 0, NULL);

    if (h == INVALID_HANDLE_VALUE)
    {
        DWORD error = GetLastError();
        warning_msg("Failed to open command pipe with error: %i", error);
    }
    return h;
}

bool
open_smem(char* name, void** smem)
{
    HANDLE h = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                     0, DATA_SMEM_SIZE, name);

    if (h == NULL)
    {
        DWORD error = GetLastError();
        warning_msg("Failed to create data shared memory '%s' with error: %i", name, error);
        return false;
    }

    *smem = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, DATA_SMEM_SIZE);

    if (*smem == NULL)
    {
        DWORD error = GetLastError();
        warning_msg("Failed to map data shared memory '%s' with error: %i", name, error);
        CloseHandle(h);
        return false;
    }

    CloseHandle(h);
    return true;
}

bool setup_shared_resources()
{
    if(!g_data)
    {
        if (!open_smem(DATA_SMEM_NAME, &g_data))
        {
            return false;
        }
        
    }
    memset(g_data, 0, DATA_SMEM_SIZE);

    if(!g_params)
    {
        if (!open_smem(PARAMETERS_SMEM_NAME, (void**)&g_params))
        {
            return false;
        }
    }
    memset(g_params, 0, sizeof(SharedMemoryParams));
    
    if(g_command_pipe == INVALID_HANDLE_VALUE)
    {
        g_command_pipe = open_command_pipe();
        if (g_command_pipe == INVALID_HANDLE_VALUE)
        {
            return false;
        }
    }

    return true;
}

bool send_command(CudaCommand command, size_t data_size)
{
    CommandPipeMessage message;
    message.opcode = command;
    message.data_size = data_size;
    message.frame_count = 0;

    DWORD bytes_written = 0;
    if (!WriteFile(g_command_pipe, &message, sizeof(message), &bytes_written, NULL))
    {
        DWORD error = GetLastError();
        warning_msg("Failed to write to command pipe with error: %i", error);
        return false;
    }

    return true;
}

bool wait_for_ack()
{
    CommandPipeMessage message;
    DWORD bytes_read = 0;

    if (!ReadFile(g_command_pipe, &message, sizeof(message), &bytes_read, NULL))
    {
        DWORD error = GetLastError();
        warning_msg("Failed to read from command pipe with error: %i", error);
        return false;
    }

    if (bytes_read != sizeof(message))
    {
        warning_msg("Invalid message size read from command pipe: %zu", bytes_read);
        return false;
    }

    if (message.opcode != ACK)
    {
        warning_msg("Invalid command received from command pipe: %d", message.opcode);
        return false;
    }

    return true;
}

CommandPipeMessage 
wait_for_result()
{
    CommandPipeMessage message;
    DWORD bytes_read = 0;

    if (!ReadFile(g_command_pipe, &message, sizeof(message), &bytes_read, NULL))
    {
        DWORD error = GetLastError();
        warning_msg("Failed to read from command pipe with error: %i", error);
        message.opcode = ERR;
    }

    if (bytes_read != sizeof(message))
    {
        warning_msg("Invalid message size read from command pipe: %zu", bytes_read);
        message.opcode = ERR;
    }

    if (message.opcode != SUCCESS)
    {
        warning_msg("Invalid command received from command pipe: %d", message.opcode);
        message.opcode = ERR;
    }

    return message;
}

void 
beamform(const void* data, size_t data_size,
              const CudaBeamformerParameters* bp, float* output)
{
    if (data_size > DATA_SMEM_SIZE)
    {
        error_msg("Data size too large: %zu, Max: %zu", data_size, DATA_SMEM_SIZE);
        return;
    }

    if(!setup_shared_resources())
    {
        error_msg("Failed to setup shared resources");
        return;
    }

    // Load the beamformer parameters
    memcpy(g_params, bp, sizeof(CudaBeamformerParameters)); 

    // Copy the raw data to shared memory
    memcpy(g_data, data, data_size);

    // Send the command to the CUDA server
    if (!send_command(BEAMFORM_VOLUME, data_size))
    {
        error_msg("Failed to send command to CUDA server");
        cleanup_shared_resources();
        return;
    }

    // Wait for an ack
    if (!wait_for_ack())
    {
        error_msg("Failed to receive ack from CUDA server");
        cleanup_shared_resources();
        return;
    }

    // Wait for a result
    CommandPipeMessage result = wait_for_result();
    if (result.opcode == ERR)
    {
        error_msg("Failed to receive valid result from CUDA server");
        cleanup_shared_resources();
        return;
    }

    size_t output_size = result.data_size;
    memcpy(output, g_data, output_size);

    cleanup_shared_resources();
}


void
beamform_i16(const int16_t* data, const CudaBeamformerParameters* bp, float* output)
{
    size_t data_size = bp->rf_raw_dim[0] * bp->rf_raw_dim[1] * sizeof(int16_t);
    beamform(data, data_size, bp, output);
}

void
beamform_f32(const float* data, const CudaBeamformerParameters* bp, float* output)
{
    size_t data_size = bp->rf_raw_dim[0] * bp->rf_raw_dim[1] * sizeof(float);
    beamform(data, data_size, bp, output);
}


