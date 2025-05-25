/* See LICENSE for license details. */
#include "cuda_transfer.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef struct {
	BeamformerParameters raw;
	b32                  upload;
	b32                  export_next_frame;
	c8                   export_pipe_name[1024];
} BeamformerParametersFull;


#define INVALID_FILE (-1)

#define ARRAY_COUNT(a) (sizeof(a) / sizeof(*a))

#define MS_TO_S (1000ULL)
#define NS_TO_S (1000ULL * 1000ULL)

#define POLL_PERIOD 100  // 100s
#define POLL_TIMEOUT 1200 * 1000 // 20 minutes

#define OS_EXPORT_PIPE_NAME "\\\\.\\pipe\\beamformer_output_fifo"


#define mexErrMsgIdAndTxt  mexErrMsgIdAndTxt_800
#define mexWarnMsgIdAndTxt mexWarnMsgIdAndTxt_800
void mexErrMsgIdAndTxt(const c8*, c8*, ...);
void mexWarnMsgIdAndTxt(const c8*, c8*, ...);
#define error_tag "matlab_transfer:error"
#define error_msg(...)   mexErrMsgIdAndTxt(error_tag, __VA_ARGS__)
#define warning_msg(...) mexWarnMsgIdAndTxt(error_tag, __VA_ARGS__)

static volatile BeamformerParametersFull* g_bp;
static HANDLE g_data_pipe = INVALID_HANDLE_VALUE;


static BeamformerParametersFull *
open_shared_memory_area(char *name)
{
	HANDLE h = OpenFileMappingA(FILE_MAP_ALL_ACCESS, 0, name);
	if (h == INVALID_HANDLE_VALUE)
	{
		warning_msg("Failed to open smem area with error %i", GetLastError());
		return 0;
	}
		

	BeamformerParametersFull *smem;
	HANDLE view = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(*smem));

	if (!view)
	{
		warning_msg("Failed to map smem '%s' from handle '%p' with error %i, attmepted size '%d'", name, h, GetLastError(), sizeof(*smem));
	}

	smem = (BeamformerParametersFull *)view;
	CloseHandle(h);

	return smem;
}

static b32
check_shared_memory(char *name)
{
	if (!g_bp) {
		g_bp = open_shared_memory_area(name);
		if (!g_bp) {
			error_msg("failed to open shared memory area with error");
			return 0;
		}
	}
	return 1;
}

HANDLE 
create_volume_pipe()
{
	HANDLE volume_pipe = CreateNamedPipeA(OS_EXPORT_PIPE_NAME, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE | PIPE_NOWAIT, 1,
		0, 1024UL * 1024UL, 0, 0);

	if (volume_pipe == INVALID_HANDLE_VALUE) {
		return INVALID_HANDLE_VALUE;
	}

	return volume_pipe;
}


void
cleanup_pipe_server(HANDLE* pipe)
{
	if (!DisconnectNamedPipe(*pipe))
	{
		warning_msg("Failed to disconnect from pipe server with error: %i", GetLastError());
	}
	if (!CloseHandle(*pipe))
	{
		warning_msg("Failed to close pipe server with error: %i", GetLastError());
	}
	*pipe = INVALID_HANDLE_VALUE;
}

static int
poll_pipe_server(HANDLE* p)
{
	// Try and read 0 bytes, this will give more pipe status information than PeakNamedPipe
	u8 data = 0;
	DWORD total_read = 0;
	b32 result = ReadFile(*p, &data, 0, &total_read, NULL);

	i32 error = GetLastError();
	if (result)
	{
		//warning_msg("Poll success, Error: %i", error);
		return 1;
	}

	if (error == ERROR_BROKEN_PIPE)
	{
		// Only time the client will disconnect is if it crashes, cleanup 
		// and error out.
		cleanup_pipe_server(p);
		error_msg("Beamformer has exited without returning a volume\n");
	}
	// These three errors just mean nothing's been sent yet, otherwise the pipe is in a bad state
	// and needs to be recreated.
	else if (error != ERROR_NO_DATA && error != ERROR_PIPE_LISTENING && error != ERROR_PIPE_NOT_CONNECTED)
	{
		warning_msg("Poll failed, Windows error '%i'.\n", error);
		cleanup_pipe_server(p);
		*p = create_volume_pipe();

		if (*p == INVALID_HANDLE_VALUE)
		{
			error = GetLastError();
			error_msg("Failed to reopen volume pipe after error, "
				"Windows error '%i'.\n", error);
		}

	}
	return 0;
}


b32
set_beamformer_parameters(char *shm_name, BeamformerParameters *new_bp)
{
	if (!check_shared_memory(shm_name))
		return 0;

	u8 *src = (u8 *)new_bp, *dest = (u8 *)&g_bp->raw;
	for (size i = 0; i < sizeof(BeamformerParameters); i++)
		dest[i] = src[i];
	g_bp->upload = 1;

	return 1;
}

void
beamform(char* pipe_name, char* shm_name, void* data, size_t data_size,
	uv4 output_points, f32* out_data)
{
	if (!check_shared_memory(shm_name))
		return;

	if (output_points.x == 0) output_points.x = 1;
	if (output_points.y == 0) output_points.y = 1;
	if (output_points.z == 0) output_points.z = 1;
	output_points.w = 1;

	s8 export_name = s8(OS_EXPORT_PIPE_NAME);

	HANDLE volume_pipe = create_volume_pipe();

	if (volume_pipe == INVALID_HANDLE_VALUE) {

		error_msg("failed to open volume pipe with error, '%d'", GetLastError());
		return;
	}
	else
	{
		//warning_msg("Opened export pipe '%s', file '%p'", OS_EXPORT_PIPE_NAME, volume_pipe);
	}

	g_data_pipe = CreateFileA(pipe_name, GENERIC_WRITE, 0, 0, OPEN_EXISTING, 0, 0);

	if (g_data_pipe == INVALID_HANDLE_VALUE)
	{
		error_msg("failed to open data pipe");
		cleanup_pipe_server(&volume_pipe);
		return;
	}

	DWORD bytes_written = 0;
	u32 elapsed = 0;
	while (elapsed <= POLL_TIMEOUT)
	{
		WriteFile(g_data_pipe, data, data_size, &bytes_written, 0);
		if (bytes_written != data_size)
		{
			if (GetLastError() != ERROR_PIPE_NOT_CONNECTED)
			{
				cleanup_pipe_server(&volume_pipe);
				error_msg("Failed to write full data to pipe: Total: %ld, Wrote: %i",
					data_size, bytes_written);
				return;
			}
			else
			{
				// Client just not connected, wait
			}
		}
		else
		{
			// Success
			break;
		}

		warning_msg("Waiting\n");
		Sleep(POLL_PERIOD);
		elapsed += POLL_PERIOD;
	}

	b32 result = CloseHandle(g_data_pipe);
	if (!result)
	{
		warning_msg("Failed to close pipe '%s' with error: %i", pipe_name, GetLastError());
	}

	b32 pipe_ready = 0;
	b32 success = 0;

	size output_size = output_points.x * output_points.y * output_points.z * sizeof(f32) * 2;
	while (elapsed <= POLL_TIMEOUT)
	{

		if (poll_pipe_server(&volume_pipe))
		{

			DWORD total_read = 0;
			success = ReadFile(volume_pipe, out_data, output_size, &total_read, 0);

			if (!success)
			{
				i32 error_code = GetLastError();
				// Use warning_msg, error_msg exits MEX early preventing cleanup
				warning_msg("Read pipe error, Data size: %i, Total read: %i, Error code: %i, Handle: %p\n", output_size, total_read, error_code, volume_pipe);
			}


			break;
		}
		else
		{
			Sleep(POLL_PERIOD);
			elapsed += POLL_PERIOD;
		}
	}

	cleanup_pipe_server(&volume_pipe);


	if (!success)
	{
		error_msg("failed to read full export data from pipe\n");
	}
}

void
beamform_i16(char *pipe_name, char *shm_name, i16 *data, uv2 data_dim,
                           uv4 output_points, f32 *out_data)
{
	size data_size = data_dim.x * data_dim.y * sizeof(i16);
	beamform(pipe_name, shm_name, (void*)data, data_size, output_points, out_data);
}

void
beamform_f32(char* pipe_name, char* shm_name, f32* data, uv2 data_dim,
	uv4 output_points, f32* out_data)
{
	size data_size = data_dim.x * data_dim.y * sizeof(f32);
	beamform(pipe_name, shm_name, (void*)data, data_size, output_points, out_data);
}

