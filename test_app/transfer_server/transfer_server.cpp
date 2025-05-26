#include "transfer_server.h"

#define COMMAND_WAIT_TIMEOUT 60 * 60 * 1000 // 1 hour
#define COMMAND_POLL_PERIOD 100 // 100ms

#define MAX_DATA_SIZE UINT32_MAX

TransferServer::TransferServer( const char* command_pipe_name,
                                const char* data_smem_name,
                                const char* header_smem_name,
                                uint data_smem_size )
    : _command_pipe_name( command_pipe_name )
    , _data_smem_name( data_smem_name )
    , _params_smem_name( header_smem_name )
    , _data_smem_size( data_smem_size )
    , _command_pipe_h( INVALID_HANDLE_VALUE )
    , _data_smem_h( INVALID_HANDLE_VALUE )
    , _params_smem_h( INVALID_HANDLE_VALUE )
    , _parameters_smem( nullptr )
	, _data_smem_raw(nullptr)
{

    if( _data_smem_size > MAX_DATA_SIZE )
    {
        throw std::runtime_error( "Data size too large" );
    }

    if( !_create_command_pipe() )
    {
        throw std::runtime_error( "Failed to setup command pipe" );
    }

    if(!_create_data_smem())
    {
        throw std::runtime_error( "Failed to setup data shared memory" );
    }

    if(!_create_params_smem())
    {
        throw std::runtime_error( "Failed to setup header shared memory" );
    }

    // Test writing to the shared memory
    memset( _data_smem_raw, 1, _data_smem_size );
    _data_smem[0x40000000u] = 0xFFu; // Test writing to a large offset
}

TransferServer::~TransferServer()
{
    _cleanup_smem();
    _cleanup_command_pipe();
}


bool TransferServer::_create_command_pipe()
{
    _command_pipe_h = CreateNamedPipeA( _command_pipe_name, PIPE_ACCESS_DUPLEX, PIPE_TYPE_MESSAGE | PIPE_NOWAIT,
                                        1, 0, MEGABYTE, 0, 0 );

    if( _command_pipe_h == INVALID_HANDLE_VALUE )
    {
        WINDOWS_ERROR_MESSAGE( "Error creating command pipe", GetLastError() );
        return false;
    }
    return true;
}

bool TransferServer::_cleanup_command_pipe()
{
    if( IS_HANDLE_INVALID( _command_pipe_h ) )
    {
        return true;
    }

    bool result = DisconnectNamedPipe( _command_pipe_h );
    if( !result )
    {
        DWORD error = GetLastError();
        WINDOWS_ERROR_MESSAGE( "Error disconnecting command pipe", error );
        return false;
    }
    result = CloseHandle( _command_pipe_h );
    if( !result )
    {
        DWORD error = GetLastError();
        WINDOWS_ERROR_MESSAGE( "Error closing command pipe", error );
        return false;
    }
    _command_pipe_h = INVALID_HANDLE_VALUE;
    return result;
}

bool TransferServer::_create_data_smem()
{
    _data_smem_h = CreateFileMappingA( INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, _data_smem_size, _data_smem_name );
    if( IS_HANDLE_INVALID( _data_smem_h ) )
    {
        WINDOWS_ERROR_MESSAGE( "Error creating data shared memory", GetLastError() );
        return false;
    }

    _data_smem_raw = static_cast<u8*>(MapViewOfFile( _data_smem_h, FILE_MAP_ALL_ACCESS, 0, 0, _data_smem_size ));
    if( _data_smem_raw == nullptr )
    {
        WINDOWS_ERROR_MESSAGE( "Error mapping data shared memory", GetLastError() );
        return false;
    }
    _data_smem = std::span<u8>(_data_smem_raw, _data_smem_size);
    return true;
}

bool 
TransferServer::_create_params_smem()
{
    _params_smem_h = CreateFileMappingA( INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                         0, sizeof( SharedMemoryParams ), _params_smem_name );
    if( IS_HANDLE_INVALID( _params_smem_h ) )
    {
        WINDOWS_ERROR_MESSAGE( "Error creating header shared memory", GetLastError() );
        return false;
    }
    _parameters_smem = static_cast< SharedMemoryParams* >( MapViewOfFile( _params_smem_h, FILE_MAP_ALL_ACCESS,
                                                                          0, 0, sizeof( SharedMemoryParams ) ) );

    if( _parameters_smem == NULL )
    {
        WINDOWS_ERROR_MESSAGE( "Error mapping header shared memory", GetLastError() );
        return false;
    }
    return true;
}

bool 
TransferServer::_cleanup_smem()
{
    if( _data_smem_raw )
    {
        UnmapViewOfFile( _data_smem_raw );
        _data_smem_raw = nullptr;

		_data_smem = std::span<u8>();

        CloseHandle( _data_smem_h );
        _data_smem_h = nullptr;
    }

    if( _parameters_smem )
    {
        UnmapViewOfFile( _parameters_smem );
        _parameters_smem = nullptr;

        CloseHandle( _params_smem_h );
        _params_smem_h = nullptr;
    }

    return true;
}

bool
TransferServer::respond_ack()
{
    CommandPipeMessage ack_message = { CudaCommand::ACK, 0, 0 };
    return _send_command_response( ack_message );
}
bool
TransferServer::respond_success( u32 output_size )
{
    CommandPipeMessage success_message = { CudaCommand::SUCCESS, output_size, 0 };
    return _send_command_response( success_message );
}
bool
TransferServer::respond_error()
{
    CommandPipeMessage error_message = { CudaCommand::ERR, 0, 0 };
    return _send_command_response( error_message );
}

bool
TransferServer::write_output_data( std::span<const u8> output_data )
{
    if( output_data.size() > _data_smem_size )
    {
        std::cerr << "Output data size exceeds shared memory size: " << output_data.size() << " > " << _data_smem_size << std::endl;
        return false;
    }
    memcpy( _data_smem.data(), output_data.data(), output_data.size() );
    return true;
}

std::optional<CommandPipeMessage> 
TransferServer::wait_for_command()
{
    DWORD bytes_read = 0;
    CommandPipeMessage command;
    uint elapsed = 0;

    while( elapsed < COMMAND_WAIT_TIMEOUT )
    {
        if( ReadFile( _command_pipe_h, &command, sizeof( command ), &bytes_read, NULL ) )
        {
            if( bytes_read == sizeof( command ) )
            {
                if( command.data_size > _data_smem_size )
                {
                    std::cerr << "Data size too large: " << command.data_size << ", Max: " << _data_smem_size << std::endl;
                    respond_error();
                    return std::nullopt;
                }
                return command;
            }
        }

        DWORD error = GetLastError();
        if( error == ERROR_BROKEN_PIPE )
        {
            // Client disconnected
            std::cerr << "Client disconnected" << std::endl;
            if(!_restart_command_pipe() )
            {
                std::cerr << "Error restarting command pipe" << std::endl;
                return std::nullopt;
            }
            continue;
        }
        else if( error == ERROR_NO_DATA || error == ERROR_PIPE_LISTENING || error == ERROR_PIPE_NOT_CONNECTED)
        {
            // No data available, continue waiting
            Sleep( COMMAND_POLL_PERIOD );
            elapsed += COMMAND_POLL_PERIOD;
            continue;
        }
        else if( error != ERROR_SUCCESS)
        {
            WINDOWS_ERROR_MESSAGE( "Error reading command pipe", error );
            return std::nullopt;
        }

        Sleep( COMMAND_POLL_PERIOD );
        elapsed += COMMAND_POLL_PERIOD;
    }

    std::cerr << "Command wait timed out" << std::endl;
    return std::nullopt;
}


bool 
TransferServer::_send_command_response( CommandPipeMessage message )
{
    DWORD bytes_written = 0;
    if( !WriteFile( _command_pipe_h, &message, sizeof( message ), &bytes_written, NULL ) )
    {
        DWORD error = GetLastError();
        std::cerr << "Error writing to command pipe: " << format_windows_error_message(error) << std::endl;
        return false;
    }

    return true;
}