# CUDA Ultrasound Toolkit

This repository provides GPU accelerated ultrasound processing utilities. The code is organized as a Visual Studio solution and exposes two public APIs.

## Library Interfaces

### C API with OpenGL buffers
The header [`cuda_toolkit_ogl.h`](include/cuda_toolkit/cuda_toolkit_ogl.h) defines a C interface designed to share OpenGL buffers with CUDA. Functions allow the application to initialise kernel parameters, register OpenGL buffer IDs, decode data and perform Hilbert transforms:

- `init_cuda_configuration` – configure kernel sizes once input and output dimensions are known.
- `register_cuda_buffers` – register raw and decoded RF data buffers from OpenGL.
- `cuda_set_channel_mapping` and `cuda_set_match_filter` – load channel map and optional match filter.
- `cuda_decode` / `cuda_hilbert` – run decoding or Hilbert transforms on registered buffers.
- `deinit_cuda_configuration` – release CUDA resources.

These functions are declared around lines 32‑75 of the header.

### C++ Convenience API
The header [`cuda_toolkit.hpp`](include/cuda_toolkit/cuda_toolkit.hpp) exposes a simpler interface:

```cpp
namespace cuda_toolkit {
    bool beamform(std::span<const uint8_t> input_data,
                  std::span<uint8_t> output_data,
                  const CudaBeamformerParameters& bp);
}
```

The function takes raw RF samples and fills an output buffer with the beamformed volume.

## Features
The library performs several steps of the ultrasound pipeline on the GPU:

- Data conversion and channel reordering.
- Optional Hadamard decoding for READI schemes.
- Hilbert transform with optional match filtering.
- 3‑D beamforming using parameters described by `CudaBeamformerParameters`.

The parameter structure provides transducer orientation, output volume size, frequency information and other options such as interpolation or coherency weighting【F:include/cuda_toolkit/cuda_beamformer_parameters.h†L57-L113】.

## Building
The project contains Visual Studio projects for Windows. Open `cuda_toolkit.sln` and build the `cuda_toolkit` dynamic library and the `test_app` executable. CUDA 12.x and the NVIDIA build tools must be installed.

A small client library is provided in `test_app/client_lib`. Running `build.sh` compiles a lightweight DLL (`cuda_transfer`) and a command line example using the system C compiler.

## Running the test application
1. Start `test_app.exe` – this process waits for commands via the named pipe defined in `communication_params.h`【F:test_app/src/communication_params.h†L10-L31】.
2. Use the client library (`cuda_transfer`) to send the `BEAMFORM_VOLUME` command and raw data through the shared memory segment. The library exports `beamform_i16` and `beamform_f32` for different input types【F:test_app/client_lib/cuda_transfer.h†L17-L19】.
3. Results are written back to the shared memory block and returned to the client.

The server handles additional opcodes (`SVD_FILTER`, `NCC_MOTION_DETECT`), though the implementations are placeholders at this time.


