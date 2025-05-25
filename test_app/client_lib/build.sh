cuda_lib_header_path="../../include/cuda_toolkit"

cflags="-march=native -std=c11 -Wall"
libcflags="${cflags} -fPIC -shared -Wno-unused-variable  -I.. -I ${cuda_lib_header_path}"
ldflags="-lm"

output_dir="output"
mkdir -p ${output_dir}

cc=${CC:-cc}
build=release


cflags="${cflags} -I.. -I ${cuda_lib_header_path}"

debug_flags="-O0 -fuse-ld=lld -g -gcodeview -Wl,--pdb="	

ldflags="${ldflags} -lm -lgdi32 -lwinmm -L./${output_dir}"

${cc} ${libcflags} ${debug_flags} cuda_transfer_new.c -o ${output_dir}/cuda_transfer_new.dll \
	${ldflags}


matlab_libcflags="${libcflags} -DMATLAB_CONSOLE"
matlab_ldflags="-llibmat -llibmex"

${cc} ${matlab_libcflags} cuda_transfer.c -o ${output_dir}/cuda_transfer.dll \
	-L'C:/Program Files/MATLAB/R2024a/extern/lib/win64/mingw64' \
	${matlab_ldflags}

${cc} ${matlab_libcflags} cuda_transfer_new.c -o ${output_dir}/cuda_transfer_new_matlab.dll \
	-L'C:/Program Files/MATLAB/R2024a/extern/lib/win64/mingw64' \
	${matlab_ldflags}

${cc} ${cflags} ${debug_flags} test_lib.c -o ${output_dir}/test_lib.exe ${ldflags} -lcuda_transfer_new


cp -f cuda_transfer.h ${output_dir}/cuda_transfer.h
cp -f cuda_transfer_new.h ${output_dir}/cuda_transfer_new.h
cp -f ../parameter_defs.h ${output_dir}/parameter_defs.h
cp -f ${cuda_lib_header_path}/cuda_beamformer_parameters.h ${output_dir}/cuda_beamformer_parameters.h