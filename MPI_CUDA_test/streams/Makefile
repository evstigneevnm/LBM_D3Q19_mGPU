MPI_INCLUDE_DIR = /usr/local/ompi_3_1_2/include
MPI_LIB_DIR = /usr/local/ompi_3_1_2/lib
mpi_aware_copy: mpi_aware_copy.cu
	nvcc -arch=sm_35 -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) mpi_aware_copy.cu -o mpi_aware_copy -lmpi
mpi_aware_copy_int_ahead: mpi_aware_copy.cu
	nvcc -arch=sm_35 -DINTERNAL_AHEAD -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) mpi_aware_copy.cu -o mpi_aware_copy_int_ahead -lmpi
mpi_aware_copy_stream: mpi_aware_copy.cu
	nvcc -arch=sm_35 -DUSE_STREAM -I$(MPI_INCLUDE_DIR) -L$(MPI_LIB_DIR) mpi_aware_copy.cu -o mpi_aware_copy_stream -lmpi
test_kernels_streams: test_kernels_streams.cu
	nvcc -arch sm_35 test_kernels_streams.cu -o test_kernels_streams