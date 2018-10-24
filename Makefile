MPI_INCLUDE_DIR = /usr/local/ompi_3_1_2/include
MPI_LIB_DIR = /usr/local/ompi_3_1_2/lib
CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/ompi_3_1_2/lib

reduction:
	#cuda self-written optimized reduction
	2>result_make.txt
	nvcc -g -m64 -arch=sm_35 Source/min_max_reduction.cu -o O/min_max_reduction.o -c 2>>result_make.txt

solver:
	#solver files...
	2>result_make.txt
	nvcc -g -m64 -arch=sm_35 Source/stream_mpi.cu -o O/stream_mpi.o -c 2>>result_make.txt
	g++ -g -I$(CUDA_INCLUDE_DIR) Source/initial_conditions.cpp -lm -o O/initial_conditions.o -c 2>>result_make.txt
	nvcc -g -m64 -arch=sm_35 Source/stream.cu -o O/stream.o -c 2>>result_make.txt
	nvcc -g -m64 -arch=sm_35 Source/collide.cu -o O/collide.o -c 2>>result_make.txt
	nvcc -I$(MPI_INCLUDE_DIR)  -g -m64 -arch=sm_35 Source/single_step.cu -o O/single_step.o -c 2>>result_make.txt


supp:
	#supplimentary files...
	2>result_make.txt
	nvcc -g -m64 -arch=sm_35 Source/cuda_support.cu -c -o O/cuda_support.o 2>>result_make.txt
	g++ -g -I$(CUDA_INCLUDE_DIR) -I$(MPI_INCLUDE_DIR) Source/IO_operations.cpp -lm -c -o O/IO_operations.o 2>>result_make.txt

communication:
	#communication MPI files
	2>result_make.txt
	nvcc -std=c++11 -I$(MPI_INCLUDE_DIR) -lm -g -arch=sm_35 -m64 -c Source/map.cu -o O/map.o 2>>result_make.txt

main:
	#main file and link
	2>result_make.txt
	nvcc -std=c++11 -I$(MPI_INCLUDE_DIR) -lm -g -arch=sm_35 -m64 -c Source/LBM_D3Q19.cu -o O/LBM_D3Q19.o 2>>result_make.txt
	cd O; \
	nvcc LBM_D3Q19.o map.o single_step.o stream.o stream_mpi.o collide.o cuda_support.o initial_conditions.o IO_operations.o -L$(CUDA_LIB_DIR) -L$(MPI_LIB_DIR) -lmpi -lm -o ../LBM_D3Q19.exe 2>>../result_make.txt

all:
	make communication solver supp solver main

clear:
	cd O; \
	rm *.o 
	rm *.pos *.dat LBM_D3Q19.exe
	


	
	
	
