compile:
	2>result_make.txt
	nvcc -I/usr/local/ompi_3_1_2/include -g -m64 -arch=sm_35 test_cuda_aware_MPI.cu -o test_cuda_aware_MPI.o -c 2>>result_make.txt
link:	
	2>result_make.txt
	/usr/local/ompi_3_1_2/bin/mpic++ -g test_cuda_aware_MPI.o -L/usr/local/cuda/lib64 -L/usr/local/ompi_3_1_2/lib -lcudart -o test_cuda_aware_MPI.exe 2>>result_make.txt