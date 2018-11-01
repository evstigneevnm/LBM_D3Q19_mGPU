compile:
	2>result_make.txt
	nvcc -I/usr/include/mpich -g -m64 -arch=sm_35 test_cuda_aware_MPICH.cu -o test_cuda_aware_MPICH.o -c 2>>result_make.txt
link:	
	2>result_make.txt
	mpic++.mpich -g test_cuda_aware_MPICH.o -L/usr/local/cuda/lib64 -lcudart -o test_cuda_aware_MPICH.exe 2>>result_make.txt