deb_mpch:
	2>result_make.txt
	g++ -I/usr/include/mpich -g  mpi_send_test.c -o mpi_send_test.o -c 2>>result_make.txt
	mpic++.mpich -g mpi_send_test.o -o mpich_mpi_send_test.exe 2>>result_make.txt
deb_ompi:
	2>result_make.txt
	g++ -I/usr/local/ompi_3_1_2/include -g  mpi_send_test.c -o mpi_send_test.o -c 2>>result_make.txt
	/usr/local/ompi_3_1_2/bin/mpic++  -g mpi_send_test.o -L/usr/local/ompi_3_1_2/lib -o ompi_mpi_send_test.exe 2>>result_make.txt