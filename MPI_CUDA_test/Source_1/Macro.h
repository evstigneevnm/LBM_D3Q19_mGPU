#pragma once

#include <cuda_runtime.h>

#ifndef real
    #define real float
#endif

#ifndef MPI_real
    #define MPI_real MPI_FLOAT
#endif

#ifndef MASTER
    #define MASTER 0
#endif

#ifndef stride
    #define stride 1
#endif

#ifndef Q
    #define Q 19
#endif

#ifndef I4
    #define I4(i,j,k,q) ( (q)*(Nx)*(Ny)*(Nz)+(i)*(Ny)*(Nz)+(Nz)*(j)+(k))
#endif

#ifndef I3_MPI
    #define I3_MPI(q,j,k,Nj,Nk) ( (q)+(j)*19+(k)*19*(Nj) )
#endif

#ifndef I3
    //#define I3(i,j,k) ( (i)+(Nx)*(j)+(Ny)*(Nx)*(k) ) //(i-1)*n2*n3 + (j-1)*n3 + (k-1)
    #define I3(i,j,k) ( (i)*(Ny)*(Nz)+(Nz)*(j)+(k) )
#endif

#ifndef I3P
    #define I3P(i,j,k)  (Ny)*(Nz)*((i)>(Nx-1)?(i)-(Nx):(i)<0?((Nx)+(i)):(i)) + (Nz)*((j)>(Ny-1)?(j)-(Ny):(j)<0?(Ny+(j)):(j)) + ((k)>(Nz-1)?(k)-(Nz):(k)<0?(Nz+(k)):(k))
#endif


#ifndef I2
    //#define I2(i,j) ( (i)+(Nx)*(j) )
    #define I2(n,t) ((n)+(3)*(t))
#endif

#ifndef I2p
    #define I2p(p,n,t) ((p)+3*(n)+3*(NumP)*(t))  //No - number of points!
#endif

const real w[Q]={1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};


#define FLUID 0
#define MPI_block 1
#define WALL 2
#define PERIODIC 3
#define IN 4
#define OUT 5

/************************************************************************/
/* GPU KERNELS!!!!                                                      */
/************************************************************************/
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16




/************************************************************************/
/* structures!!!!                                                       */
/************************************************************************/
struct microscopic_variables
{
    real *d0;
    real *d1;
    real *d2;
    real *d3;
    real *d4;
    real *d5;
    real *d6;
    real *d7;
    real *d8;
    real *d9;
    real *d10;
    real *d11;
    real *d12;
    real *d13;
    real *d14;
    real *d15;
    real *d16;
    real *d17;
    real *d18;   
};

struct macroscopic_variables
{
    real *rho;
    real *ux;
    real *uy;
    real *uz;
    real *abs_u;
    real *rot_x;
    real *rot_y;
    real *rot_z;
    real *abs_rot;
    real *s;
    real *lk;
    real *H;
};

struct control_variables
{
    int *bc;
    int *proc;

};

struct communication_variables
{
    cudaStream_t streams[2];
    int myrank;
    int totalrank;
    int Nx;
    int Ny;
    int Nz;
    real Lx;
    real Ly;
    real Lz;
    real L0x;
    real L0y;
    real L0z;
    real dh;
    int timesteps;
    real Reynolds;
    int device; //0 - CPU, 1 - GPU, 2 - PHI
    int Blade_ID;
    int PCI_ID;
    int device_ID;
    char MPI_sections;

    char FaceA;
    int FaceAproc;
    char FaceB;
    int FaceBproc;
    char FaceC;
    int FaceCproc;
    char FaceD;
    int FaceDproc;
    char FaceE;
    int FaceEproc;
    char FaceF;
    int FaceFproc;

    real *Face1BuferSend_host;
    real *Face2BuferSend_host;    
    real *Face1BuferRecv_host;
    real *Face2BuferRecv_host;    

    real *Face1BuferSend_device;
    real *Face2BuferSend_device;
    real *Face1BuferRecv_device;
    real *Face2BuferRecv_device;    

    int Face1Bufer_size;
    int Face2Bufer_size;

    int Face1proc;
    int Face2proc;

};