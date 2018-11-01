#include "stream.h"



__global__ void kernel_copy_0_18(int Nx, int Ny, int Nz, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6,
                                 real *f7, real *f8, real *f9, 
                                real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                 real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, real *f6p, 
                                 real *f7p, real *f8p, real *f9p, 
                                 real *f10p, real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p)
{


    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
            
        unsigned int i=xIndex, j=yIndex, k=zIndex;

        unsigned int kk=I3(i,j,k);//k+j*(Nz)+i*(Ny)*(Nz);
        f0p[kk]=f0[kk];
        f1p[kk]=f1[kk];
        f2p[kk]=f2[kk];
        f3p[kk]=f3[kk];
        f4p[kk]=f4[kk];
        f5p[kk]=f5[kk];
        f6p[kk]=f6[kk];
        f7p[kk]=f7[kk];
        f8p[kk]=f8[kk];
        f9p[kk]=f9[kk];
        f10p[kk]=f10[kk];
        f11p[kk]=f11[kk];
        f12p[kk]=f12[kk];
        f13p[kk]=f13[kk];
        f14p[kk]=f14[kk];
        f15p[kk]=f15[kk];
        f16p[kk]=f16[kk];
        f17p[kk]=f17[kk];
        f18p[kk]=f18[kk];




    }

}


__global__ void kernel_stream3D_0_18_forward(int Nx, int Ny, int Nz, int* bc_v, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, 
                                            real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                            real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, real *f6p, real *f7p, real *f8p, real *f9p, 
                                            real *f10p, real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p)
{





    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
            
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        unsigned int kk=I3(i,j,k);//k+j*(Nz)+i*(Ny)*(Nz);




        real vf0=f0[kk];
        real vf1=f1[kk];
        real vf2=f2[kk];
        real vf3=f3[kk];
        real vf4=f4[kk];
        real vf5=f5[kk];
        real vf6=f6[kk];
        real vf7=f7[kk];
        real vf8=f8[kk];
        real vf9=f9[kk];
        real vf10=f10[kk];
        real vf11=f11[kk];
        real vf12=f12[kk];
        real vf13=f13[kk];
        real vf14=f14[kk];
        real vf15=f15[kk];
        real vf16=f16[kk];
        real vf17=f17[kk];
        real vf18=f18[kk];

        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};




        //2<=>1; 4<=>3; 6<=>5; 8<=>7; 10<=>9; 12<=>11; 14<=>13; 16<=>15; 18<=>17
            
        // if((i<1)||(i>Nx-2)||(j<1)||(j>Ny-2)||(k<1)||(k>Nz-2))
        //     return;
                                             
        f0p[kk]=vf0;
                
        int _i=i+cx[1]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        int _j=j+cy[1]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        int _k=k+cz[1]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        int kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        //this is BAD!!!
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else
            f1p[kk1]=vf1;


        _i=i+cx[2]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[2]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[2]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else        
        f2p[kk1]=vf2;


        _i=i+cx[3]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[3]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[3]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else        
            f3p[kk1]=vf3;


        _i=i+cx[4]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[4]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[4]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else 
            f4p[kk1]=vf4;


        _i=i+cx[5]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[5]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[5]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f5p[kk1]=vf5;


        _i=i+cx[6]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[6]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[6]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f6p[kk1]=vf6;




        _i=i+cx[7]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[7]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[7]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f7p[kk1]=vf7;


        _i=i+cx[8]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[8]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[8]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f8p[kk1]=vf8;


        _i=i+cx[9]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[9]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[9]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f9p[kk1]=vf9;



        _i=i+cx[10]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[10]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[10]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f10p[kk1]=vf10;


        _i=i+cx[11]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[11]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[11]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f11p[kk1]=vf11;


        _i=i+cx[12]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[12]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[12]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f12p[kk1]=vf12;



        _i=i+cx[13]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[13]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[13]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f13p[kk1]=vf13;


        _i=i+cx[14]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[14]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[14]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f14p[kk1]=vf14;


        _i=i+cx[15]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[15]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[15]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f15p[kk1]=vf15;


        _i=i+cx[16]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[16]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[16]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else             
            f16p[kk1]=vf16;


        _i=i+cx[17]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[17]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[17]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f17p[kk1]=vf17;


        _i=i+cx[18]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[18]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[18]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            f18p[kk1]=vf18;
         

    }

}




__global__ void kernel_wall3D_0_18(int Nx, int Ny, int Nz, int* bc_v, 
                                   real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, real *f10,
                                   real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                   real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, real *f6p, real *f7p, real *f8p, real *f9p, real *f10p, 
                                   real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p)
{





    unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
    // step 1: compute gridIndex in 1-D and 1-D data index "index_in"
    unsigned int sizeOfData=(unsigned int) Nx*Ny*Nz;
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
    index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
    // step 2: extract 3-D data index via
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1
    if ( index_in < sizeOfData )
    {
        t1 =  index_in/ Nz; 
        zIndex = index_in - Nz*t1 ;
        xIndex =  t1/ Ny; 
        yIndex = t1 - Ny * xIndex ;
            
        unsigned int i=xIndex, j=yIndex, k=zIndex;
        unsigned int kk=I3(i,j,k);




        real vf0=f0[I3(i,j,k)];
        // real vf1=f1[I3(i,j,k)];
        // real vf2=f2[I3(i,j,k)];
        // real vf3=f3[I3(i,j,k)];
        // real vf4=f4[I3(i,j,k)];
        // real vf5=f5[I3(i,j,k)];
        // real vf6=f6[I3(i,j,k)];
        // real vf7=f7[I3(i,j,k)];
        // real vf8=f8[I3(i,j,k)];
        // real vf9=f9[I3(i,j,k)];
        // real vf10=f10[I3(i,j,k)];
        // real vf11=f11[I3(i,j,k)];
        // real vf12=f12[I3(i,j,k)];
        // real vf13=f13[I3(i,j,k)];
        // real vf14=f14[I3(i,j,k)];
        // real vf15=f15[I3(i,j,k)];
        // real vf16=f16[I3(i,j,k)];
        // real vf17=f17[I3(i,j,k)];
        // real vf18=f18[I3(i,j,k)];


        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};



        //2<=>1; 4<=>3; 6<=>5; 8<=>7; 10<=>9; 12<=>11; 14<=>13; 16<=>15; 18<=>17

        if(bc_v[kk]==WALL)
        {                          
            f0p[kk]=vf0;
                    
            //  int _i=i+cx[1]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  int _j=j+cy[1]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  int _k=k+cz[1]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  int kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f1p[kk]=f2[kk];


            //  _i=i+cx[2]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[2]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[2]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f2p[kk]=f1[kk];


            //  _i=i+cx[3]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[3]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[3]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f3p[kk]=f4[kk];


            //  _i=i+cx[4]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[4]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[4]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f4p[kk]=f3[kk];


            //  _i=i+cx[5]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[5]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[5]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f5p[kk]=f6[kk];


            //  _i=i+cx[6]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[6]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[6]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f6p[kk]=f5[kk];




            //  _i=i+cx[7]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[7]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[7]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f7p[kk]=f8[kk];


            //  _i=i+cx[8]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[8]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[8]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f8p[kk]=f7[kk];


            //  _i=i+cx[9]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[9]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[9]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f9p[kk]=f10[kk];

            //  _i=i+cx[10]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[10]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[10]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f10p[kk]=f9[kk];

            //  _i=i+cx[11]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[11]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[11]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f11p[kk]=f12[kk];


            //  _i=i+cx[12]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[12]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[12]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f12p[kk]=f11[kk];


            //  _i=i+cx[13]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[13]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[13]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f13p[kk]=f14[kk];


            //  _i=i+cx[14]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[14]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[14]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f14p[kk]=f13[kk];


            //  _i=i+cx[15]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[15]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[15]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f15p[kk]=f16[kk];


            //  _i=i+cx[16]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[16]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[16]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f16p[kk]=f15[kk];




            //  _i=i+cx[17]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[17]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[17]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f17p[kk]=f18[kk];


            //  _i=i+cx[18]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
            //  _j=j+cy[18]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
            //  _k=k+cz[18]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
            //  kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
            f18p[kk]=f17[kk];

        }
        /*
        else
        {
            f0p[kk]=vf0;
            f1p[kk]=vf1;
            f2p[kk]=vf2;
            f3p[kk]=vf3;
            f4p[kk]=vf4;
            f5p[kk]=vf5;
            f6p[kk]=vf6;
            f7p[kk]=vf7;
            f8p[kk]=vf8;
            f9p[kk]=vf9;
            f10p[kk]=vf10;
            f11p[kk]=vf11;
            f12p[kk]=vf12;
            f13p[kk]=vf13;
            f14p[kk]=vf14;
            f15p[kk]=vf15;
            f16p[kk]=vf16;
            f17p[kk]=vf17;
            f18p[kk]=vf18;

        }
        //*/


    }

}

