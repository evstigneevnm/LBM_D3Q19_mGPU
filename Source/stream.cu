#include "stream.h"



__global__ void kernel_stream3D_0_18_forward(int Nx, int Ny, int Nz, int* bc_v, microscopic_variables MV_d_source, microscopic_variables MV_d_dest)
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




        real vf0 = MV_d_source.d0[kk];
        real vf1 = MV_d_source.d1[kk];
        real vf2 = MV_d_source.d2[kk];
        real vf3 = MV_d_source.d3[kk];
        real vf4 = MV_d_source.d4[kk];
        real vf5 = MV_d_source.d5[kk];
        real vf6 = MV_d_source.d6[kk];
        real vf7 = MV_d_source.d7[kk];
        real vf8 = MV_d_source.d8[kk];
        real vf9 = MV_d_source.d9[kk];
        real vf10 = MV_d_source.d10[kk];
        real vf11 = MV_d_source.d11[kk];
        real vf12 = MV_d_source.d12[kk];
        real vf13 = MV_d_source.d13[kk];
        real vf14 = MV_d_source.d14[kk];
        real vf15 = MV_d_source.d15[kk];
        real vf16 = MV_d_source.d16[kk];
        real vf17 = MV_d_source.d17[kk];
        real vf18 = MV_d_source.d18[kk];

        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};




        //2<=>1; 4<=>3; 6<=>5; 8<=>7; 10<=>9; 12<=>11; 14<=>13; 16<=>15; 18<=>17
            
        // if((i<1)||(i>Nx-2)||(j<1)||(j>Ny-2)||(k<1)||(k>Nz-2))
        //     return;
                                             
        MV_d_dest.d0[kk]=vf0;
                
        int _i=i+cx[1]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        int _j=j+cy[1]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        int _k=k+cz[1]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        int kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        //this is BAD!!!
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {
            
        }
        else
            MV_d_dest.d1[kk1]=vf1;


        _i=i+cx[2]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[2]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[2]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else        
            MV_d_dest.d2[kk1]=vf2;


        _i=i+cx[3]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[3]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[3]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else        
            MV_d_dest.d3[kk1]=vf3;


        _i=i+cx[4]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[4]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[4]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else 
            MV_d_dest.d4[kk1]=vf4;


        _i=i+cx[5]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[5]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[5]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d5[kk1]=vf5;


        _i=i+cx[6]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[6]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[6]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d6[kk1]=vf6;




        _i=i+cx[7]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[7]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[7]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d7[kk1]=vf7;


        _i=i+cx[8]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[8]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[8]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d8[kk1]=vf8;


        _i=i+cx[9]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[9]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[9]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d9[kk1]=vf9;



        _i=i+cx[10]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[10]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[10]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d10[kk1]=vf10;


        _i=i+cx[11]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[11]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[11]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d11[kk1]=vf11;


        _i=i+cx[12]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[12]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[12]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d12[kk1]=vf12;



        _i=i+cx[13]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[13]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[13]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d13[kk1]=vf13;


        _i=i+cx[14]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[14]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[14]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d14[kk1]=vf14;


        _i=i+cx[15]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[15]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[15]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d15[kk1]=vf15;


        _i=i+cx[16]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[16]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[16]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else             
            MV_d_dest.d16[kk1]=vf16;


        _i=i+cx[17]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[17]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[17]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d17[kk1]=vf17;


        _i=i+cx[18]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[18]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[18]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        if((_i<0)||(_i>Nx-1)||(_j<0)||(_j>Ny-1)||(_k<0)||(_k>Nz-1))
        {}
        else         
            MV_d_dest.d18[kk1]=vf18;
         

    }

}





__global__ void kernel_stream3D_0_18_forward_periodic(int Nx, int Ny, int Nz, int* bc_v, microscopic_variables MV_d_source, microscopic_variables MV_d_dest)
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




        real vf0 = MV_d_source.d0[kk];
        real vf1 = MV_d_source.d1[kk];
        real vf2 = MV_d_source.d2[kk];
        real vf3 = MV_d_source.d3[kk];
        real vf4 = MV_d_source.d4[kk];
        real vf5 = MV_d_source.d5[kk];
        real vf6 = MV_d_source.d6[kk];
        real vf7 = MV_d_source.d7[kk];
        real vf8 = MV_d_source.d8[kk];
        real vf9 = MV_d_source.d9[kk];
        real vf10 = MV_d_source.d10[kk];
        real vf11 = MV_d_source.d11[kk];
        real vf12 = MV_d_source.d12[kk];
        real vf13 = MV_d_source.d13[kk];
        real vf14 = MV_d_source.d14[kk];
        real vf15 = MV_d_source.d15[kk];
        real vf16 = MV_d_source.d16[kk];
        real vf17 = MV_d_source.d17[kk];
        real vf18 = MV_d_source.d18[kk];

        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};




        //2<=>1; 4<=>3; 6<=>5; 8<=>7; 10<=>9; 12<=>11; 14<=>13; 16<=>15; 18<=>17
            
        // if((i<1)||(i>Nx-2)||(j<1)||(j>Ny-2)||(k<1)||(k>Nz-2))
        //     return;
                                             
        MV_d_dest.d0[kk]=vf0;
                
        int _i=i+cx[1]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        int _j=j+cy[1]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        int _k=k+cz[1]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        int kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d1[kk1]=vf1;


        _i=i+cx[2]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[2]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[2]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d2[kk1]=vf2;


        _i=i+cx[3]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[3]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[3]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d3[kk1]=vf3;


        _i=i+cx[4]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[4]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[4]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d4[kk1]=vf4;


        _i=i+cx[5]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[5]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[5]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d5[kk1]=vf5;


        _i=i+cx[6]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[6]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[6]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d6[kk1]=vf6;




        _i=i+cx[7]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[7]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[7]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d7[kk1]=vf7;


        _i=i+cx[8]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[8]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[8]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d8[kk1]=vf8;


        _i=i+cx[9]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[9]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[9]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d9[kk1]=vf9;



        _i=i+cx[10]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[10]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[10]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d10[kk1]=vf10;


        _i=i+cx[11]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[11]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[11]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d11[kk1]=vf11;


        _i=i+cx[12]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[12]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[12]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d12[kk1]=vf12;



        _i=i+cx[13]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[13]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[13]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d13[kk1]=vf13;


        _i=i+cx[14]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[14]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[14]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d14[kk1]=vf14;


        _i=i+cx[15]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[15]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[15]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d15[kk1]=vf15;


        _i=i+cx[16]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[16]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[16]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d16[kk1]=vf16;


        _i=i+cx[17]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[17]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[17]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d17[kk1]=vf17;


        _i=i+cx[18]; //if(_i<0) _i=Nx-1; if(_i>Nx-1) _i=0;
        _j=j+cy[18]; //if(_j<0) _j=Ny-1; if(_j>Ny-1) _j=0;
        _k=k+cz[18]; //if(_k<0) _k=Nz-1; if(_k>Nz-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(Nz)+_i*(Ny)*(Nz);
        MV_d_dest.d18[kk1]=vf18;
         

    }

}

