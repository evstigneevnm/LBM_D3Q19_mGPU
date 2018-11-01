#include "collide.h"



__global__ void kernel_macro_0_18(int Nx, int Ny, int Nz, real *ux_v, real *uy_v, real *uz_v, real *ro_v, int* bc_v, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18){
    

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
        int kk=k+j*(Nz)+i*(Ny)*(Nz);

        //Local constants!
        real ux_in=0.1,uy_in=0.0,uz_in=0.0;
        real w[Q]={1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
        //2<=>1; 4<=>3; 6<=>5; 8<=>7; 10<=>9; 12<=>11; 14<=>13; 16<=>15; 18<=>17


        real vf0=f0[I3(i,j,k)];
        real vf1=f1[I3(i,j,k)];
        real vf2=f2[I3(i,j,k)];
        real vf3=f3[I3(i,j,k)];
        real vf4=f4[I3(i,j,k)];
        real vf5=f5[I3(i,j,k)];
        real vf6=f6[I3(i,j,k)];
        real vf7=f7[I3(i,j,k)];
        real vf8=f8[I3(i,j,k)];
        real vf9=f9[I3(i,j,k)];
        real vf10=f10[I3(i,j,k)];
        real vf11=f11[I3(i,j,k)];
        real vf12=f12[I3(i,j,k)];
        real vf13=f13[I3(i,j,k)];
        real vf14=f14[I3(i,j,k)];
        real vf15=f15[I3(i,j,k)];
        real vf16=f16[I3(i,j,k)];
        real vf17=f17[I3(i,j,k)];
        real vf18=f18[I3(i,j,k)];



    //init macroscopic variables!
    /*  
        real ro=1.0+((vf0-w[0])+(vf1-w[1])+
            (vf2-w[2])+(vf3-w[3])+(vf4-w[4])+
            (vf5-w[5])+(vf6-w[6])+(vf7-w[7])+
            (vf8-w[8])+(vf9-w[9])+(vf10-w[10])+
            (vf11-w[11])+(vf12-w[12])+(vf13-w[13])+
            (vf14-w[14])+(vf15-w[15])+(vf16-w[16])+(vf17-w[17])+(vf18-w[18]));

        real v_x=(cx[0]*(vf0-w[0])+cx[1]*(vf1-w[1])+
            cx[2]*(vf2-w[2])+cx[3]*(vf3-w[3])+cx[4]*(vf4-w[4])+
            cx[5]*(vf5-w[5])+cx[6]*(vf6-w[6])+cx[7]*(vf7-w[7])+
            cx[8]*(vf8-w[8])+cx[9]*(vf9-w[9])+cx[10]*(vf10-w[10])+
            cx[11]*(vf11-w[11])+cx[12]*(vf12-w[12])+cx[13]*(vf13-w[13])+
            cx[14]*(vf14-w[14])+cx[15]*(vf15-w[15])+cx[16]*(vf16-w[16])+cx[17]*(vf17-w[17])+cx[18]*(vf18-w[18]))/ro;
        
        real v_y=(cy[0]*(vf0-w[0])+cy[1]*(vf1-w[1])+
            cy[2]*(vf2-w[2])+cy[3]*(vf3-w[3])+cy[4]*(vf4-w[4])+
            cy[5]*(vf5-w[5])+cy[6]*(vf6-w[6])+cy[7]*(vf7-w[7])+
            cy[8]*(vf8-w[8])+cy[9]*(vf9-w[9])+cy[10]*(vf10-w[10])+
            cy[11]*(vf11-w[11])+cy[12]*(vf12-w[12])+cy[13]*(vf13-w[13])+
            cy[14]*(vf14-w[14])+cy[15]*(vf15-w[15])+cy[16]*(vf16-w[16])+cy[17]*(vf17-w[17])+cy[18]*(vf18-w[18]))/ro;    

        real v_z=(cz[0]*(vf0-w[0])+cz[1]*(vf1-w[1])+
            cz[2]*(vf2-w[2])+cz[3]*(vf3-w[3])+cz[4]*(vf4-w[4])+
            cz[5]*(vf5-w[5])+cz[6]*(vf6-w[6])+cz[7]*(vf7-w[7])+
            cz[8]*(vf8-w[8])+cz[9]*(vf9-w[9])+cz[10]*(vf10-w[10])+
            cz[11]*(vf11-w[11])+cz[12]*(vf12-w[12])+cz[13]*(vf13-w[13])+
            cz[14]*(vf14-w[14])+cz[15]*(vf15-w[15])+cz[16]*(vf16-w[16])+cz[17]*(vf17-w[17])+cz[18]*(vf18-w[18]))/ro;
    //*/


    //*
        real ro=vf0+vf1+vf2+vf3+vf4+vf5+vf6+vf7+vf8+vf9+vf10+vf11+vf12+vf13+vf14+vf15+vf16+vf17+vf18;
            

        real v_x=(cx[0]*vf0+cx[1]*vf1+cx[2]*vf2+cx[3]*vf3+cx[4]*vf4+cx[5]*vf5+cx[6]*vf6+cx[7]*vf7+cx[8]*vf8+cx[9]*vf9+cx[10]*vf10+cx[11]*vf11+cx[12]*vf12+cx[13]*vf13+cx[14]*vf14+cx[15]*vf15+cx[16]*vf16+cx[17]*vf17+cx[18]*vf18)/ro;
            
        real v_y=(cy[0]*vf0+cy[1]*vf1+cy[2]*vf2+cy[3]*vf3+cy[4]*vf4+cy[5]*vf5+cy[6]*vf6+cy[7]*vf7+cy[8]*vf8+cy[9]*vf9+cy[10]*vf10+cy[11]*vf11+cy[12]*vf12+cy[13]*vf13+cy[14]*vf14+cy[15]*vf15+cy[16]*vf16+cy[17]*vf17+cy[18]*vf18)/ro;

        real v_z=(cz[0]*vf0+cz[1]*vf1+cz[2]*vf2+cz[3]*vf3+cz[4]*vf4+cz[5]*vf5+cz[6]*vf6+cz[7]*vf7+cz[8]*vf8+cz[9]*vf9+cz[10]*vf10+cz[11]*vf11+cz[12]*vf12+cz[13]*vf13+cz[14]*vf14+cz[15]*vf15+cz[16]*vf16+cz[17]*vf17+cz[18]*vf18)/ro;

    //*/ Replace into separate kernel using mask!
        if(bc_v[I3(i,j,k)]==IN)
        {
                
            v_x=ux_in;//uinit[I3(0,j,k)];
            v_y=uy_in;
            v_z=uz_in;
            //  ro=ro_v[I3(i,j-1,k)];
            ro=ro_v[I3(i+1,j,k)];
        }
        else if(bc_v[I3(i,j,k)]==OUT)
        {
            ro=1.0;//+(Ny-float(j))*0.05;
            //v_x=fmaxf(ux_v[I3(i-1,j,k)],0.0);
            v_x=ux_v[I3(i-1,j,k)];
            v_y=uy_v[I3(i-1,j,k)];
            v_z=uz_v[I3(i-1,j,k)];
        }
            

        ro_v[I3(i,j,k)]=ro;
        ux_v[I3(i,j,k)]=v_x;
        uy_v[I3(i,j,k)]=v_y;
        uz_v[I3(i,j,k)]=v_z;
            
    }



}



__global__ void kernel_collide_0_18(real delta, real *ux_old_v, real gx, real gy, real gz, int Nx, int Ny, int Nz, real omega, real *ux_v, real *uy_v, real *uz_v, real *ro_v,  int* bc_v, 
                                    real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, 
                                    real *f6, real *f7, real *f8, real *f9, 
                                    real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                    real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, 
                                    real *f6p, real *f7p, real *f8p, real *f9p, 
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
        int kk=k+j*(Nz)+i*(Ny)*(Nz);

        //Local constants!
        real ux_in=0.1,uy_in=0.0,uz_in=0.0;
        const real w[Q]={1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
        //2<=>1; 4<=>3; 6<=>5; 8<=>7; 10<=>9; 12<=>11; 14<=>13; 16<=>15; 18<=>17

        //load micro variables!

        real vf0=f0[I3(i,j,k)];
        real vf1=f1[I3(i,j,k)];
        real vf2=f2[I3(i,j,k)];
        real vf3=f3[I3(i,j,k)];
        real vf4=f4[I3(i,j,k)];
        real vf5=f5[I3(i,j,k)];
        real vf6=f6[I3(i,j,k)];
        real vf7=f7[I3(i,j,k)];
        real vf8=f8[I3(i,j,k)];
        real vf9=f9[I3(i,j,k)];
        real vf10=f10[I3(i,j,k)];
        real vf11=f11[I3(i,j,k)];
        real vf12=f12[I3(i,j,k)];
        real vf13=f13[I3(i,j,k)];
        real vf14=f14[I3(i,j,k)];
        real vf15=f15[I3(i,j,k)];
        real vf16=f16[I3(i,j,k)];
        real vf17=f17[I3(i,j,k)];
        real vf18=f18[I3(i,j,k)];


        //load macro variables!
        // real ro=ro_v[I3(i,j,k)];
        // real v_x=ux_v[I3(i,j,k)];
        // real v_y=uy_v[I3(i,j,k)];
        // real v_z=uz_v[I3(i,j,k)];
        real ro=vf0+vf1+vf2+vf3+vf4+vf5+vf6+vf7+vf8+vf9+vf10+vf11+vf12+vf13+vf14+vf15+vf16+vf17+vf18;
            

        real v_x=(cx[0]*vf0+cx[1]*vf1+cx[2]*vf2+cx[3]*vf3+cx[4]*vf4+cx[5]*vf5+cx[6]*vf6+cx[7]*vf7+cx[8]*vf8+cx[9]*vf9+cx[10]*vf10+cx[11]*vf11+cx[12]*vf12+cx[13]*vf13+cx[14]*vf14+cx[15]*vf15+cx[16]*vf16+cx[17]*vf17+cx[18]*vf18)/ro;
            
        real v_y=(cy[0]*vf0+cy[1]*vf1+cy[2]*vf2+cy[3]*vf3+cy[4]*vf4+cy[5]*vf5+cy[6]*vf6+cy[7]*vf7+cy[8]*vf8+cy[9]*vf9+cy[10]*vf10+cy[11]*vf11+cy[12]*vf12+cy[13]*vf13+cy[14]*vf14+cy[15]*vf15+cy[16]*vf16+cy[17]*vf17+cy[18]*vf18)/ro;

        real v_z=(cz[0]*vf0+cz[1]*vf1+cz[2]*vf2+cz[3]*vf3+cz[4]*vf4+cz[5]*vf5+cz[6]*vf6+cz[7]*vf7+cz[8]*vf8+cz[9]*vf9+cz[10]*vf10+cz[11]*vf11+cz[12]*vf12+cz[13]*vf13+cz[14]*vf14+cz[15]*vf15+cz[16]*vf16+cz[17]*vf17+cz[18]*vf18)/ro;

    //*/ Replace into separate kernel using mask!
        if(bc_v[I3(i,j,k)]==IN)
        {
                
            v_x=ux_in;//uinit[I3(0,j,k)];
            v_y=uy_in;
            v_z=uz_in;
            //  ro=ro_v[I3(i,j-1,k)];
            ro=ro_v[I3(i+1,j,k)];
        }
        else if(bc_v[I3(i,j,k)]==OUT)
        {
            ro=1.0;//+(Ny-float(j))*0.05;
            //v_x=fmaxf(ux_v[I3(i-1,j,k)],0.0);
            v_x=ux_v[I3(i-1,j,k)];
            v_y=uy_v[I3(i-1,j,k)];
            v_z=uz_v[I3(i-1,j,k)];
        }
        else if(bc_v[I3(i,j,k)]==WALL){
            v_x=v_y=v_z=0.0;
        }
            

        ro_v[I3(i,j,k)]=ro;
        ux_v[I3(i,j,k)]=v_x;
        uy_v[I3(i,j,k)]=v_y;
        uz_v[I3(i,j,k)]=v_z;
            




        //Calculate equilibrium distribution
        real cs=1.0;//0.33333333333333333333333333333;
        real vv=0.0;
        real vf=(cx[0]-v_x)*gx+(cy[0]-v_y)*gy+(cz[0]-v_z)*gz;
        real f_eq0=w[0]*ro*(1.0-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[1]*v_x+cy[1]*v_y+cz[1]*v_z;
        vf=(cx[1]-v_x)*gx+(cy[1]-v_y)*gy+(cz[1]-v_z)*gz;
        real f_eq1=w[1]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[2]*v_x+cy[2]*v_y+cz[2]*v_z;
        vf=(cx[2]-v_x)*gx+(cy[2]-v_y)*gy+(cz[2]-v_z)*gz;
        real f_eq2=w[2]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[3]*v_x+cy[3]*v_y+cz[3]*v_z;
        vf=(cx[3]-v_x)*gx+(cy[3]-v_y)*gy+(cz[3]-v_z)*gz;
        real f_eq3=w[3]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[4]*v_x+cy[4]*v_y+cz[4]*v_z;
        vf=(cx[4]-v_x)*gx+(cy[4]-v_y)*gy+(cz[4]-v_z)*gz;
        real f_eq4=w[4]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[5]*v_x+cy[5]*v_y+cz[5]*v_z;
        vf=(cx[5]-v_x)*gx+(cy[5]-v_y)*gy+(cz[5]-v_z)*gz;
        real f_eq5=w[5]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[6]*v_x+cy[6]*v_y+cz[6]*v_z;
        vf=(cx[6]-v_x)*gx+(cy[6]-v_y)*gy+(cz[6]-v_z)*gz;
        real f_eq6=w[6]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[7]*v_x+cy[7]*v_y+cz[7]*v_z;
        vf=(cx[7]-v_x)*gx+(cy[7]-v_y)*gy+(cz[7]-v_z)*gz;    
        real f_eq7=w[7]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[8]*v_x+cy[8]*v_y+cz[8]*v_z;
        vf=(cx[8]-v_x)*gx+(cy[8]-v_y)*gy+(cz[8]-v_z)*gz;
        real f_eq8=w[8]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[9]*v_x+cy[9]*v_y+cz[9]*v_z;
        vf=(cx[9]-v_x)*gx+(cy[9]-v_y)*gy+(cz[9]-v_z)*gz;
        real f_eq9=w[9]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

                
        vv=cx[10]*v_x+cy[10]*v_y+cz[10]*v_z;
        vf=(cx[10]-v_x)*gx+(cy[10]-v_y)*gy+(cz[10]-v_z)*gz;
        real f_eq10=w[10]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[11]*v_x+cy[11]*v_y+cz[11]*v_z;
        vf=(cx[11]-v_x)*gx+(cy[11]-v_y)*gy+(cz[11]-v_z)*gz;
        real f_eq11=w[11]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);


        vv=cx[12]*v_x+cy[12]*v_y+cz[12]*v_z;
        vf=(cx[12]-v_x)*gx+(cy[12]-v_y)*gy+(cz[12]-v_z)*gz;
        real f_eq12=w[12]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[13]*v_x+cy[13]*v_y+cz[13]*v_z;
        vf=(cx[13]-v_x)*gx+(cy[13]-v_y)*gy+(cz[13]-v_z)*gz;
        real f_eq13=w[13]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[14]*v_x+cy[14]*v_y+cz[14]*v_z;
        vf=(cx[14]-v_x)*gx+(cy[14]-v_y)*gy+(cz[14]-v_z)*gz;
        real f_eq14=w[14]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[15]*v_x+cy[15]*v_y+cz[15]*v_z;
        vf=(cx[15]-v_x)*gx+(cy[15]-v_y)*gy+(cz[15]-v_z)*gz;
        real f_eq15=w[15]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[16]*v_x+cy[16]*v_y+cz[16]*v_z;
        vf=(cx[16]-v_x)*gx+(cy[16]-v_y)*gy+(cz[16]-v_z)*gz;
        real f_eq16=w[16]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[17]*v_x+cy[17]*v_y+cz[17]*v_z;
        vf=(cx[17]-v_x)*gx+(cy[17]-v_y)*gy+(cz[17]-v_z)*gz;
        real f_eq17=w[17]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[18]*v_x+cy[18]*v_y+cz[18]*v_z;
        vf=(cx[18]-v_x)*gx+(cy[18]-v_y)*gy+(cz[18]-v_z)*gz;
        real f_eq18=w[18]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        //Set PPDF boundaries for in and out

        //*
        if((bc_v[I3(i,j,k)]==IN)||(bc_v[I3(i,j,k)]==OUT))
        {
            vf0=f_eq0;  vf1=f_eq1;  vf2=f_eq2;  vf3=f_eq3;  vf4=f_eq4;  vf5=f_eq5;  
            vf6=f_eq6;  vf7=f_eq7;  vf8=f_eq8;  vf9=f_eq9;  vf10=f_eq10;    vf11=f_eq11;  
            vf12=f_eq12;  vf13=f_eq13;    vf14=f_eq14;    vf15=f_eq15; vf16=f_eq16;vf17=f_eq17; vf18=f_eq18;
        }

        //*/

        //We have collision!
        real f0c=vf0;
        real f1c=vf1;
        real f2c=vf2;
        real f3c=vf3;
        real f4c=vf4;
        real f5c=vf5;
        real f6c=vf6;
        real f7c=vf7;
        real f8c=vf8;
        real f9c=vf9;
        real f10c=vf10;
        real f11c=vf11;
        real f12c=vf12;
        real f13c=vf13;
        real f14c=vf14;
        real f15c=vf15;
        real f16c=vf16;
        real f17c=vf17;
        real f18c=vf18;

        if(bc_v[I3(i,j,k)]!=WALL)
        {
            f0c=vf0-omega*(vf0-f_eq0);
            f1c=vf1-omega*(vf1-f_eq1);
            f2c=vf2-omega*(vf2-f_eq2);
            f3c=vf3-omega*(vf3-f_eq3);
            f4c=vf4-omega*(vf4-f_eq4);
            f5c=vf5-omega*(vf5-f_eq5);
            f6c=vf6-omega*(vf6-f_eq6);
            f7c=vf7-omega*(vf7-f_eq7);
            f8c=vf8-omega*(vf8-f_eq8);
            f9c=vf9-omega*(vf9-f_eq9);
            f10c=vf10-omega*(vf10-f_eq10);
            f11c=vf11-omega*(vf11-f_eq11);
            f12c=vf12-omega*(vf12-f_eq12);
            f13c=vf13-omega*(vf13-f_eq13);
            f14c=vf14-omega*(vf14-f_eq14);
            f15c=vf15-omega*(vf15-f_eq15);
            f16c=vf16-omega*(vf16-f_eq16);
            f17c=vf17-omega*(vf17-f_eq17);
            f18c=vf18-omega*(vf18-f_eq18);         
        }
        else{
            f0c=vf0;
            f1c=vf2;
            f2c=vf1;
            f3c=vf4;
            f4c=vf3;
            f5c=vf6;
            f6c=vf5;
            f7c=vf8;
            f8c=vf7;
            f9c=vf10;
            f10c=vf9;
            f11c=vf12;
            f12c=vf11;
            f13c=vf14;
            f14c=vf13;
            f15c=vf16;
            f16c=vf15;
            f17c=vf18;
            f18c=vf17;

        }

    /*  if(fabsf(ux_old_v[kk])>delta){//*uy_old_v[0])
        {
            f0c=f_eq0;
            f1c=f_eq1;
            f2c=f_eq2;
            f3c=f_eq3;
            f4c=f_eq4;
            f5c=f_eq5;
        //          f6c=f_eq6;
        //          f7c=f_eq7;
        //          f8c=f_eq8;
        //          f9c=f_eq9;
        }
        */



            

        f0p[kk]=f0c;
        f1p[kk]=f1c;
        f2p[kk]=f2c;
        f3p[kk]=f3c;
        f4p[kk]=f4c;
        f5p[kk]=f5c;
        f6p[kk]=f6c;
        f7p[kk]=f7c;
        f8p[kk]=f8c;
        f9p[kk]=f9c;
        f10p[kk]=f10c;
        f11p[kk]=f11c;
        f12p[kk]=f12c;
        f13p[kk]=f13c;
        f14p[kk]=f14c;
        f15p[kk]=f15c;
        f16p[kk]=f16c;
        f17p[kk]=f17c;
        f18p[kk]=f18c;

    }

}







__global__ void kernel_correction_0_18(real* ux_old_v, real* uy_old_v, int Nx, int Ny, int Nz, real *ux_v, real *uy_v, real *uz_v, real *ro_v, int* bc_v, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18)
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
        int kk=k+j*(Nz)+i*(Ny)*(Nz);



        real vf0=f0[I3(i,j,k)];
        real vf1=f1[I3(i,j,k)];
        real vf2=f2[I3(i,j,k)];
        real vf3=f3[I3(i,j,k)];
        real vf4=f4[I3(i,j,k)];
        real vf5=f5[I3(i,j,k)];
        real vf6=f6[I3(i,j,k)];
        real vf7=f7[I3(i,j,k)];
        real vf8=f8[I3(i,j,k)];
        real vf9=f9[I3(i,j,k)];
        real vf10=f10[I3(i,j,k)];
        real vf11=f11[I3(i,j,k)];
        real vf12=f12[I3(i,j,k)];
        real vf13=f13[I3(i,j,k)];
        real vf14=f14[I3(i,j,k)];
        real vf15=f15[I3(i,j,k)];
        real vf16=f16[I3(i,j,k)];
        real vf17=f17[I3(i,j,k)];
        real vf18=f18[I3(i,j,k)];

        real w[Q]={1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
        const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
        const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
        const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};

        real ro=ro_v[kk];
        real v_x=ux_v[kk];
        real v_y=uy_v[kk];
        real v_z=uz_v[kk];

        //Calculate equilibrium distribution
        real cs=1.0;//0.33333333333333333333333333333;
        real vv=0.0;
        
        real f_eq0=w[0]*ro*(1.0-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[1]*v_x+cy[1]*v_y+cz[1]*v_z;
        real f_eq1=w[1]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[2]*v_x+cy[2]*v_y+cz[2]*v_z;
        real f_eq2=w[2]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[3]*v_x+cy[3]*v_y+cz[3]*v_z;
        real f_eq3=w[3]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[4]*v_x+cy[4]*v_y+cz[4]*v_z;
        real f_eq4=w[4]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[5]*v_x+cy[5]*v_y+cz[5]*v_z;
        real f_eq5=w[5]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[6]*v_x+cy[6]*v_y+cz[6]*v_z;
        real f_eq6=w[6]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[7]*v_x+cy[7]*v_y+cz[7]*v_z;
        real f_eq7=w[7]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[8]*v_x+cy[8]*v_y+cz[8]*v_z;
        real f_eq8=w[8]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[9]*v_x+cy[9]*v_y+cz[9]*v_z;
        real f_eq9=w[9]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[10]*v_x+cy[10]*v_y+cz[10]*v_z;
        real f_eq10=w[10]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[11]*v_x+cy[11]*v_y+cz[11]*v_z;
        real f_eq11=w[11]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[12]*v_x+cy[12]*v_y+cz[12]*v_z;
        real f_eq12=w[12]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[13]*v_x+cy[13]*v_y+cz[13]*v_z;
        real f_eq13=w[13]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[14]*v_x+cy[14]*v_y+cz[14]*v_z;
        real f_eq14=w[14]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[15]*v_x+cy[15]*v_y+cz[15]*v_z;
        real f_eq15=w[15]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);

        vv=cx[16]*v_x+cy[16]*v_y+cz[16]*v_z;
        real f_eq16=w[16]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[17]*v_x+cy[17]*v_y+cz[17]*v_z;
        real f_eq17=w[17]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
        
        vv=cx[18]*v_x+cy[18]*v_y+cz[18]*v_z;
        real f_eq18=w[18]*ro*(1.0+3.0*vv/(cs)+4.5*(vv*vv)/(cs*cs)-1.5*(v_x*v_x+v_y*v_y+v_z*v_z)/cs);
//Here i should put Kulbrak stabilization!!!xxx!!!
    

        real summ=vf0*__logf(fabsf(vf0)/f_eq0)+
            vf1*__logf(fabsf(vf1)/f_eq1)+
            vf2*__logf(fabsf(vf2)/f_eq2)+
            vf3*__logf(fabsf(vf3)/f_eq3)+
            vf4*__logf(fabsf(vf4)/f_eq4)+
            vf5*__logf(fabsf(vf5)/f_eq5)+
            vf6*__logf(fabsf(vf6)/f_eq6)+
            vf7*__logf(fabsf(vf7)/f_eq7)+
            vf8*__logf(fabsf(vf8)/f_eq8)+
            vf9*__logf(fabsf(vf9)/f_eq9)+
            vf10*__logf(fabsf(vf10)/f_eq10)+
            vf11*__logf(fabsf(vf11)/f_eq11)+
            vf12*__logf(fabsf(vf12)/f_eq12)+
            vf13*__logf(fabsf(vf13)/f_eq13)+
            vf14*__logf(fabsf(vf14)/f_eq14)+
            vf15*__logf(fabsf(vf15)/f_eq15)+
            vf16*__logf(fabsf(vf16)/f_eq16)+
            vf17*__logf(fabsf(vf17)/f_eq17)+
            vf18*__logf(fabsf(vf18)/f_eq18);

            
        ux_old_v[kk]=summ;
        uy_old_v[0]=0.0;
        if(uy_old_v[0]<fabsf(ux_old_v[kk])) 
        uy_old_v[0]=fabsf(ux_old_v[kk]);

    }
}

