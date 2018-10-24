/********************************************************************
*  sample.cu
*  This is a example of the CUDA program.
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
//#include <cutil.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h> //for timer support

#define real double


#define PI 3.1415926535897932384626433832795
const int Q=19;
const int G=6;
#define I4(i,j,k,q) ( (Q)*(i)+(Q)*(M)*(j)+(Q)*(N)*(M)*(k)+(q) )
//#define I3(i,j,k) ( (i)+(M)*(j)+(N)*(M)*(k) ) //(i-1)*n2*n3 + (j-1)*n3 + (k-1)
#define I3(i,j,k) ( (i)*(N)*(K)+(K)*(j)+(k) )
//#define I2(i,j) ( (i)+(M)*(j) )

#define I2(n,t) ((n)+(3)*(t))
#define I2p(p,n,t) ((p)+3*(n)+3*(NumP)*(t))  //No - number of points!


#define FLUID 0
#define WALL 1
#define IN 2
#define OUT 3
#define PERIODIC 4

#define Th 1.0
#define Tc 0.0


#ifndef I3P
    #define I3P(i,j,k)  (N)*(K)*((i)>(M-1)?(i)-(M):(i)<0?((M)+(i)):(i))+((j)>(N-1)?(j)-(N):(j)<0?(N+(j)):(j))*(K)+((k)>(K-1)?(k)-(K):(k)<0?(K+(k)):(k))
#endif


real w[Q]={1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
int opp[Q] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

real linesize=2.5;


real *f, *f_equ, *f_old;
real *ux, *uy, *uz, *ro, *rot, *rota, *s;
real *uxa,*uya,*uza;
real *uxi,*uyi,*uzi;
real *ux_rms, *uy_rms, *uz_rms;
real *k_ux, *k_uy, *k_uz;
real *lk1,*lk2,*lk3, *lk12, *lk21, *lk32, *lk23, *lk13, *lk31;
int *bc, *maxDS;
real *lk;

//Let's try this =)
real *f0, *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8, *f9, *f10, *f11, *f12, *f13, *f14, *f15, *f16, *f17, *f18;// NS1
real *f0p, *f1p, *f2p, *f3p, *f4p, *f5p, *f6p, *f7p, *f8p, *f9p, *f10p, *f11p, *f12p, *f13p, *f14p, *f15p, *f16p, *f17p, *f18p;// NS2


real *d0, *d1, *d2, *d3, *d4, *d5, *d6, *d7, *d8, *d9, *d10, *d11, *d12, *d13, *d14, *d15, *d16, *d17, *d18;// NS

real *ux_v, *uy_v, *ux_old_v, *uy_old_v, *uz_v, *ro_v, *ro_old_v;
int *bc_v, *maxDS_v;


real *uxi_v,*uyi_v,*uzi_v;
real *uxa_v,*uya_v,*uza_v;
real *uxa1_v,*uya1_v,*uza1_v;
real *lk1_v,*lk2_v,*lk3_v, *lk12_v, *lk21_v, *lk32_v, *lk23_v, *lk13_v, *lk31_v;
real *ux_rms_v, *uy_rms_v, *uz_rms_v;
real *k_ux_v, *k_uy_v, *k_uz_v;


//Point data CPU
real *Points;
real *Coords;

//Points data GPU

real *vPoints;
real *vCoords;

real *uinit;


//random normal distribution
real rand_normal(real mean, real stddev)
{//Box muller method
    static real n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        real x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            real d = sqrt(-2.0*log(r)/r);
            real n1 = x*d;
            n2 = y*d;
            real result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return real(n2*stddev + mean);
    }
}



/************************************************************************/
/* File Operations!                                                     */
/************************************************************************/


int read_point_data_file(real* co, int _M,int _N,int _K,char f_name[]){
FILE *stream;
int No;
double _x,_y,_z;

    stream=fopen( f_name, "r+" );
    fseek( stream, 0L, SEEK_SET );

    fscanf( stream, "%i", &No);
    
    co=new real[No*3+1];

    for(int n=0;n<No;n++){
                
        fscanf( stream, "%lf %lf %lf ", &_x,&_y,&_z);
        co[I2(0,n)]=real((real)_x*real(_M));
        co[I2(1,n)]=real((real)_y*real(_N));
        co[I2(2,n)]=real((real)_z*real(_K));

    }
            
        fclose(stream);
    

    return(No);
}





void write_out_points(char f_name[],int NumP,int timesteps){

    FILE *stream;


    stream=fopen( f_name, "w" );



    for(int t=0;t<timesteps;t++){
        fprintf( stream, "%i    ", t);
        
        for(int n=0;n<NumP;n++){
            
            fprintf( stream, "%.16le %.16le %.16le  ", (double)Points[I2p(0,n,t)],(double)Points[I2p(1,n,t)],(double)Points[I2p(2,n,t)]);
        }
    
        fprintf( stream, "\n");
    }



    fclose(stream);


}



void write_control_file(char f_name[], int M, int N, int K){

    FILE *stream;

    printf("\n");
    stream=fopen( f_name, "w" ); //"general_out.dat"


    

    for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
    for(int k=0;k<K;k++){


        
        fprintf( stream, "%.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le\n", 
                (double)d0[I3(i,j,k)], (double)d1[I3(i,j,k)], (double)d2[I3(i,j,k)], (double)d3[I3(i,j,k)], (double)d4[I3(i,j,k)], 
                (double)d5[I3(i,j,k)], (double)d6[I3(i,j,k)], (double)d7[I3(i,j,k)],(double)d8[I3(i,j,k)], (double)d9[I3(i,j,k)],
                (double)d10[I3(i,j,k)], (double)d11[I3(i,j,k)], (double)d12[I3(i,j,k)], (double)d13[I3(i,j,k)], (double)d14[I3(i,j,k)], 
                (double)d15[I3(i,j,k)], (double)d16[I3(i,j,k)], (double)d17[I3(i,j,k)], (double)d18[I3(i,j,k)]);

        
        }   
        }
        printf("general_out.dat [%.03f%%]   \r",100.0f*real(i)/real(M-1) );
    }
    

        fclose(stream);
    

}




void write_avaraged_file(int M, int N, int K){

    FILE *stream;


    stream=fopen( "avaraged_out.dat", "w" );


    

    for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
    for(int k=0;k<K;k++){


        
        fprintf( stream, "%.16le %.16le %.16le\n", (double)uxa[I3(i,j,k)], (double)uya[I3(i,j,k)],(double)uza[I3(i,j,k)]);
                

        
        }   
        }
        printf("avaraged_out.dat [%.03f%%]  \r",100.0f*real(i)/real(M-1) );
    }
    

        fclose(stream);
    

}



void read_control_file(int M, int N, int K){

    FILE *stream;

    printf("\n");
    stream=fopen( "general_out.dat", "r+" );
    fseek( stream, 0L, SEEK_SET );

    double d0c, d1c, d2c, d3c, d4c, d5c, d6c, d7c, d8c, d9c, d10c, d11c, d12c, d13c, d14c, d15c, d16c, d17c, d18c;


    
    for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
    for(int k=0;k<K;k++){
                
            
            fscanf( stream, "%le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le", &d0c, &d1c, &d2c, &d3c, &d4c, &d5c, &d6c, &d7c, &d8c, &d9c, &d10c, &d11c, &d12c, &d13c, &d14c, &d15c, &d16c, &d17c, &d18c);

            d0[I3(i,j,k)]=(real)d0c;
            d1[I3(i,j,k)]=(real)d1c;
            d2[I3(i,j,k)]=(real)d2c;
            d3[I3(i,j,k)]=(real)d3c;
            d4[I3(i,j,k)]=(real)d4c;
            d5[I3(i,j,k)]=(real)d5c;
            d6[I3(i,j,k)]=(real)d6c;
            d7[I3(i,j,k)]=(real)d7c;
            d8[I3(i,j,k)]=(real)d8c;
            d9[I3(i,j,k)]=(real)d9c;
            d10[I3(i,j,k)]=(real)d10c;
            d11[I3(i,j,k)]=(real)d11c;
            d12[I3(i,j,k)]=(real)d12c;
            d13[I3(i,j,k)]=(real)d13c; 
            d14[I3(i,j,k)]=(real)d14c;
            d15[I3(i,j,k)]=(real)d15c;
            d16[I3(i,j,k)]=(real)d16c;
            d17[I3(i,j,k)]=(real)d17c;
            d18[I3(i,j,k)]=(real)d18c;
            

        }   
        }
        printf("general_out.dat [%.03f%%]   \r",100.0f*real(i)/real(M-1) );
    }

        fclose(stream);

}


void read_avaraged_file(int M, int N, int K){

    FILE *stream;


    stream=fopen( "avaraged_out.dat", "r+" );
    fseek( stream, 0L, SEEK_SET );
    double uxac, uyac, uzac;

    
    for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
    for(int k=0;k<K;k++){
                
            
            fscanf( stream, "%le %le %le ", &uxac, &uyac, &uzac);

             uxa[I3(i,j,k)]=(real)uxac;
             uya[I3(i,j,k)]=(real)uyac;
             uza[I3(i,j,k)]=(real)uzac;

        }   

        }
        printf("avaraged_out.dat [%.03f%%]  \r",100.0f*real(i)/real(M-1) );
        
    }

        fclose(stream);
    

}


void write_out_file_1ord(char f_name[], int M, int N, int K, real dh, char what){

    FILE *stream;
    stream=fopen( f_name, "w" );


    fprintf( stream, "View");
    fprintf( stream, " '");
    fprintf( stream, f_name);
    fprintf( stream, "' {\n");
    fprintf( stream, "TIME{0};\n");
    for(int i=0;i<M;i++)
    for(int j=0;j<N;j++)
    for(int k=0;k<K;k++){

        fprintf( stream, "SH(");
            
        fprintf( stream,"%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf",
        dh*i-0.5*dh,dh*j-0.5*dh,dh*k-0.5*dh, 
        dh*i+0.5*dh,dh*j-0.5*dh,dh*k-0.5*dh,
        dh*i+0.5*dh,dh*j+0.5*dh,dh*k-0.5*dh,
        dh*i-0.5*dh,dh*j+0.5*dh,dh*k-0.5*dh,
        dh*i-0.5*dh,dh*j-0.5*dh,dh*k+0.5*dh,
        dh*i+0.5*dh,dh*j-0.5*dh,dh*k+0.5*dh,
        dh*i+0.5*dh,dh*j+0.5*dh,dh*k+0.5*dh,
        dh*i-0.5*dh,dh*j+0.5*dh,dh*k+0.5*dh);
        
        real par=0.0;
        if(what=='B'){
            par=(real)bc[I3(i,j,k)]*1.0f;

        }


        fprintf( stream, "){");
        fprintf( stream,"%le, %le, %le, %le, %le, %le, %le, %le};\n",par ,par,par,par,par ,par,par,par);

        }
    fprintf( stream, "};");

    fclose(stream);


}




void write_out_file(char f_name[], int M, int N, int K, char type, real dh, real xm, real ym, real zm, int flagi, int flagj, int flagk){

        FILE *stream;
        
        int what=2;

        real dx=linesize/real(M);
        real dy=dx;//linesize/real(N);
        real dz=dx;//linesize/real(K);

        stream=fopen( f_name, "w" );


        fprintf( stream, "View");
        fprintf( stream, " '");
        fprintf( stream, f_name);
        fprintf( stream, "' {\n");
        fprintf( stream, "TIME{0};\n");
    
        real k_scale=10.0;

    if((type=='v')||(type=='a')||(type=='i')){
        
        real *ux_l=ux;
        real *uy_l=uy;
        real *uz_l=uz;
        if(type=='a'){
            ux_l=uxa;
            uy_l=uya;
            uz_l=uza;
            k_scale=1.0;

        }
        else if(type=='i'){
            ux_l=uxi;
            uy_l=uyi;
            uz_l=uzi;
            k_scale=1.0;

        }

        for(int j=1;j<M-1;j++)
        for(int k=1;k<N-1;k++)
        for(int l=1;l<K-1;l++)
        if(bc[I3(j,k,l)]!=WALL){
            real par_x=0.0,par_y=0.0,par_z=0.0;
            real par_x_mmm=0.0;
            real par_x_pmm=0.0;
            real par_x_ppm=0.0;
            real par_x_ppp=0.0;
            real par_x_mpp=0.0;
            real par_x_mmp=0.0;
            real par_x_pmp=0.0;
            real par_x_mpm=0.0;
            real par_y_mmm=0.0;
            real par_y_pmm=0.0;
            real par_y_ppm=0.0;
            real par_y_ppp=0.0;
            real par_y_mpp=0.0;
            real par_y_mmp=0.0;
            real par_y_pmp=0.0;
            real par_y_mpm=0.0;
            real par_z_mmm=0.0;
            real par_z_pmm=0.0;
            real par_z_ppm=0.0;
            real par_z_ppp=0.0;
            real par_z_mpp=0.0;
            real par_z_mmp=0.0;
            real par_z_pmp=0.0;
            real par_z_mpm=0.0;

            
            par_x=ux_l[I3(j,k,l)];
            par_y=uy_l[I3(j,k,l)];
            par_z=uz_l[I3(j,k,l)];
            par_x_mmm=0.125f*(ux_l[I3(j,k,l)]+ux_l[I3(j-1,k,l)]+ux_l[I3(j,k-1,l)]+ux_l[I3(j,k,l-1)]+ux_l[I3(j-1,k-1,l)]+ux_l[I3(j,k-1,l-1)]+ux_l[I3(j-1,k,l-1)]+ux_l[I3(j-1,k-1,l-1)]);
            par_x_pmm=0.125f*(ux_l[I3(j,k,l)]+ux_l[I3(j+1,k,l)]+ux_l[I3(j,k-1,l)]+ux_l[I3(j,k,l-1)]+ux_l[I3(j+1,k-1,l)]+ux_l[I3(j,k-1,l-1)]+ux_l[I3(j+1,k,l-1)]+ux_l[I3(j+1,k-1,l-1)]);
            par_x_ppm=0.125f*(ux_l[I3(j,k,l)]+ux_l[I3(j+1,k,l)]+ux_l[I3(j,k+1,l)]+ux_l[I3(j,k,l-1)]+ux_l[I3(j+1,k+1,l)]+ux_l[I3(j,k+1,l-1)]+ux_l[I3(j+1,k,l-1)]+ux_l[I3(j+1,k+1,l-1)]);
            par_x_ppp=0.125f*(ux_l[I3(j,k,l)]+ux_l[I3(j+1,k,l)]+ux_l[I3(j,k+1,l)]+ux_l[I3(j,k,l+1)]+ux_l[I3(j+1,k+1,l)]+ux_l[I3(j,k+1,l+1)]+ux_l[I3(j+1,k,l+1)]+ux_l[I3(j+1,k+1,l+1)]);
            par_x_mpp=0.125f*(ux_l[I3(j,k,l)]+ux_l[I3(j-1,k,l)]+ux_l[I3(j,k+1,l)]+ux_l[I3(j,k,l+1)]+ux_l[I3(j-1,k+1,l)]+ux_l[I3(j,k+1,l+1)]+ux_l[I3(j-1,k,l+1)]+ux_l[I3(j-1,k+1,l+1)]);
            par_x_mmp=0.125f*(ux_l[I3(j,k,l)]+ux_l[I3(j-1,k,l)]+ux_l[I3(j,k-1,l)]+ux_l[I3(j,k,l+1)]+ux_l[I3(j-1,k-1,l)]+ux_l[I3(j,k-1,l+1)]+ux_l[I3(j-1,k,l+1)]+ux_l[I3(j-1,k-1,l+1)]);
            par_x_pmp=0.125f*(ux_l[I3(j,k,l)]+ux_l[I3(j+1,k,l)]+ux_l[I3(j,k-1,l)]+ux_l[I3(j,k,l+1)]+ux_l[I3(j+1,k-1,l)]+ux_l[I3(j,k-1,l+1)]+ux_l[I3(j+1,k,l+1)]+ux_l[I3(j+1,k-1,l+1)]);
            par_x_mpm=0.125f*(ux_l[I3(j,k,l)]+ux_l[I3(j-1,k,l)]+ux_l[I3(j,k+1,l)]+ux_l[I3(j,k,l-1)]+ux_l[I3(j-1,k+1,l)]+ux_l[I3(j,k+1,l-1)]+ux_l[I3(j-1,k,l-1)]+ux_l[I3(j-1,k+1,l-1)]);
            
            par_y_mmm=0.125f*(uy_l[I3(j,k,l)]+uy_l[I3(j-1,k,l)]+uy_l[I3(j,k-1,l)]+uy_l[I3(j,k,l-1)]+uy_l[I3(j-1,k-1,l)]+uy_l[I3(j,k-1,l-1)]+uy_l[I3(j-1,k,l-1)]+uy_l[I3(j-1,k-1,l-1)]);
            par_y_pmm=0.125f*(uy_l[I3(j,k,l)]+uy_l[I3(j+1,k,l)]+uy_l[I3(j,k-1,l)]+uy_l[I3(j,k,l-1)]+uy_l[I3(j+1,k-1,l)]+uy_l[I3(j,k-1,l-1)]+uy_l[I3(j+1,k,l-1)]+uy_l[I3(j+1,k-1,l-1)]);
            par_y_ppm=0.125f*(uy_l[I3(j,k,l)]+uy_l[I3(j+1,k,l)]+uy_l[I3(j,k+1,l)]+uy_l[I3(j,k,l-1)]+uy_l[I3(j+1,k+1,l)]+uy_l[I3(j,k+1,l-1)]+uy_l[I3(j+1,k,l-1)]+uy_l[I3(j+1,k+1,l-1)]);
            par_y_ppp=0.125f*(uy_l[I3(j,k,l)]+uy_l[I3(j+1,k,l)]+uy_l[I3(j,k+1,l)]+uy_l[I3(j,k,l+1)]+uy_l[I3(j+1,k+1,l)]+uy_l[I3(j,k+1,l+1)]+uy_l[I3(j+1,k,l+1)]+uy_l[I3(j+1,k+1,l+1)]);
            par_y_mpp=0.125f*(uy_l[I3(j,k,l)]+uy_l[I3(j-1,k,l)]+uy_l[I3(j,k+1,l)]+uy_l[I3(j,k,l+1)]+uy_l[I3(j-1,k+1,l)]+uy_l[I3(j,k+1,l+1)]+uy_l[I3(j-1,k,l+1)]+uy_l[I3(j-1,k+1,l+1)]);
            par_y_mmp=0.125f*(uy_l[I3(j,k,l)]+uy_l[I3(j-1,k,l)]+uy_l[I3(j,k-1,l)]+uy_l[I3(j,k,l+1)]+uy_l[I3(j-1,k-1,l)]+uy_l[I3(j,k-1,l+1)]+uy_l[I3(j-1,k,l+1)]+uy_l[I3(j-1,k-1,l+1)]);
            par_y_pmp=0.125f*(uy_l[I3(j,k,l)]+uy_l[I3(j+1,k,l)]+uy_l[I3(j,k-1,l)]+uy_l[I3(j,k,l+1)]+uy_l[I3(j+1,k-1,l)]+uy_l[I3(j,k-1,l+1)]+uy_l[I3(j+1,k,l+1)]+uy_l[I3(j+1,k-1,l+1)]);
            par_y_mpm=0.125f*(uy_l[I3(j,k,l)]+uy_l[I3(j-1,k,l)]+uy_l[I3(j,k+1,l)]+uy_l[I3(j,k,l-1)]+uy_l[I3(j-1,k+1,l)]+uy_l[I3(j,k+1,l-1)]+uy_l[I3(j-1,k,l-1)]+uy_l[I3(j-1,k+1,l-1)]);
            
                    
            par_z_mmm=0.125f*(uz_l[I3(j,k,l)]+uz_l[I3(j-1,k,l)]+uz_l[I3(j,k-1,l)]+uz_l[I3(j,k,l-1)]+uz_l[I3(j-1,k-1,l)]+uz_l[I3(j,k-1,l-1)]+uz_l[I3(j-1,k,l-1)]+uz_l[I3(j-1,k-1,l-1)]);
            par_z_pmm=0.125f*(uz_l[I3(j,k,l)]+uz_l[I3(j+1,k,l)]+uz_l[I3(j,k-1,l)]+uz_l[I3(j,k,l-1)]+uz_l[I3(j+1,k-1,l)]+uz_l[I3(j,k-1,l-1)]+uz_l[I3(j+1,k,l-1)]+uz_l[I3(j+1,k-1,l-1)]);
            par_z_ppm=0.125f*(uz_l[I3(j,k,l)]+uz_l[I3(j+1,k,l)]+uz_l[I3(j,k+1,l)]+uz_l[I3(j,k,l-1)]+uz_l[I3(j+1,k+1,l)]+uz_l[I3(j,k+1,l-1)]+uz_l[I3(j+1,k,l-1)]+uz_l[I3(j+1,k+1,l-1)]);
            par_z_ppp=0.125f*(uz_l[I3(j,k,l)]+uz_l[I3(j+1,k,l)]+uz_l[I3(j,k+1,l)]+uz_l[I3(j,k,l+1)]+uz_l[I3(j+1,k+1,l)]+uz_l[I3(j,k+1,l+1)]+uz_l[I3(j+1,k,l+1)]+uz_l[I3(j+1,k+1,l+1)]);
            par_z_mpp=0.125f*(uz_l[I3(j,k,l)]+uz_l[I3(j-1,k,l)]+uz_l[I3(j,k+1,l)]+uz_l[I3(j,k,l+1)]+uz_l[I3(j-1,k+1,l)]+uz_l[I3(j,k+1,l+1)]+uz_l[I3(j-1,k,l+1)]+uz_l[I3(j-1,k+1,l+1)]);
            par_z_mmp=0.125f*(uz_l[I3(j,k,l)]+uz_l[I3(j-1,k,l)]+uz_l[I3(j,k-1,l)]+uz_l[I3(j,k,l+1)]+uz_l[I3(j-1,k-1,l)]+uz_l[I3(j,k-1,l+1)]+uz_l[I3(j-1,k,l+1)]+uz_l[I3(j-1,k-1,l+1)]);
            par_z_pmp=0.125f*(uz_l[I3(j,k,l)]+uz_l[I3(j+1,k,l)]+uz_l[I3(j,k-1,l)]+uz_l[I3(j,k,l+1)]+uz_l[I3(j+1,k-1,l)]+uz_l[I3(j,k-1,l+1)]+uz_l[I3(j+1,k,l+1)]+uz_l[I3(j+1,k-1,l+1)]);
            par_z_mpm=0.125f*(uz_l[I3(j,k,l)]+uz_l[I3(j-1,k,l)]+uz_l[I3(j,k+1,l)]+uz_l[I3(j,k,l-1)]+uz_l[I3(j-1,k+1,l)]+uz_l[I3(j,k+1,l-1)]+uz_l[I3(j-1,k,l-1)]+uz_l[I3(j-1,k+1,l-1)]);

            dx=1.0;
            dy=1.0;
            dz=1.0;

            fprintf( stream, "VH(%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf)",
                    xm+dx*j-0.5*dx, ym+dy*k-0.5*dy, zm+dz*l-0.5*dz, 
                    xm+dx*j+0.5*dx, ym+dy*k-0.5*dy, zm+dz*l-0.5*dz, 
                    xm+dx*j+0.5*dx, ym+dy*k+0.5*dy, zm+dz*l-0.5*dz,
                    xm+dx*j+0.5*dx, ym+dy*k+0.5*dy, zm+dz*l+0.5*dz,
                    xm+dx*j-0.5*dx, ym+dy*k+0.5*dy, zm+dz*l+0.5*dz, 
                    xm+dx*j-0.5*dx, ym+dy*k-0.5*dy, zm+dz*l+0.5*dz, 
                    xm+dx*j+0.5*dx, ym+dy*k-0.5*dy, zm+dz*l+0.5*dz,
                    xm+dx*j-0.5*dx, ym+dy*k+0.5*dy, zm+dz*l-0.5*dz);

            if(what==2){
                fprintf( stream,"{");
                fprintf(stream, "%le,%le,%le,",par_x_mmm,par_y_mmm,par_z_mmm);
                fprintf(stream, "%le,%le,%le,",par_x_pmm,par_y_pmm,par_z_pmm);
                fprintf(stream, "%le,%le,%le,",par_x_ppm,par_y_ppm,par_z_ppm);
                fprintf(stream, "%le,%le,%le,",par_x_ppp,par_y_ppp,par_z_ppp);
                fprintf(stream, "%le,%le,%le,",par_x_mpp,par_y_mpp,par_z_mpp);
                fprintf(stream, "%le,%le,%le,",par_x_mmp,par_y_mmp,par_z_mmp);
                fprintf(stream, "%le,%le,%le,",par_x_pmp,par_y_pmp,par_z_pmp);
                fprintf(stream, "%le,%le,%le",par_x_mpm,par_y_mpm,par_z_mpm);
                fprintf(stream, "};\n");
            }
            else if(what==1){
                fprintf( stream,"{");
                fprintf(stream, "%le,%le,%le,",par_x*k_scale,par_y*k_scale,par_z*k_scale);
                fprintf(stream, "%le,%le,%le,",par_x*k_scale,par_y*k_scale,par_z*k_scale);
                fprintf(stream, "%le,%le,%le,",par_x*k_scale,par_y*k_scale,par_z*k_scale);
                fprintf(stream, "%le,%le,%le,",par_x*k_scale,par_y*k_scale,par_z*k_scale);
                fprintf(stream, "%le,%le,%le,",par_x*k_scale,par_y*k_scale,par_z*k_scale);
                fprintf(stream, "%le,%le,%le,",par_x*k_scale,par_y*k_scale,par_z*k_scale);
                fprintf(stream, "%le,%le,%le,",par_x*k_scale,par_y*k_scale,par_z*k_scale);
                fprintf(stream, "%le,%le,%le" ,par_x*k_scale,par_y*k_scale,par_z*k_scale);
                fprintf(stream, "};\n");
            }


        }   
        
        
    } //for ends here!

    else{
    real *uiso;


    if(type=='p'){
        uiso=new real[(M)*(N)*(K)];

        for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
        for(int k=0;k<K;k++)
            uiso[I3(i,j,k)]=sqrt(ux[I3(i,j,k)]*ux[I3(i,j,k)]+uy[I3(i,j,k)]*uy[I3(i,j,k)]+uz[I3(i,j,k)]*uz[I3(i,j,k)])*10.0;

    }
    if(type=='u'){
        uiso=new real[(M)*(N)*(K)];
        for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
        for(int k=0;k<K;k++)
            uiso[I3(i,j,k)]=sqrt(uxa[I3(i,j,k)]*uxa[I3(i,j,k)]+uya[I3(i,j,k)]*uya[I3(i,j,k)]+uza[I3(i,j,k)]*uza[I3(i,j,k)]);

    }


    for(int i=1;i<M-1;i++)
    for(int j=1;j<N-1;j++)
    for(int k=1;k<K-1;k++)
        if(bc[I3(i,j,k)]!=WALL)
        if((i % flagi== 0)||(i==1)||(i==M-1))
        if((j % flagj== 0)||(j==1)||(j==N-1))
        if((k % flagk== 0)||(k==1)||(k==K-1)){

        real par=0.0;
        real par_mmm=0.0;
        real par_pmm=0.0;
        real par_ppm=0.0;
        real par_ppp=0.0;
        real par_mpp=0.0;
        real par_mmp=0.0;
        real par_pmp=0.0;
        real par_mpm=0.0;

        if(type=='l'){
            par_mmm=0.125f*(lk[I3(i,j,k)]+lk[I3(i-1,j,k)]+lk[I3(i,j-1,k)]+lk[I3(i,j,k-1)]+lk[I3(i-1,j-1,k)]+lk[I3(i,j-1,k-1)]+lk[I3(i-1,j,k-1)]+lk[I3(i-1,j-1,k-1)]);
            par_pmm=0.125f*(lk[I3(i,j,k)]+lk[I3(i+1,j,k)]+lk[I3(i,j-1,k)]+lk[I3(i,j,k-1)]+lk[I3(i+1,j-1,k)]+lk[I3(i,j-1,k-1)]+lk[I3(i+1,j,k-1)]+lk[I3(i+1,j-1,k-1)]);
            par_ppm=0.125f*(lk[I3(i,j,k)]+lk[I3(i+1,j,k)]+lk[I3(i,j+1,k)]+lk[I3(i,j,k-1)]+lk[I3(i+1,j+1,k)]+lk[I3(i,j+1,k-1)]+lk[I3(i+1,j,k-1)]+lk[I3(i+1,j+1,k-1)]);
            par_ppp=0.125f*(lk[I3(i,j,k)]+lk[I3(i+1,j,k)]+lk[I3(i,j+1,k)]+lk[I3(i,j,k+1)]+lk[I3(i+1,j+1,k)]+lk[I3(i,j+1,k+1)]+lk[I3(i+1,j,k+1)]+lk[I3(i+1,j+1,k+1)]);
            par_mpp=0.125f*(lk[I3(i,j,k)]+lk[I3(i-1,j,k)]+lk[I3(i,j+1,k)]+lk[I3(i,j,k+1)]+lk[I3(i-1,j+1,k)]+lk[I3(i,j+1,k+1)]+lk[I3(i-1,j,k+1)]+lk[I3(i-1,j+1,k+1)]);
            par_mmp=0.125f*(lk[I3(i,j,k)]+lk[I3(i-1,j,k)]+lk[I3(i,j-1,k)]+lk[I3(i,j,k+1)]+lk[I3(i-1,j-1,k)]+lk[I3(i,j-1,k+1)]+lk[I3(i-1,j,k+1)]+lk[I3(i-1,j-1,k+1)]);
            par_pmp=0.125f*(lk[I3(i,j,k)]+lk[I3(i+1,j,k)]+lk[I3(i,j-1,k)]+lk[I3(i,j,k+1)]+lk[I3(i+1,j-1,k)]+lk[I3(i,j-1,k+1)]+lk[I3(i+1,j,k+1)]+lk[I3(i+1,j-1,k+1)]);
            par_mpm=0.125f*(lk[I3(i,j,k)]+lk[I3(i-1,j,k)]+lk[I3(i,j+1,k)]+lk[I3(i,j,k-1)]+lk[I3(i-1,j+1,k)]+lk[I3(i,j+1,k-1)]+lk[I3(i-1,j,k-1)]+lk[I3(i-1,j+1,k-1)]);


        }
        if(type=='r'){
            par_mmm=0.125f*(ro[I3(i,j,k)]+ro[I3(i-1,j,k)]+ro[I3(i,j-1,k)]+ro[I3(i,j,k-1)]+ro[I3(i-1,j-1,k)]+ro[I3(i,j-1,k-1)]+ro[I3(i-1,j,k-1)]+ro[I3(i-1,j-1,k-1)]);
            par_pmm=0.125f*(ro[I3(i,j,k)]+ro[I3(i+1,j,k)]+ro[I3(i,j-1,k)]+ro[I3(i,j,k-1)]+ro[I3(i+1,j-1,k)]+ro[I3(i,j-1,k-1)]+ro[I3(i+1,j,k-1)]+ro[I3(i+1,j-1,k-1)]);
            par_ppm=0.125f*(ro[I3(i,j,k)]+ro[I3(i+1,j,k)]+ro[I3(i,j+1,k)]+ro[I3(i,j,k-1)]+ro[I3(i+1,j+1,k)]+ro[I3(i,j+1,k-1)]+ro[I3(i+1,j,k-1)]+ro[I3(i+1,j+1,k-1)]);
            par_ppp=0.125f*(ro[I3(i,j,k)]+ro[I3(i+1,j,k)]+ro[I3(i,j+1,k)]+ro[I3(i,j,k+1)]+ro[I3(i+1,j+1,k)]+ro[I3(i,j+1,k+1)]+ro[I3(i+1,j,k+1)]+ro[I3(i+1,j+1,k+1)]);
            par_mpp=0.125f*(ro[I3(i,j,k)]+ro[I3(i-1,j,k)]+ro[I3(i,j+1,k)]+ro[I3(i,j,k+1)]+ro[I3(i-1,j+1,k)]+ro[I3(i,j+1,k+1)]+ro[I3(i-1,j,k+1)]+ro[I3(i-1,j+1,k+1)]);
            par_mmp=0.125f*(ro[I3(i,j,k)]+ro[I3(i-1,j,k)]+ro[I3(i,j-1,k)]+ro[I3(i,j,k+1)]+ro[I3(i-1,j-1,k)]+ro[I3(i,j-1,k+1)]+ro[I3(i-1,j,k+1)]+ro[I3(i-1,j-1,k+1)]);
            par_pmp=0.125f*(ro[I3(i,j,k)]+ro[I3(i+1,j,k)]+ro[I3(i,j-1,k)]+ro[I3(i,j,k+1)]+ro[I3(i+1,j-1,k)]+ro[I3(i,j-1,k+1)]+ro[I3(i+1,j,k+1)]+ro[I3(i+1,j-1,k+1)]);
            par_mpm=0.125f*(ro[I3(i,j,k)]+ro[I3(i-1,j,k)]+ro[I3(i,j+1,k)]+ro[I3(i,j,k-1)]+ro[I3(i-1,j+1,k)]+ro[I3(i,j+1,k-1)]+ro[I3(i-1,j,k-1)]+ro[I3(i-1,j+1,k-1)]);


        }
        if(type=='e'){
        //  par=rot[I3(i,j,k)];
            par_mmm=0.125f*(rot[I3(i,j,k)]+rot[I3(i-1,j,k)]+rot[I3(i,j-1,k)]+rot[I3(i,j,k-1)]+rot[I3(i-1,j-1,k)]+rot[I3(i,j-1,k-1)]+rot[I3(i-1,j,k-1)]+rot[I3(i-1,j-1,k-1)]);
            par_pmm=0.125f*(rot[I3(i,j,k)]+rot[I3(i+1,j,k)]+rot[I3(i,j-1,k)]+rot[I3(i,j,k-1)]+rot[I3(i+1,j-1,k)]+rot[I3(i,j-1,k-1)]+rot[I3(i+1,j,k-1)]+rot[I3(i+1,j-1,k-1)]);
            par_ppm=0.125f*(rot[I3(i,j,k)]+rot[I3(i+1,j,k)]+rot[I3(i,j+1,k)]+rot[I3(i,j,k-1)]+rot[I3(i+1,j+1,k)]+rot[I3(i,j+1,k-1)]+rot[I3(i+1,j,k-1)]+rot[I3(i+1,j+1,k-1)]);
            par_ppp=0.125f*(rot[I3(i,j,k)]+rot[I3(i+1,j,k)]+rot[I3(i,j+1,k)]+rot[I3(i,j,k+1)]+rot[I3(i+1,j+1,k)]+rot[I3(i,j+1,k+1)]+rot[I3(i+1,j,k+1)]+rot[I3(i+1,j+1,k+1)]);
            par_mpp=0.125f*(rot[I3(i,j,k)]+rot[I3(i-1,j,k)]+rot[I3(i,j+1,k)]+rot[I3(i,j,k+1)]+rot[I3(i-1,j+1,k)]+rot[I3(i,j+1,k+1)]+rot[I3(i-1,j,k+1)]+rot[I3(i-1,j+1,k+1)]);
            par_mmp=0.125f*(rot[I3(i,j,k)]+rot[I3(i-1,j,k)]+rot[I3(i,j-1,k)]+rot[I3(i,j,k+1)]+rot[I3(i-1,j-1,k)]+rot[I3(i,j-1,k+1)]+rot[I3(i-1,j,k+1)]+rot[I3(i-1,j-1,k+1)]);
            par_pmp=0.125f*(rot[I3(i,j,k)]+rot[I3(i+1,j,k)]+rot[I3(i,j-1,k)]+rot[I3(i,j,k+1)]+rot[I3(i+1,j-1,k)]+rot[I3(i,j-1,k+1)]+rot[I3(i+1,j,k+1)]+rot[I3(i+1,j-1,k+1)]);
            par_mpm=0.125f*(rot[I3(i,j,k)]+rot[I3(i-1,j,k)]+rot[I3(i,j+1,k)]+rot[I3(i,j,k-1)]+rot[I3(i-1,j+1,k)]+rot[I3(i,j+1,k-1)]+rot[I3(i-1,j,k-1)]+rot[I3(i-1,j+1,k-1)]);

        }
        if(type=='q'){
        //  par=rot[I3(i,j,k)];
            par_mmm=0.125f*(rota[I3(i,j,k)]+rota[I3(i-1,j,k)]+rota[I3(i,j-1,k)]+rota[I3(i,j,k-1)]+rota[I3(i-1,j-1,k)]+rota[I3(i,j-1,k-1)]+rota[I3(i-1,j,k-1)]+rota[I3(i-1,j-1,k-1)]);
            par_pmm=0.125f*(rota[I3(i,j,k)]+rota[I3(i+1,j,k)]+rota[I3(i,j-1,k)]+rota[I3(i,j,k-1)]+rota[I3(i+1,j-1,k)]+rota[I3(i,j-1,k-1)]+rota[I3(i+1,j,k-1)]+rota[I3(i+1,j-1,k-1)]);
            par_ppm=0.125f*(rota[I3(i,j,k)]+rota[I3(i+1,j,k)]+rota[I3(i,j+1,k)]+rota[I3(i,j,k-1)]+rota[I3(i+1,j+1,k)]+rota[I3(i,j+1,k-1)]+rota[I3(i+1,j,k-1)]+rota[I3(i+1,j+1,k-1)]);
            par_ppp=0.125f*(rota[I3(i,j,k)]+rota[I3(i+1,j,k)]+rota[I3(i,j+1,k)]+rota[I3(i,j,k+1)]+rota[I3(i+1,j+1,k)]+rota[I3(i,j+1,k+1)]+rota[I3(i+1,j,k+1)]+rota[I3(i+1,j+1,k+1)]);
            par_mpp=0.125f*(rota[I3(i,j,k)]+rota[I3(i-1,j,k)]+rota[I3(i,j+1,k)]+rota[I3(i,j,k+1)]+rota[I3(i-1,j+1,k)]+rota[I3(i,j+1,k+1)]+rota[I3(i-1,j,k+1)]+rota[I3(i-1,j+1,k+1)]);
            par_mmp=0.125f*(rota[I3(i,j,k)]+rota[I3(i-1,j,k)]+rota[I3(i,j-1,k)]+rota[I3(i,j,k+1)]+rota[I3(i-1,j-1,k)]+rota[I3(i,j-1,k+1)]+rota[I3(i-1,j,k+1)]+rota[I3(i-1,j-1,k+1)]);
            par_pmp=0.125f*(rota[I3(i,j,k)]+rota[I3(i+1,j,k)]+rota[I3(i,j-1,k)]+rota[I3(i,j,k+1)]+rota[I3(i+1,j-1,k)]+rota[I3(i,j-1,k+1)]+rota[I3(i+1,j,k+1)]+rota[I3(i+1,j-1,k+1)]);
            par_mpm=0.125f*(rota[I3(i,j,k)]+rota[I3(i-1,j,k)]+rota[I3(i,j+1,k)]+rota[I3(i,j,k-1)]+rota[I3(i-1,j+1,k)]+rota[I3(i,j+1,k-1)]+rota[I3(i-1,j,k-1)]+rota[I3(i-1,j+1,k-1)]);

        }

        if((type=='p')||(type=='u')){
            par_mmm=0.125f*(uiso[I3(i,j,k)]+uiso[I3(i-1,j,k)]+uiso[I3(i,j-1,k)]+uiso[I3(i,j,k-1)]+uiso[I3(i-1,j-1,k)]+uiso[I3(i,j-1,k-1)]+uiso[I3(i-1,j,k-1)]+uiso[I3(i-1,j-1,k-1)]);
            par_pmm=0.125f*(uiso[I3(i,j,k)]+uiso[I3(i+1,j,k)]+uiso[I3(i,j-1,k)]+uiso[I3(i,j,k-1)]+uiso[I3(i+1,j-1,k)]+uiso[I3(i,j-1,k-1)]+uiso[I3(i+1,j,k-1)]+uiso[I3(i+1,j-1,k-1)]);
            par_ppm=0.125f*(uiso[I3(i,j,k)]+uiso[I3(i+1,j,k)]+uiso[I3(i,j+1,k)]+uiso[I3(i,j,k-1)]+uiso[I3(i+1,j+1,k)]+uiso[I3(i,j+1,k-1)]+uiso[I3(i+1,j,k-1)]+uiso[I3(i+1,j+1,k-1)]);
            par_ppp=0.125f*(uiso[I3(i,j,k)]+uiso[I3(i+1,j,k)]+uiso[I3(i,j+1,k)]+uiso[I3(i,j,k+1)]+uiso[I3(i+1,j+1,k)]+uiso[I3(i,j+1,k+1)]+uiso[I3(i+1,j,k+1)]+uiso[I3(i+1,j+1,k+1)]);
            par_mpp=0.125f*(uiso[I3(i,j,k)]+uiso[I3(i-1,j,k)]+uiso[I3(i,j+1,k)]+uiso[I3(i,j,k+1)]+uiso[I3(i-1,j+1,k)]+uiso[I3(i,j+1,k+1)]+uiso[I3(i-1,j,k+1)]+uiso[I3(i-1,j+1,k+1)]);
            par_mmp=0.125f*(uiso[I3(i,j,k)]+uiso[I3(i-1,j,k)]+uiso[I3(i,j-1,k)]+uiso[I3(i,j,k+1)]+uiso[I3(i-1,j-1,k)]+uiso[I3(i,j-1,k+1)]+uiso[I3(i-1,j,k+1)]+uiso[I3(i-1,j-1,k+1)]);
            par_pmp=0.125f*(uiso[I3(i,j,k)]+uiso[I3(i+1,j,k)]+uiso[I3(i,j-1,k)]+uiso[I3(i,j,k+1)]+uiso[I3(i+1,j-1,k)]+uiso[I3(i,j-1,k+1)]+uiso[I3(i+1,j,k+1)]+uiso[I3(i+1,j-1,k+1)]);
            par_mpm=0.125f*(uiso[I3(i,j,k)]+uiso[I3(i-1,j,k)]+uiso[I3(i,j+1,k)]+uiso[I3(i,j,k-1)]+uiso[I3(i-1,j+1,k)]+uiso[I3(i,j+1,k-1)]+uiso[I3(i-1,j,k-1)]+uiso[I3(i-1,j+1,k-1)]);

        }
        if(type=='x'){
            par_mmm=0.125f*(ux[I3(i,j,k)]+ux[I3(i-1,j,k)]+ux[I3(i,j-1,k)]+ux[I3(i,j,k-1)]+ux[I3(i-1,j-1,k)]+ux[I3(i,j-1,k-1)]+ux[I3(i-1,j,k-1)]+ux[I3(i-1,j-1,k-1)]);
            par_pmm=0.125f*(ux[I3(i,j,k)]+ux[I3(i+1,j,k)]+ux[I3(i,j-1,k)]+ux[I3(i,j,k-1)]+ux[I3(i+1,j-1,k)]+ux[I3(i,j-1,k-1)]+ux[I3(i+1,j,k-1)]+ux[I3(i+1,j-1,k-1)]);
            par_ppm=0.125f*(ux[I3(i,j,k)]+ux[I3(i+1,j,k)]+ux[I3(i,j+1,k)]+ux[I3(i,j,k-1)]+ux[I3(i+1,j+1,k)]+ux[I3(i,j+1,k-1)]+ux[I3(i+1,j,k-1)]+ux[I3(i+1,j+1,k-1)]);
            par_ppp=0.125f*(ux[I3(i,j,k)]+ux[I3(i+1,j,k)]+ux[I3(i,j+1,k)]+ux[I3(i,j,k+1)]+ux[I3(i+1,j+1,k)]+ux[I3(i,j+1,k+1)]+ux[I3(i+1,j,k+1)]+ux[I3(i+1,j+1,k+1)]);
            par_mpp=0.125f*(ux[I3(i,j,k)]+ux[I3(i-1,j,k)]+ux[I3(i,j+1,k)]+ux[I3(i,j,k+1)]+ux[I3(i-1,j+1,k)]+ux[I3(i,j+1,k+1)]+ux[I3(i-1,j,k+1)]+ux[I3(i-1,j+1,k+1)]);
            par_mmp=0.125f*(ux[I3(i,j,k)]+ux[I3(i-1,j,k)]+ux[I3(i,j-1,k)]+ux[I3(i,j,k+1)]+ux[I3(i-1,j-1,k)]+ux[I3(i,j-1,k+1)]+ux[I3(i-1,j,k+1)]+ux[I3(i-1,j-1,k+1)]);
            par_pmp=0.125f*(ux[I3(i,j,k)]+ux[I3(i+1,j,k)]+ux[I3(i,j-1,k)]+ux[I3(i,j,k+1)]+ux[I3(i+1,j-1,k)]+ux[I3(i,j-1,k+1)]+ux[I3(i+1,j,k+1)]+ux[I3(i+1,j-1,k+1)]);
            par_mpm=0.125f*(ux[I3(i,j,k)]+ux[I3(i-1,j,k)]+ux[I3(i,j+1,k)]+ux[I3(i,j,k-1)]+ux[I3(i-1,j+1,k)]+ux[I3(i,j+1,k-1)]+ux[I3(i-1,j,k-1)]+ux[I3(i-1,j+1,k-1)]);

        }
        if(type=='s')
            par=s[I3(i,j,k)];   
        if(type=='m')
            par=real(maxDS[I3(i,j,k)]); 




            fprintf( stream, "SH(");
            
            fprintf( stream,"%lf,%lf, %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf",
            dh*i-0.5*dh+xm,dh*j-0.5*dh+ym, dh*k-0.5*dh+zm, 
            dh*i+0.5*dh+xm,dh*j-0.5*dh+ym,dh*k-0.5*dh+zm,
            dh*i+0.5*dh+xm,dh*j+0.5*dh+ym,dh*k-0.5*dh+zm,
            dh*i-0.5*dh+xm,dh*j+0.5*dh+ym,dh*k-0.5*dh+zm,
            dh*i-0.5*dh+xm,dh*j-0.5*dh+ym,dh*k+0.5*dh+zm,
            dh*i+0.5*dh+xm,dh*j-0.5*dh+ym,dh*k+0.5*dh+zm,
            dh*i+0.5*dh+xm,dh*j+0.5*dh+ym, dh*k+0.5*dh+zm,
            dh*i-0.5*dh+xm,dh*j+0.5*dh+ym,dh*k+0.5*dh+zm);



            fprintf( stream, "){");
            fprintf( stream,"%le,    %le, %le, %le, %le, %le, %le, %le};\n",par_mmm ,par_pmm,par_ppm,par_mpm,par_mmp ,par_pmp,par_ppp,par_mpp);

        }   

        if((type=='p')||(type=='u'))
            delete [] uiso;

    }       

        fprintf( stream, "};");

        fclose(stream);
        




}



void print_data_in_z_direction(int M, int N, int K, int min2, real R){
    
    char fname[200];

    real *u_mean=new real[K];
    real *v_mean=new real[K];
    real *w_mean=new real[K];
    real *u_rms=new real[K];
    real *v_rms=new real[K];
    real *w_rms=new real[K];
    real *k_z=new real[K];
    real *eps_z=new real[K];

    for(int k=0;k<K;k++){
        u_mean[k]=0.0;
        v_mean[k]=0.0;
        w_mean[k]=0.0;
        u_rms[k]=0.0;
        v_rms[k]=0.0;
        w_rms[k]=0.0;
        k_z[k]=0.0;
        eps_z[k]=0.0;
    }

    for(int k=0;k<K;k++)
    for(int i=0;i<M;i++)
    for(int j=0;j<N;j++){
        u_mean[k]+=uxa[I3(i,j,k)]/(1.0*M*N);
        v_mean[k]+=uya[I3(i,j,k)]/(1.0*M*N);
        w_mean[k]+=uza[I3(i,j,k)]/(1.0*M*N);
        u_rms[k]+=sqrt(ux_rms[I3(i,j,k)])/(1.0*M*N);
        v_rms[k]+=sqrt(uy_rms[I3(i,j,k)])/(1.0*M*N);
        w_rms[k]+=sqrt(uz_rms[I3(i,j,k)])/(1.0*M*N);
        k_z[k]+=(k_ux[I3(i,j,k)]+k_uy[I3(i,j,k)]+k_uz[I3(i,j,k)])/(1.0*M*N);
        eps_z[k]+=(lk1[I3(i,j,k)]+lk2[I3(i,j,k)]+lk3[I3(i,j,k)]+lk12[I3(i,j,k)]+lk21[I3(i,j,k)]+lk13[I3(i,j,k)]+lk31[I3(i,j,k)]+lk32[I3(i,j,k)]+lk23[I3(i,j,k)])/(1.0*M*N);
    }
    
    FILE *stream;
        
    sprintf(fname, "%i_%.02lf_U_mean.dat", min2, (double)R);
    stream=fopen(fname, "w" );
    for(int k=0;k<K;k++){
        fprintf( stream, "%le %le %le\n",(double)u_mean[k],(double)v_mean[k],(double)w_mean[k]);
    }
    fclose(stream);
    sprintf(fname, "%i_%.02lf_U_rms.dat", min2, (double)R);
    stream=fopen(fname, "w" );
    for(int k=0;k<K;k++){
        fprintf( stream, "%le %le %le\n",(double)u_rms[k],(double)v_rms[k],(double)w_rms[k]);
    }
    sprintf(fname, "%i_%.02lf_k_eps.dat", min2, (double)R);
    stream=fopen(fname, "w" );
    for(int k=0;k<K;k++){
        fprintf( stream, "%le %le\n",(double)k_z[k],(double)eps_z[k]);
    }


    fclose(stream);

    delete [] u_mean, v_mean, w_mean, u_rms, v_rms, w_rms, eps_z, k_z;
}





int get_number_of_points(char f_name[]){
int tempi=0;
char tempc[25];

FILE *stream;

    
    stream=fopen( f_name, "r+" );
    fseek( stream, 0L, SEEK_SET );
    
    
    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );


    fscanf( stream, "%i", &tempi );
    
    fclose( stream );

    return tempi;

}






real* get_point_array(char f_name[], int* number_of_max2_p){
real tempd=0.0;
int tempi=0;
char tempc[20];
FILE *stream;


    stream=fopen( f_name, "r+" );
    fseek( stream, 0L, SEEK_SET );

    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );
    fscanf( stream, "%s", tempc );


    fscanf( stream, "%i", &tempi );

    int number_of_points=tempi;
    
    real* pnt;
    pnt=new real[number_of_points*4];
    


    for(int i=0;i<number_of_points;i++){
        fscanf( stream, "%i", &tempi );
        pnt[i]=tempi;
        fscanf( stream, "%lf", &tempd );
        pnt[i+number_of_points]=tempd;
        fscanf( stream, "%lf", &tempd );
        pnt[i+number_of_points*2]=tempd;
        fscanf( stream, "%lf", &tempd );
        pnt[i+number_of_points*3]=tempd;


    }
    
    fclose( stream );

    int max2_number=int(pnt[number_of_points-1]);
    real* point1;
    point1=new real[max2_number*3+1];
    *number_of_max2_p=max2_number;
    
    for(int i=0;i<max2_number*3;i++)
        point1[i]=0.0;



    for(int i=0;i<number_of_points;i++){
        int point_number=(int)pnt[i];
        point1[point_number]=pnt[i+number_of_points];   //x=offset+segment
        point1[point_number+max2_number]=pnt[i+number_of_points*2]; //y=offset+segment*2
        point1[point_number+max2_number*2]=pnt[i+number_of_points*3]; //z=offset+segment*3

    }



    
    delete [] pnt;

    return point1;
}







void get_number_of_terahedra(char f_name[], int* n_o_t, int* n_o_b){

int tempi=0;
char tempc[250];
tempc[1]='a';tempc[0]='n';tempc[2]='b';
FILE *stream;


    stream=fopen( f_name, "r+" );
    fseek( stream, 0L, SEEK_SET );  
        

    while( (tempc[1]!='E')||(tempc[0]!='$')||(tempc[2]!='l')){  
        fscanf( stream, "%s", tempc );
    }

    fscanf( stream, "%i", &tempi );
    int number_of_total_elements=tempi;
    int element_number=0, flag=1;
    int number_of_tetra=0;
    int number_of_boundary=0;
    //int number_of_tetrahedra=0;

    for(int k=0;k<number_of_total_elements;k++){
        fscanf( stream, "%i", &element_number );
        fscanf( stream, "%i", &flag );
    
        if(flag==15){
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
        }

        else if(flag==1){
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            

        }

        else if(flag==2){
            number_of_boundary++;
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );

        }

        else if(flag==4){
            number_of_tetra++;
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );

                        
            fscanf( stream, "%i", &tempi );
                    
            fscanf( stream, "%i", &tempi );
                
            fscanf( stream, "%i", &tempi );
        
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
        

        }   
    
    }

    fclose( stream );

//  return number_of_tetra;

    *n_o_t=number_of_tetra;
    *n_o_b=number_of_boundary;
}



void get_tetrahedra_array(char f_name[], int* tetra_array, int number_of_tetrahedra, int* boundaries, int number_of_boundary_planes){
int tempi=0;
char tempc[250];
tempc[1]='a';tempc[0]='n';tempc[2]='b';
FILE *stream;


    stream=fopen( f_name, "r+" );
    fseek( stream, 0L, SEEK_SET );

    

    while( (tempc[1]!='E')||(tempc[0]!='$')||(tempc[2]!='l')){  
        fscanf( stream, "%s", tempc );
    }

    fscanf( stream, "%i", &tempi );
    int number_of_total_elements=tempi;
    int element_number=0, flag=1, k_delta=0;
    bool init_array=true;
    int boundary_plane_count=0;
    

    for(int k=0;k<number_of_total_elements;k++){
        fscanf( stream, "%i", &element_number );
        fscanf( stream, "%i", &flag );
        
        if(flag==15){
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
    
        }
        

        else if(flag==1){
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
        }

        else if(flag==2){
            
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            boundaries[boundary_plane_count]=tempi;

            fscanf( stream, "%i", &tempi );
            //void
            fscanf( stream, "%i", &tempi );
            boundaries[boundary_plane_count+number_of_boundary_planes]=tempi;
            fscanf( stream, "%i", &tempi );
            boundaries[boundary_plane_count+number_of_boundary_planes*2]=tempi;
            fscanf( stream, "%i", &tempi );
            boundaries[boundary_plane_count+number_of_boundary_planes*3]=tempi;

            boundary_plane_count++;
        }
        else if(flag==4){
            if(init_array==true){
                init_array=false;
                k_delta=element_number-1;
            }
            if((k-k_delta+1)<=number_of_tetrahedra){
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );
            fscanf( stream, "%i", &tempi );

            tetra_array[(k-k_delta)]=element_number;
            
            fscanf( stream, "%i", &tempi );
            tetra_array[(k-k_delta)+number_of_tetrahedra]=tempi;
        
            fscanf( stream, "%i", &tempi );
            tetra_array[(k-k_delta)+number_of_tetrahedra*2]=tempi;
        
            fscanf( stream, "%i", &tempi );
            tetra_array[(k-k_delta)+number_of_tetrahedra*3]=tempi;
        
            fscanf( stream, "%i", &tempi );
            tetra_array[(k-k_delta)+number_of_tetrahedra*4]=tempi;

            }

        }   
    
    }

    fclose(stream);
    

}








/************************************************************************/
/* INIT and DELETE ALL                                                  */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }


    for(i = 0; i < count; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf( "#%i:   %s  \n", i, &deviceProp);
    }
    int device_number=0;
    printf("\nEnter device number to use>>>");
    scanf("%i", &device_number);

    
    
    cudaSetDevice(device_number);

    printf("CUDA initialized.\n");
    return true;
}

#endif

void init_arrays(int M, int N, int K){

d0=new real[(M)*(N)*(K)];
d1=new real[(M)*(N)*(K)];
d2=new real[(M)*(N)*(K)];
d3=new real[(M)*(N)*(K)];
d4=new real[(M)*(N)*(K)];
d5=new real[(M)*(N)*(K)];
d6=new real[(M)*(N)*(K)];
d7=new real[(M)*(N)*(K)];
d8=new real[(M)*(N)*(K)];
d9=new real[(M)*(N)*(K)];
d10=new real[(M)*(N)*(K)];
d11=new real[(M)*(N)*(K)];
d12=new real[(M)*(N)*(K)];
d13=new real[(M)*(N)*(K)];
d14=new real[(M)*(N)*(K)];
d15=new real[(M)*(N)*(K)];
d16=new real[(M)*(N)*(K)];
d17=new real[(M)*(N)*(K)];
d18=new real[(M)*(N)*(K)];



//bc=new int[(M)*(N)*(K)];
maxDS=new int[(M)*(N)*(K)];
ux=new real[(M)*(N)*(K)];
uy=new real[(M)*(N)*(K)];
uz=new real[(M)*(N)*(K)];
uxa=new real[(M)*(N)*(K)];
uya=new real[(M)*(N)*(K)];
uza=new real[(M)*(N)*(K)];
ro=new real[(M)*(N)*(K)];
rot=new real[(M)*(N)*(K)];
rota=new real[(M)*(N)*(K)];
s=new real[(M)*(N)*(K)];
lk=new real[(M)*(N)*(K)];

uxi=new real[(M)*(N)*(K)];
uyi=new real[(M)*(N)*(K)];
uzi=new real[(M)*(N)*(K)];


ux_rms=new real[(M)*(N)*(K)];
uy_rms=new real[(M)*(N)*(K)];
uz_rms=new real[(M)*(N)*(K)];
k_ux=new real[(M)*(N)*(K)];
k_uy=new real[(M)*(N)*(K)];
k_uz=new real[(M)*(N)*(K)];
lk1=new real[(M)*(N)*(K)];
lk2=new real[(M)*(N)*(K)];
lk3=new real[(M)*(N)*(K)];
lk12=new real[(M)*(N)*(K)];
lk21=new real[(M)*(N)*(K)];
lk31=new real[(M)*(N)*(K)];
lk13=new real[(M)*(N)*(K)];
lk23=new real[(M)*(N)*(K)];
lk32=new real[(M)*(N)*(K)];


    for(int i=0;i<M;i++)
    for(int j=0;j<N;j++)
    for(int k=0;k<K;k++){
            //bc[I3(i,j,k)]=FLUID;
            maxDS[I3(i,j,k)]=0;
            ux[I3(i,j,k)]=0.0f;
            uy[I3(i,j,k)]=0.0f;
            uz[I3(i,j,k)]=0.0f;
            ro[I3(i,j,k)]=0.0f;
            rot[I3(i,j,k)]=0.0f;
            rota[I3(i,j,k)]=0.0f;
            s[I3(i,j,k)]=0.0f;
            uxa[I3(i,j,k)]=0.0f;
            uya[I3(i,j,k)]=0.0f;
            uza[I3(i,j,k)]=0.0f;
            uxi[I3(i,j,k)]=0.0f;
            uyi[I3(i,j,k)]=0.0f;
            uzi[I3(i,j,k)]=0.0f;    
            ux_rms[I3(i,j,k)]=0.0f; 
            uy_rms[I3(i,j,k)]=0.0f;
            uz_rms[I3(i,j,k)]=0.0f;
            k_ux[I3(i,j,k)]=0.0f;   
            k_uy[I3(i,j,k)]=0.0f;   
            k_uz[I3(i,j,k)]=0.0f;   

            lk[I3(i,j,k)]=0.0f;
            lk1[I3(i,j,k)]=0.0f;
            lk2[I3(i,j,k)]=0.0f;
            lk3[I3(i,j,k)]=0.0f;
            
            
            lk12[I3(i,j,k)]=0.0f;
            lk21[I3(i,j,k)]=0.0f;
            lk32[I3(i,j,k)]=0.0f;
            lk23[I3(i,j,k)]=0.0f;
            lk13[I3(i,j,k)]=0.0f;
            lk31[I3(i,j,k)]=0.0f;

            d0[I3(i,j,k)]=w[0];
            d1[I3(i,j,k)]=w[1];
            d2[I3(i,j,k)]=w[2];
            d3[I3(i,j,k)]=w[3];
            d4[I3(i,j,k)]=w[4];
            d5[I3(i,j,k)]=w[5];         
            d6[I3(i,j,k)]=w[6];
            d7[I3(i,j,k)]=w[7];
            d8[I3(i,j,k)]=w[8];
            d9[I3(i,j,k)]=w[9];
            d10[I3(i,j,k)]=w[10];
            d11[I3(i,j,k)]=w[11];
            d12[I3(i,j,k)]=w[12];
            d13[I3(i,j,k)]=w[13];
            d14[I3(i,j,k)]=w[14];
            d15[I3(i,j,k)]=w[15];
            d16[I3(i,j,k)]=w[16];
            d17[I3(i,j,k)]=w[17];           
            d18[I3(i,j,k)]=w[18];
            
    }



}

void delete_arrays_CPU(){

    delete [] ux,uy,uz,ro,bc,rot,rota,maxDS,s,uxa,uya,uza,lk,uxi,uyi,uzi;
    delete [] ux_rms, uy_rms, uz_rms,k_ux_v, k_uy_v, k_uz_v;
    delete [] lk1,lk2,lk3,lk12,lk21,lk32,lk23,lk13,lk31;
    delete [] d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18;

    //delete [] Points;
    //delete [] Coords;

}



real read_mesh(int Num, int* M, int* N, int* K, real* x_min1, real* y_min1, real* z_min1){
char f_name[25];    
real dh=1.0;
x_min1[0]=0.0;
y_min1[0]=0.0;
z_min1[0]=0.0;

    char st1='a';
    char st='a';
    printf("\nLoad Mesh File?(y/n)>>>");
    scanf("%c",&st1);
    scanf("%c",&st);
    if(st1=='y'){
        
        int Num1=Num;



        printf("\nMesh file name>>>");
        scanf("%25s",f_name);
        scanf("%c",&st);

        printf("\nMesh scale factor(int)>>>");
        scanf("%i",&Num1);
        scanf("%c",&st);


        real *point;
        int *tetrahedra;
        int* boundaries;

        int number_of_points=0;
        int number_of_t=0;
        int number_of_b=0;
        int number_of_max2_p=0;
    


        number_of_points=get_number_of_points(f_name);
        point=get_point_array(f_name, &number_of_max2_p);
    
        get_number_of_terahedra(f_name,&number_of_t,&number_of_b);
    
        boundaries=new int[number_of_b*4];
        tetrahedra=new int[number_of_t*5];

        get_tetrahedra_array(f_name, tetrahedra, number_of_t, boundaries, number_of_b);
        int n_cv=number_of_t;
        int n_points=number_of_max2_p;
        
        real x_max=-1.0E5;
        real x_min=+1.0E5;
        real y_max=-1.0E5;
        real y_min=+1.0E5;
        real z_max=-1.0E5;
        real z_min=+1.0E5;
        //real *XC=new real[number_of_t];
        //real *YC=new real[number_of_t];
        //real *ZC=new real[number_of_t];
        
        real *XA=new real[number_of_t];
        real *YA=new real[number_of_t];
        real *ZA=new real[number_of_t];

        real *XB=new real[number_of_t];
        real *YB=new real[number_of_t];
        real *ZB=new real[number_of_t];

        real *XC=new real[number_of_t];
        real *YC=new real[number_of_t];
        real *ZC=new real[number_of_t];

        real *XD=new real[number_of_t];
        real *YD=new real[number_of_t];
        real *ZD=new real[number_of_t]; 

        for(int no=0;no<n_points;no++){
            real x=point[no+1];
            real y=point[no+1+n_points];
            real z=point[no+1+n_points*2];
            if(x_max<x)
                x_max=x;
            if(x_min>x)
                x_min=x;
            if(y_max<y)
                y_max=y;
            if(y_min>y)
                y_min=y;
            if(z_max<z)
                z_max=z;
            if(z_min>z)
                z_min=z;

        }
        for(int no=0;no<n_cv;no++){
            int A_Point=tetrahedra[no+n_cv];
            int B_Point=tetrahedra[no+n_cv*2];
            int C_Point=tetrahedra[no+n_cv*3];
            int D_Point=tetrahedra[no+n_cv*4];
            //real x_c=0.25*(point[A_Point]+point[B_Point]+point[C_Point]+point[D_Point]);
            //real y_c=0.25*(point[A_Point+n_points]+point[B_Point+n_points]+point[C_Point+n_points]+point[D_Point+n_points]);
            //real z_c=0.25*(point[A_Point+n_points*2]+point[B_Point+n_points*2]+point[C_Point+n_points*2]+point[D_Point+n_points*2]);
            
            XA[no]=point[A_Point];
            YA[no]=point[A_Point+n_points];
            ZA[no]=point[A_Point+n_points*2];

            XB[no]=point[B_Point];
            YB[no]=point[B_Point+n_points];
            ZB[no]=point[B_Point+n_points*2];

            XC[no]=point[C_Point];
            YC[no]=point[C_Point+n_points];
            ZC[no]=point[C_Point+n_points*2];


            XD[no]=point[D_Point];
            YD[no]=point[D_Point+n_points];
            ZD[no]=point[D_Point+n_points*2];


            //XC[no]=x_c;
            //YC[no]=y_c;
            //ZC[no]=z_c;
            


        }

        
        real* minX=new real[number_of_t];
        real* maxX=new real[number_of_t];
        real* minY=new real[number_of_t];
        real* maxY=new real[number_of_t];
        real* minZ=new real[number_of_t];
        real* maxZ=new real[number_of_t];

        for(int no=0;no<n_cv;no++){
            
            minX[no]=XA[no];
            if(minX[no]>XB[no]) minX[no]=XB[no];
            if(minX[no]>XC[no]) minX[no]=XC[no];
            if(minX[no]>XD[no]) minX[no]=XD[no];
            maxX[no]=XA[no];
            if(maxX[no]<XB[no]) maxX[no]=XB[no];
            if(maxX[no]<XC[no]) maxX[no]=XC[no];
            if(maxX[no]<XD[no]) maxX[no]=XD[no];
            
            minY[no]=YA[no];
            if(minY[no]>YB[no]) minY[no]=YB[no];
            if(minY[no]>YC[no]) minY[no]=YC[no];
            if(minY[no]>YD[no]) minY[no]=YD[no];
            maxY[no]=YA[no];
            if(maxY[no]<YB[no]) maxY[no]=YB[no];
            if(maxY[no]<YC[no]) maxY[no]=YC[no];
            if(maxY[no]<YD[no]) maxY[no]=YD[no];

            minZ[no]=ZA[no];
            if(minZ[no]>ZB[no]) minZ[no]=ZB[no];
            if(minZ[no]>ZC[no]) minZ[no]=ZC[no];
            if(minZ[no]>ZD[no]) minZ[no]=ZD[no];
            maxZ[no]=ZA[no];
            if(maxZ[no]<ZB[no]) maxZ[no]=ZB[no];
            if(maxZ[no]<ZC[no]) maxZ[no]=ZC[no];
            if(maxZ[no]<ZD[no]) maxZ[no]=ZD[no];


        }



        real x_size=x_max-x_min;
        real y_size=y_max-y_min;
        real z_size=z_max-z_min;

        x_min1[0]=x_min;
        y_min1[0]=y_min;
        z_min1[0]=z_min;

        real min_size=x_size;
        if(min_size>y_size) min_size=y_size;
        if(min_size>z_size) min_size=z_size;
        

        M[0]=int(x_size*Num1/min_size);
        N[0]=int(y_size*Num1/min_size);
        K[0]=int(z_size*Num1/min_size);
        dh=real(z_size)/real(K[0]);
        bc=new int[M[0]*N[0]*K[0]];
        
        int no1=0;
        for(int i=0;i<M[0];i++){
            for(int j=0;j<N[0];j++)
                for(int k=0;k<K[0];k++){
                    int kk=(i)*(N[0])*(K[0])+(K[0])*(j)+(k);    
                    bc[kk]=FLUID;               
                    for(int no1=0;no1<n_cv;no1++){


                        real x_m=real(i)*dh+x_min;
                        real x_p=real(i)*dh+x_min;
                        real y_m=real(j)*dh+y_min;
                        real y_p=real(j)*dh+y_min;  
                        real z_m=real(k)*dh+z_min;
                        real z_p=real(k)*dh+z_min;
                    
                            if((minX[no1]<=x_m)&&(maxX[no1]>=x_p))
                            if((minY[no1]<=y_m)&&(maxY[no1]>=y_p))
                            if((minZ[no1]<=z_m)&&(maxZ[no1]>=z_p)){
                                
                                int kk=(i)*(N[0])*(K[0])+(K[0])*(j)+(k);    
                                bc[kk]=WALL;
                                //printf("              %.i         \r",no1);

                            }
                                    
            

                    }
                }
        
            printf("[%.03f%%]   \r", real(i)*100.0f/real(M[0]) );

        }
    }
    else{

        bc=new int[M[0]*N[0]*K[0]];
        for(int i=0;i<M[0];i++)
            for(int j=0;j<N[0];j++)
                for(int k=0;k<K[0];k++){
                    int kk=(i)*(N[0])*(K[0])+(K[0])*(j)+(k);
                    bc[kk]=FLUID;
                }

    }

return (dh);

}



/************************************************************************/
/* CPU Calculation funcitons                                            */
/************************************************************************/


real feq0(int q, real ro, real ux, real uy, real uz){
    real vv=cx[q]*ux+cy[q]*uy+cz[q]*uz;
    real cs=1.0;
    
    return real(w[q]*ro*(1.0+3.0*vv/(cs)+9.0/2.0*(vv*vv)/(cs*cs)-3.0/2.0*(ux*ux+uy*uy+uz*uz)/(cs)));

}


void rotor(real *ux, real *uy, real *uz, real *rot, int M, int N,int K){

    for(int i=1;i<M-1;i++)
        for(int j=1;j<N-1;j++)
            for(int k=1;k<K-1;k++){
                real rot_x=((uz[I3(i,j+1,k)]-uz[I3(i,j-1,k)])-(uy[I3(i,j,k+1)]-uy[I3(i,j,k-1)]));
                real rot_y=((ux[I3(i,j,k+1)]-ux[I3(i,j,k-1)])-(uz[I3(i+1,j,k)]-uz[I3(i-1,j,k)]));
                real rot_z=((uy[I3(i+1,j,k)]-uy[I3(i-1,j,k)])-(ux[I3(i,j+1,k)]-ux[I3(i,j-1,k)]));
                rot[I3(i,j,k)]=
                sqrt(rot_x*rot_x+rot_y*rot_y+rot_z*rot_z);


        }

}




void add_perturbations_conditions(int M,int N,int K, real amplitude){
    
for(int i=0;i<M;i++)
    for(int j=0;j<N;j++)
        for(int k=0;k<K;k++){
            real pert=rand_normal(0.0, amplitude);
            ro[I3(i,j,k)]+=0.0f;
            ux[I3(i,j,k)]+=0.0f;
            uy[I3(i,j,k)]+=0.0f;
            uz[I3(i,j,k)]+=pert;
            pert=rand_normal(0.0, amplitude);
        }
}





void initial_conditions(int M,int N,int K){
    
for(int i=0;i<M;i++)
    for(int j=0;j<N;j++)
        for(int k=0;k<K;k++){
                ro[I3(i,j,k)]=1.0f;
                ux[I3(i,j,k)]=0.0f;//Uin;
                uy[I3(i,j,k)]=0.0f;
                uz[I3(i,j,k)]=0.0f;


                real x=(1.0*i-1.0*M/2.0)/(1.0*M);
                real y=(1.0*j-1.0*N/2.0)/(1.0*N);
                if((x*x+y*y)<0.2*0.2){
                //if((i>2*M/3)&&(i<M-1))
                //if((j>N/3)&&(j<2*N/3))
                //  if((k>K/3)&&(k<2*K/3)){
                        uz[I3(i,j,k)]=0.05f;
                //  }
                }
                else{
                    uz[I3(i,j,k)]=-0.05f;
                }


                real k_critical=3.117;
                real dh=1.0/(1.0*K);
                real Lx=dh*M*1.0;
                real Ly=dh*N*1.0;



                d0[I3(i,j,k)]=feq0(0, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d1[I3(i,j,k)]=feq0(1, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d2[I3(i,j,k)]=feq0(2, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d3[I3(i,j,k)]=feq0(3, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d4[I3(i,j,k)]=feq0(4, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d5[I3(i,j,k)]=feq0(5, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d6[I3(i,j,k)]=feq0(6, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d7[I3(i,j,k)]=feq0(7, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d8[I3(i,j,k)]=feq0(8, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d9[I3(i,j,k)]=feq0(9, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d10[I3(i,j,k)]=feq0(10, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d11[I3(i,j,k)]=feq0(11, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d12[I3(i,j,k)]=feq0(12, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d13[I3(i,j,k)]=feq0(13, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d14[I3(i,j,k)]=feq0(14, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d15[I3(i,j,k)]=feq0(15, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d16[I3(i,j,k)]=feq0(16, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d17[I3(i,j,k)]=feq0(17, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);
                d18[I3(i,j,k)]=feq0(18, ro[I3(i,j,k)], ux[I3(i,j,k)], uy[I3(i,j,k)],uz[I3(i,j,k)]);


            }

}

void set_bc(int M, int N, int K){



    for(int j=0;j<N;j++)
        for(int k=0;k<K;k++){   
          bc[I3(0,j,k)]=IN;
          bc[I3(M-1,j,k)]=OUT;
//            bc[I3(0,j,k)]=WALL;
//            bc[I3(M-1,j,k)]=WALL;
        }


    for(int i=0;i<M;i++)
        for(int k=0;k<K;k++){   
            bc[I3(i,0,k)]=WALL;
            bc[I3(i,N-1,k)]=WALL;
        }

    
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++){   
            bc[I3(i,j,0)]=WALL;
            bc[I3(i,j,K-1)]=WALL;
        }

//This regualtes a simple obsticle
//* 
    int I_start=0;
    int I_end=M/8;//45;
    int J_start=0;//N/2-1;
    int J_end=N/3.0;//N/2+1;
    int K_start=0;
    int K_end=K;
    
    //I_start=15;
    //I_end=22;
    //J_start=N/2-4;
    //J_end=N/2+4;
    //K_start=K/2-4;
    //K_end=K/2+4;


    for(int i=I_start;i<=I_end;i++)
    for(int j=J_start;j<=J_end;j++)
    for(int k=K_start;k<=K_end;k++){
            bc[I3(i,j,k)]=WALL;
        }
//*/



}


/************************************************************************/
/* GPU Memory operations!                                               */
/************************************************************************/

void copy_mem_to_device(real* aray, real* dev,int size){

 // copy host memory to device
    unsigned  int mem_size=sizeof(real) * (size);
    
    cudaMemcpy(dev, aray, mem_size, cudaMemcpyHostToDevice) ;
     
    
}

void copy_mem_to_device1(int* aray, int* dev,int size){

 // copy host memory to device
    unsigned  int mem_size=sizeof(int) * (size);
    
    cudaMemcpy(dev, aray, mem_size, cudaMemcpyHostToDevice) ;
     
    
}

void copy_device_to_mem(real* aray, real* dev,int size){

 // copy device memory to host
    unsigned  int mem_size=sizeof(real)* (size);
    
    cudaMemcpy(aray, dev, mem_size, cudaMemcpyDeviceToHost);
    
    
}

void copy_device_to_mem1(int* aray, int* dev,int size){

 // copy device memory to host
    unsigned  int mem_size=sizeof(int) * (size);
    
    cudaMemcpy(aray, dev, mem_size, cudaMemcpyDeviceToHost);
    
    
}


real* allocate_device_mem(int size, real* aray){
    
    unsigned  int mem_size=sizeof(real) * size;
    real *m_device=aray;
    
    cudaMalloc((void**)&m_device, mem_size);
   //cudaSafeCall(cudaMallocHost ((void **) &array, mem_size));
   return m_device;
}

int* allocate_device_mem1(int size, int* aray){
    
    unsigned  int mem_size=sizeof(int) * size;
    int *m_device=aray;
    
    cudaMalloc((void**)&m_device, mem_size);
   //cudaSafeCall(cudaMallocHost ((void **) &array, mem_size));
   return m_device;
}



void allocate_GPU_mem(int M, int N, int K){

    ux_v=allocate_device_mem((M)*(N)*(K), ux_v);
    uy_v=allocate_device_mem((M)*(N)*(K), uy_v);
    uz_v=allocate_device_mem((M)*(N)*(K), uz_v);
    
    uxa_v=allocate_device_mem((M)*(N)*(K), uxa_v);
    uya_v=allocate_device_mem((M)*(N)*(K), uya_v);
    uza_v=allocate_device_mem((M)*(N)*(K), uza_v);

    uxa1_v=allocate_device_mem((M)*(N)*(K), uxa1_v);
    uya1_v=allocate_device_mem((M)*(N)*(K), uya1_v);
    uza1_v=allocate_device_mem((M)*(N)*(K), uza1_v);

    uxi_v=allocate_device_mem((M)*(N)*(K), uxi_v);
    uyi_v=allocate_device_mem((M)*(N)*(K), uyi_v);
    uzi_v=allocate_device_mem((M)*(N)*(K), uzi_v);
    

    ux_rms_v=allocate_device_mem((M)*(N)*(K), ux_rms_v);
    uy_rms_v=allocate_device_mem((M)*(N)*(K), uy_rms_v);
    uz_rms_v=allocate_device_mem((M)*(N)*(K), uz_rms_v);

    k_ux_v=allocate_device_mem((M)*(N)*(K), k_ux_v);
    k_uy_v=allocate_device_mem((M)*(N)*(K), k_uy_v);
    k_uz_v=allocate_device_mem((M)*(N)*(K), k_uz_v);

    lk1_v=allocate_device_mem((M)*(N)*(K), lk1_v);
    lk2_v=allocate_device_mem((M)*(N)*(K), lk2_v);
    lk3_v=allocate_device_mem((M)*(N)*(K), lk3_v);

    lk12_v=allocate_device_mem((M)*(N)*(K), lk12_v);
    lk21_v=allocate_device_mem((M)*(N)*(K), lk21_v);
    lk31_v=allocate_device_mem((M)*(N)*(K), lk31_v);
    lk13_v=allocate_device_mem((M)*(N)*(K), lk13_v);
    lk23_v=allocate_device_mem((M)*(N)*(K), lk23_v);
    lk32_v=allocate_device_mem((M)*(N)*(K), lk32_v);

    ux_old_v=allocate_device_mem((M)*(N)*(K), ux_old_v);
    uy_old_v=allocate_device_mem((M)*(N)*(K), uy_old_v);
    ro_v=allocate_device_mem((M)*(N)*(K), ro_v);
    ro_old_v=allocate_device_mem((M)*(N)*(K), ro_old_v);
    
    bc_v=allocate_device_mem1((M)*(N)*(K), bc_v);
    maxDS_v=allocate_device_mem1((M)*(N)*(K), maxDS_v);
    
   
    f0=allocate_device_mem((M)*(N)*(K), f0);
    f1=allocate_device_mem((M)*(N)*(K), f1);
    f2=allocate_device_mem((M)*(N)*(K), f2);
    f3=allocate_device_mem((M)*(N)*(K), f3);
    f4=allocate_device_mem((M)*(N)*(K), f4);
    f5=allocate_device_mem((M)*(N)*(K), f5);
    f6=allocate_device_mem((M)*(N)*(K), f6);
    f7=allocate_device_mem((M)*(N)*(K), f7);
    f8=allocate_device_mem((M)*(N)*(K), f8);
    f9=allocate_device_mem((M)*(N)*(K), f9);
    f10=allocate_device_mem((M)*(N)*(K), f10);
    f11=allocate_device_mem((M)*(N)*(K), f11);
    f12=allocate_device_mem((M)*(N)*(K), f12);
    f13=allocate_device_mem((M)*(N)*(K), f13);
    f14=allocate_device_mem((M)*(N)*(K), f14);
    f15=allocate_device_mem((M)*(N)*(K), f15);
    f16=allocate_device_mem((M)*(N)*(K), f16);
    f17=allocate_device_mem((M)*(N)*(K), f17);
    f18=allocate_device_mem((M)*(N)*(K), f18);

    f0p=allocate_device_mem((M)*(N)*(K), f0p);
    f1p=allocate_device_mem((M)*(N)*(K), f1p);
    f2p=allocate_device_mem((M)*(N)*(K), f2p);
    f3p=allocate_device_mem((M)*(N)*(K), f3p);
    f4p=allocate_device_mem((M)*(N)*(K), f4p);
    f5p=allocate_device_mem((M)*(N)*(K), f5p);
    f6p=allocate_device_mem((M)*(N)*(K), f6p);
    f7p=allocate_device_mem((M)*(N)*(K), f7p);
    f8p=allocate_device_mem((M)*(N)*(K), f8p);
    f9p=allocate_device_mem((M)*(N)*(K), f9p);
    f10p=allocate_device_mem((M)*(N)*(K), f10p);
    f11p=allocate_device_mem((M)*(N)*(K), f11p);
    f12p=allocate_device_mem((M)*(N)*(K), f12p);
    f13p=allocate_device_mem((M)*(N)*(K), f13p);
    f14p=allocate_device_mem((M)*(N)*(K), f14p);
    f15p=allocate_device_mem((M)*(N)*(K), f15p);
    f16p=allocate_device_mem((M)*(N)*(K), f16p);
    f17p=allocate_device_mem((M)*(N)*(K), f17p);
    f18p=allocate_device_mem((M)*(N)*(K), f18p);

}




void copy_main_CPU_to_GPU(int M, int N, int K){
    
    copy_mem_to_device(ux, ux_v, (M)*(N)*(K) );
    copy_mem_to_device(uy, uy_v, (M)*(N)*(K) );
    copy_mem_to_device(uz, uz_v, (M)*(N)*(K) );
    copy_mem_to_device(ro, ro_v, (M)*(N)*(K) ); 


}



void copy_main_GPU_to_CPU(int M, int N, int K){

    copy_device_to_mem(ux,ux_v, (M)*(N)*(K) );
    copy_device_to_mem(uy,uy_v, (M)*(N)*(K) );
    copy_device_to_mem(uz,uz_v, (M)*(N)*(K) );
    copy_device_to_mem(ro,ro_v, (M)*(N)*(K) );


}


void copy_CPU_mem_to_GPU_mem(int M, int N, int K){
    
    
    copy_mem_to_device(ux, ux_v, (M)*(N)*(K) );
    copy_mem_to_device(uy, uy_v, (M)*(N)*(K) );
    copy_mem_to_device(uz, uz_v, (M)*(N)*(K) );
    
    copy_mem_to_device(ux, uxa_v, (M)*(N)*(K) );
    copy_mem_to_device(uy, uya_v, (M)*(N)*(K) );
    copy_mem_to_device(uz, uza_v, (M)*(N)*(K) );
    copy_mem_to_device(uxa, uxa1_v, (M)*(N)*(K) );
    copy_mem_to_device(uya, uya1_v, (M)*(N)*(K) );
    copy_mem_to_device(uza, uza1_v, (M)*(N)*(K) );
    
    copy_mem_to_device(ux, uxi_v, (M)*(N)*(K) );    
    copy_mem_to_device(uy, uyi_v, (M)*(N)*(K) );
    copy_mem_to_device(uz, uzi_v, (M)*(N)*(K) );
    
    copy_mem_to_device(ux, lk1_v, (M)*(N)*(K) );    
    copy_mem_to_device(uy, lk2_v, (M)*(N)*(K) );
    copy_mem_to_device(uz, lk3_v, (M)*(N)*(K) );
    copy_mem_to_device(lk12, lk12_v, (M)*(N)*(K) );
    copy_mem_to_device(lk13, lk13_v, (M)*(N)*(K) );
    copy_mem_to_device(lk32, lk32_v, (M)*(N)*(K) );
    copy_mem_to_device(lk23, lk23_v, (M)*(N)*(K) );
    copy_mem_to_device(lk21, lk21_v, (M)*(N)*(K) );
    copy_mem_to_device(lk31, lk31_v, (M)*(N)*(K) );


    copy_mem_to_device(uz_rms,ux_rms_v, (M)*(N)*(K) );
    copy_mem_to_device(uz_rms,uy_rms_v, (M)*(N)*(K) );
    copy_mem_to_device(uz_rms,uz_rms_v, (M)*(N)*(K) );

    copy_mem_to_device(k_ux,k_ux_v, (M)*(N)*(K) );
    copy_mem_to_device(k_uy,k_uy_v, (M)*(N)*(K) );
    copy_mem_to_device(k_uz,k_uz_v, (M)*(N)*(K) );


    copy_mem_to_device(ux, ux_old_v, (M)*(N)*(K) ); 
    copy_mem_to_device(uy, uy_old_v, (M)*(N)*(K) ); //?
    copy_mem_to_device(ro, ro_v, (M)*(N)*(K) ); 
    copy_mem_to_device(ro, ro_old_v, (M)*(N)*(K) );  //?
 

    copy_mem_to_device1(bc, bc_v, (M)*(N)*(K) );
    copy_mem_to_device1(maxDS,maxDS_v, (M)*(N)*(K) ); //?
    
    copy_mem_to_device(d0, f0, (M)*(N)*(K) );   
    copy_mem_to_device(d1, f1, (M)*(N)*(K) );
    copy_mem_to_device(d2, f2, (M)*(N)*(K) );
    copy_mem_to_device(d3, f3, (M)*(N)*(K) );   
    copy_mem_to_device(d4, f4, (M)*(N)*(K) );
    copy_mem_to_device(d5, f5, (M)*(N)*(K) );
    copy_mem_to_device(d6, f6, (M)*(N)*(K) );   
    copy_mem_to_device(d7, f7, (M)*(N)*(K) );
    copy_mem_to_device(d8, f8, (M)*(N)*(K) );
    copy_mem_to_device(d9, f9, (M)*(N)*(K) );   
    copy_mem_to_device(d10, f10, (M)*(N)*(K) ); 
    copy_mem_to_device(d11, f11, (M)*(N)*(K) );
    copy_mem_to_device(d12, f12, (M)*(N)*(K) );
    copy_mem_to_device(d13, f13, (M)*(N)*(K) ); 
    copy_mem_to_device(d14, f14, (M)*(N)*(K) );
    copy_mem_to_device(d15, f15, (M)*(N)*(K) );
    copy_mem_to_device(d16, f16, (M)*(N)*(K) ); 
    copy_mem_to_device(d17, f17, (M)*(N)*(K) );
    copy_mem_to_device(d18, f18, (M)*(N)*(K) );
    
    copy_mem_to_device(d0, f0p, (M)*(N)*(K) );  
    copy_mem_to_device(d1, f1p, (M)*(N)*(K) );
    copy_mem_to_device(d2, f2p, (M)*(N)*(K) );
    copy_mem_to_device(d3, f3p, (M)*(N)*(K) );  
    copy_mem_to_device(d4, f4p, (M)*(N)*(K) );
    copy_mem_to_device(d5, f5p, (M)*(N)*(K) );
    copy_mem_to_device(d6, f6p, (M)*(N)*(K) );  
    copy_mem_to_device(d7, f7p, (M)*(N)*(K) );
    copy_mem_to_device(d8, f8p, (M)*(N)*(K) );
    copy_mem_to_device(d9, f9p, (M)*(N)*(K) );  
    copy_mem_to_device(d10, f10p, (M)*(N)*(K) );    
    copy_mem_to_device(d11, f11p, (M)*(N)*(K) );
    copy_mem_to_device(d12, f12p, (M)*(N)*(K) );
    copy_mem_to_device(d13, f13p, (M)*(N)*(K) );    
    copy_mem_to_device(d14, f14p, (M)*(N)*(K) );
    copy_mem_to_device(d15, f15p, (M)*(N)*(K) );
    copy_mem_to_device(d16, f16p, (M)*(N)*(K) );    
    copy_mem_to_device(d17, f17p, (M)*(N)*(K) );
    copy_mem_to_device(d18, f18p, (M)*(N)*(K) );

   

}

void copy_GPU_mem_to_CPU_mem(int M, int N, int K){

    copy_device_to_mem(ux,ux_v, (M)*(N)*(K) );
    copy_device_to_mem(uy,uy_v, (M)*(N)*(K) );
    copy_device_to_mem(uz,uz_v, (M)*(N)*(K) );
    
    copy_device_to_mem(uxa,uxa_v, (M)*(N)*(K) );
    copy_device_to_mem(uya,uya_v, (M)*(N)*(K) );
    copy_device_to_mem(uza,uza_v, (M)*(N)*(K) );

    copy_device_to_mem(uxi,uxi_v, (M)*(N)*(K) );
    copy_device_to_mem(uyi,uyi_v, (M)*(N)*(K) );
    copy_device_to_mem(uzi,uzi_v, (M)*(N)*(K) );

    copy_device_to_mem(ux_rms,ux_rms_v, (M)*(N)*(K) );
    copy_device_to_mem(uy_rms,uy_rms_v, (M)*(N)*(K) );
    copy_device_to_mem(uz_rms,uz_rms_v, (M)*(N)*(K) );

    copy_device_to_mem(k_ux,k_ux_v, (M)*(N)*(K) );
    copy_device_to_mem(k_uy,k_uy_v, (M)*(N)*(K) );
    copy_device_to_mem(k_uz,k_uz_v, (M)*(N)*(K) );

    copy_device_to_mem(lk1, lk1_v, (M)*(N)*(K) );   
    copy_device_to_mem(lk2, lk2_v, (M)*(N)*(K) );
    copy_device_to_mem(lk3, lk3_v, (M)*(N)*(K) );
    copy_device_to_mem(lk12, lk12_v, (M)*(N)*(K) );
    copy_device_to_mem(lk13, lk13_v, (M)*(N)*(K) );
    copy_device_to_mem(lk32, lk32_v, (M)*(N)*(K) );
    copy_device_to_mem(lk23, lk23_v, (M)*(N)*(K) );
    copy_device_to_mem(lk21, lk21_v, (M)*(N)*(K) );
    copy_device_to_mem(lk31, lk31_v, (M)*(N)*(K) );

    copy_device_to_mem(lk,ux_old_v, (M)*(N)*(K) );

    copy_device_to_mem(ro,ro_v, (M)*(N)*(K) );
    copy_device_to_mem(s,ux_old_v, (M)*(N)*(K) );
    copy_device_to_mem1(maxDS,maxDS_v, (M)*(N)*(K) );
    
    copy_device_to_mem1(bc,bc_v, (M)*(N)*(K) );


    copy_device_to_mem(d0, f0, (M)*(N)*(K) );   
    copy_device_to_mem(d1, f1, (M)*(N)*(K) );
    copy_device_to_mem(d2, f2, (M)*(N)*(K) );
    copy_device_to_mem(d3, f3, (M)*(N)*(K) );   
    copy_device_to_mem(d4, f4, (M)*(N)*(K) );
    copy_device_to_mem(d5, f5, (M)*(N)*(K) );
    copy_device_to_mem(d6, f6, (M)*(N)*(K) );   
    copy_device_to_mem(d7, f7, (M)*(N)*(K) );
    copy_device_to_mem(d8, f8, (M)*(N)*(K) );
    copy_device_to_mem(d9, f9, (M)*(N)*(K) );   
    copy_device_to_mem(d10, f10, (M)*(N)*(K) ); 
    copy_device_to_mem(d11, f11, (M)*(N)*(K) );
    copy_device_to_mem(d12, f12, (M)*(N)*(K) );
    copy_device_to_mem(d13, f13, (M)*(N)*(K) ); 
    copy_device_to_mem(d14, f14, (M)*(N)*(K) );
    copy_device_to_mem(d15, f15, (M)*(N)*(K) );
    copy_device_to_mem(d16, f16, (M)*(N)*(K) ); 
    copy_device_to_mem(d17, f17, (M)*(N)*(K) );
    copy_device_to_mem(d18, f18, (M)*(N)*(K) );


}

void delete_arrays_GPU(){

    cudaFree(ux_v);
    cudaFree(uy_v);
    cudaFree(uz_v);
    
    cudaFree(uxa_v);
    cudaFree(uya_v);
    cudaFree(uza_v);
    cudaFree(uxa1_v);
    cudaFree(uya1_v);
    cudaFree(uza1_v);
    cudaFree(uxi_v);
    cudaFree(uyi_v);
    cudaFree(uzi_v);    
    
    cudaFree(ux_rms_v);
    cudaFree(uy_rms_v);
    cudaFree(uz_rms_v);
    cudaFree(k_ux_v);
    cudaFree(k_uy_v);
    cudaFree(k_uz_v);

    cudaFree(lk1_v);
    cudaFree(lk2_v);
    cudaFree(lk3_v);    
    cudaFree(lk12_v);
    cudaFree(lk13_v);
    cudaFree(lk23_v);
    cudaFree(lk21_v);
    cudaFree(lk31_v);
    cudaFree(lk32_v);


    cudaFree(ux_old_v); 
    cudaFree(uy_old_v);
    cudaFree(ro_v);
    cudaFree(ro_old_v);
    cudaFree(bc_v);
    cudaFree(maxDS_v);

    cudaFree(f0);
    cudaFree(f1);
    cudaFree(f2);
    cudaFree(f3);
    cudaFree(f4);
    cudaFree(f5);
    cudaFree(f6);
    cudaFree(f7);
    cudaFree(f8);
    cudaFree(f9);
    cudaFree(f10);
    cudaFree(f11);
    cudaFree(f12);
    cudaFree(f13);
    cudaFree(f14);
    cudaFree(f15);
    cudaFree(f16);
    cudaFree(f18);
    
    cudaFree(f0p);
    cudaFree(f1p);
    cudaFree(f2p);
    cudaFree(f3p);
    cudaFree(f4p);
    cudaFree(f5p);
    cudaFree(f6p);
    cudaFree(f7p);
    cudaFree(f8p);
    cudaFree(f9p);
    cudaFree(f10p);
    cudaFree(f11p);
    cudaFree(f12p);
    cudaFree(f13p);
    cudaFree(f14p);
    cudaFree(f15p);
    cudaFree(f16p);
    cudaFree(f18p);


}




/************************************************************************/
/* GPU KERNELS!!!!                                                      */
/************************************************************************/
    #define BLOCK_DIM_X 16
    #define BLOCK_DIM_Y 16





__global__ void  kernel_points_output(int t, int M, int N, int K, real* vPoints, int NumP, real* vCoords, real* vU,  real* vV,  real* vW){


    int n = threadIdx.x;
    



    if (n < NumP){
        int _x=int(vCoords[I2(0,n)]);
        int _y=int(vCoords[I2(1,n)]);
        int _z=int(vCoords[I2(2,n)]);

        int coord=I3(_x,_y,_z);
        vPoints[I2p(0,n,t)]=vU[coord];
        vPoints[I2p(1,n,t)]=vV[coord];
        vPoints[I2p(2,n,t)]=vW[coord];

    }



}



__global__ void kernel_avaraged_and_RMS(int M, int N, int K, real* ux_v, real* uy_v, real* uz_v,  real* uxa_v, real* uya_v, real* uza_v, real* ux_rms_v, real* uy_rms_v, real* uz_rms_v, real TimeInterval){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;

unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);

        uxa_v[kk]+=ux_v[kk]/TimeInterval*10.0;
        uya_v[kk]+=uy_v[kk]/TimeInterval*10.0;
        uza_v[kk]+=uz_v[kk]/TimeInterval*10.0;
        
        ux_rms_v[kk]+=ux_v[kk]*ux_v[kk]/TimeInterval*10.0*10.0;
        uy_rms_v[kk]+=uy_v[kk]*uy_v[kk]/TimeInterval*10.0*10.0;
        uz_rms_v[kk]+=uz_v[kk]*uz_v[kk]/TimeInterval*10.0*10.0;

    

}

}



__global__ void kernel_instant_and_turbulent_energy(int M, int N, int K, real* ux_v, real* uy_v, real* uz_v,  real* uxa_v, real* uya_v, real* uza_v, real* uxi_v, real* uyi_v, real* uzi_v, real*  k_ux_v, real* k_uy_v, real* k_uz_v, real TimeInterval){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;

unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);
    


        
        real ux_i=(ux_v[kk]*10.0-uxa_v[kk]);
        
        real uy_i=(uy_v[kk]*10.0-uya_v[kk]);
        
        real uz_i=(uz_v[kk]*10.0-uza_v[kk]);

        k_ux_v[kk]+=ux_i*ux_i/TimeInterval;
        k_uy_v[kk]+=uy_i*uy_i/TimeInterval;
        k_uz_v[kk]+=uz_i*uz_i/TimeInterval;

        uxi_v[kk]=ux_i;
        uyi_v[kk]=uy_i;
        uzi_v[kk]=uz_i;


}

}



__global__ void kernel_stat_to_zero(int M, int N, int K, real* uxa_v, real* uya_v, real* uza_v, real* ux_rms_v, real* uy_rms_v, real* uz_rms_v, real* uxi_v, real* uyi_v, real* uzi_v, real*  k_ux_v, real* k_uy_v, real* k_uz_v, real* lk1_v, real* lk2_v, real* lk3_v, real* lk12_v, real* lk13_v, real* lk23_v, real* lk21_v, real* lk31_v, real* lk32_v, real TimeInterval){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;

unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);
    
    uxa_v[kk]=0.0;
    uya_v[kk]=0.0;
    uza_v[kk]=0.0;  
    ux_rms_v[kk]=0.0;   
    uy_rms_v[kk]=0.0;   
    uz_rms_v[kk]=0.0;
    uxi_v[kk]=0.0;  
    uyi_v[kk]=0.0;
    uzi_v[kk]=0.0;
    k_ux_v[kk]=0.0; 
    k_uy_v[kk]=0.0;
    k_uz_v[kk]=0.0; 
    lk1_v[kk]=0.0;
    lk2_v[kk]=0.0;
    lk3_v[kk]=0.0;
    
    lk13_v[kk]=0.0;
    lk23_v[kk]=0.0;
    lk12_v[kk]=0.0;
    
    lk31_v[kk]=0.0;
    lk32_v[kk]=0.0;
    lk21_v[kk]=0.0;
}

}


__global__ void kernel_lk(int M, int N, int K, real* uxi_v, real* uyi_v, real* uzi_v, real* lk1_v, real* lk2_v, real* lk3_v, real* lk12_v, real* lk13_v, real* lk23_v, real* lk21_v, real* lk31_v, real* lk32_v,  real TimeInterval){

real dhx=1.0/real(M);
real dhy=1.0/real(N);
real dhz=1.0/real(K);
unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;
    
unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);


    
        if( (i<M-1)&&(i>0))
            if( (j<N-1)&&(j>0))
                if( (k<K-1)&&(k>0)){
                
                    real k11=0.5f*(uxi_v[I3(i+1,j,k)]-uxi_v[I3(i-1,j,k)])/dhx;
                    real k22=0.5f*(uyi_v[I3(i,j+1,k)]-uyi_v[I3(i,j-1,k)])/dhy;
                    real k33=0.5f*(uzi_v[I3(i,j,k+1)]-uzi_v[I3(i,j,k-1)])/dhz;

                    real k12=0.5f*(uxi_v[I3(i,j+1,k)]-uxi_v[I3(i,j-1,k)])/dhy;
                    real k13=0.5f*(uxi_v[I3(i,j,k+1)]-uxi_v[I3(i,j,k-1)])/dhz;
                    real k23=0.5f*(uyi_v[I3(i,j,k+1)]-uyi_v[I3(i,j,k-1)])/dhz;
                    
                    real k21=0.5f*(uyi_v[I3(i+1,j,k)]-uyi_v[I3(i-1,j,k)])/dhx;
                    real k31=0.5f*(uzi_v[I3(i+1,j,k)]-uzi_v[I3(i-1,j,k)])/dhx;
                    real k32=0.5f*(uzi_v[I3(i,j+1,k)]-uzi_v[I3(i,j-1,k)])/dhy;

                    lk1_v[kk]+=k11*k11/TimeInterval;
                    lk2_v[kk]+=k22*k22/TimeInterval;
                    lk3_v[kk]+=k33*k33/TimeInterval;
                    
                    lk12_v[kk]+=k12*k12/TimeInterval;
                    lk13_v[kk]+=k13*k13/TimeInterval;
                    lk23_v[kk]+=k23*k23/TimeInterval;
                    
                    lk21_v[kk]+=k21*k21/TimeInterval;
                    lk31_v[kk]+=k31*k31/TimeInterval;
                    lk32_v[kk]+=k32*k32/TimeInterval;
                


                }

}

}



__global__ void kernel_lk_calc(int t, real R,  int M, int N, int K, real* lk1_v, real* lk2_v, real* lk3_v, real* ux_old_v){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;
    
unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);


    
        
                

                    real sumdirav=lk1_v[kk]+lk2_v[kk]+lk3_v[kk];
                    if(sumdirav==0.0)
                        ux_old_v[kk]=1.0;
                    else
                        ux_old_v[kk]=pow(real(0.5*R*R*sumdirav),real(-0.25));
                

}

}





__global__ void kernel_copy_0_18(int M, int N, int K, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6,
                                 real *f7, real *f8, real *f9, 
                                real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                 real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, real *f6p, 
                                 real *f7p, real *f8p, real *f9p, 
                                 real *f10p, real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p){

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;
    
unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);

//*


//*/
        //periodic boundary conditions
        //const int cx[Q]={0, 1, -1, 0, 0, 0,  0,  1, -1,  1,  -1 , 1,  -1,  1, -1,  0,  0,  0,  0};
        //const int cy[Q]={0, 0, 0, 1, -1, 0,  0,  1, -1, -1,   1,  0,   0,  0,  0,  1, -1,  1, -1};
        //const int cz[Q]={0, 0, 0, 0, 0,  1, -1,  0,  0,  0,   0,  1,  -1, -1,  1,  1, -1, -1,  1};
        //                 0  1   2  3  4  5   6   7   8   9   10   11   12  13  14  15  16  17  18
/*      
        if(i==0){
        //1,7,9, 11,13
            f1p[kk]=f1[I3(M-1,j,k)];
            f7p[kk]=f7[I3(M-1,j,k)];
            f9p[kk]=f9[I3(M-1,j,k)];
            f11p[kk]=f11[I3(M-1,j,k)];
            f13p[kk]=f13[I3(M-1,j,k)];
        }
        else if(i==M-1){
        //2,8,10,12,14
            f2p[kk]=f2[I3(0,j,k)];
            f8p[kk]=f8[I3(0,j,k)];
            f10p[kk]=f10[I3(0,j,k)];
            f12p[kk]=f12[I3(0,j,k)];
            f14p[kk]=f14[I3(0,j,k)];    

        }
        else if(j==0){
        //3,7,10,15,17
            f3p[kk]=f3[I3(i,N-1,k)];
            f7p[kk]=f7[I3(i,N-1,k)];
            f10p[kk]=f10[I3(i,N-1,k)];
            f15p[kk]=f15[I3(i,N-1,k)];
            f17p[kk]=f17[I3(i,N-1,k)];  
                                    
        }
        else if(j==N-1){
        //4,8,9,16,18
            f4p[kk]=f4[I3(i,0,k)];
            f8p[kk]=f8[I3(i,0,k)];
            f9p[kk]=f9[I3(i,0,k)];
            f16p[kk]=f16[I3(i,0,k)];
            f18p[kk]=f18[I3(i,0,k)];                    


        }

        else{
//*/            
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

//      }



//*/


}

}




__global__ void kernel_stream3D_0_18_forward(int M, int N, int K, int* bc_v, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, 
                                            real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                            real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, real *f6p, real *f7p, real *f8p, real *f9p, 
                                            real *f10p, real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p){





unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;
    
unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);




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

const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};




//2<=>1; 4<=>3; 6<=>5; 8<=>7; 10<=>9; 12<=>11; 14<=>13; 16<=>15; 18<=>17
    

                                     
        f0p[kk]=vf0;
        
        int _i=i+cx[1]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        int _j=j+cy[1]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        int _k=k+cz[1]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        int kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f1p[kk1]=vf1;


        _i=i+cx[2]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[2]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[2]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f2p[kk1]=vf2;


        _i=i+cx[3]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[3]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[3]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f3p[kk1]=vf3;


        _i=i+cx[4]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[4]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[4]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f4p[kk1]=vf4;


        _i=i+cx[5]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[5]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[5]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f5p[kk1]=vf5;


        _i=i+cx[6]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[6]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[6]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f6p[kk1]=vf6;




        _i=i+cx[7]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[7]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[7]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f7p[kk1]=vf7;


        _i=i+cx[8]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[8]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[8]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f8p[kk1]=vf8;


        _i=i+cx[9]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[9]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[9]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f9p[kk1]=vf9;



        _i=i+cx[10]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[10]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[10]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f10p[kk1]=vf10;


        _i=i+cx[11]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[11]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[11]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f11p[kk1]=vf11;


        _i=i+cx[12]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[12]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[12]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f12p[kk1]=vf12;



        _i=i+cx[13]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[13]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[13]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f13p[kk1]=vf13;


        _i=i+cx[14]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[14]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[14]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f14p[kk1]=vf14;


        _i=i+cx[15]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[15]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[15]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f15p[kk1]=vf15;


        _i=i+cx[16]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[16]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[16]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f16p[kk1]=vf16;


        _i=i+cx[17]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[17]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[17]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f17p[kk1]=vf17;


        _i=i+cx[18]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        _j=j+cy[18]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        _k=k+cz[18]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
        f18p[kk1]=vf18;
 
}

}





__global__ void kernel_wall3D_0_18(int M, int N, int K, int* bc_v, 
                                   real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, real *f10,
                                   real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                   real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, real *f6p, real *f7p, real *f8p, real *f9p, real *f10p, 
                                   real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p){





unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;
    
unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=I3(i,j,k);




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


const int cx[Q]={0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
const int cy[Q]={0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
const int cz[Q]={0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};



//2<=>1; 4<=>3; 6<=>5; 8<=>7; 10<=>9; 12<=>11; 14<=>13; 16<=>15; 18<=>17

        if(bc_v[kk]==WALL){                          
            f0p[kk]=vf0;
            
        //  int _i=i+cx[1]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  int _j=j+cy[1]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  int _k=k+cz[1]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  int kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f1p[kk]=f2[kk];


        //  _i=i+cx[2]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[2]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[2]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f2p[kk]=f1[kk];


        //  _i=i+cx[3]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[3]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[3]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f3p[kk]=f4[kk];


        //  _i=i+cx[4]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[4]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[4]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f4p[kk]=f3[kk];


        //  _i=i+cx[5]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[5]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[5]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f5p[kk]=f6[kk];


        //  _i=i+cx[6]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[6]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[6]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f6p[kk]=f5[kk];




        //  _i=i+cx[7]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[7]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[7]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f7p[kk]=f8[kk];


        //  _i=i+cx[8]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[8]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[8]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f8p[kk]=f7[kk];


        //  _i=i+cx[9]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[9]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[9]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f9p[kk]=f10[kk];

        //  _i=i+cx[10]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[10]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[10]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f10p[kk]=f9[kk];

        //  _i=i+cx[11]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[11]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[11]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f11p[kk]=f12[kk];


        //  _i=i+cx[12]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[12]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[12]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f12p[kk]=f11[kk];


        //  _i=i+cx[13]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[13]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[13]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f13p[kk]=f14[kk];


        //  _i=i+cx[14]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[14]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[14]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f14p[kk]=f13[kk];


        //  _i=i+cx[15]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[15]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[15]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f15p[kk]=f16[kk];


        //  _i=i+cx[16]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[16]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[16]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f16p[kk]=f15[kk];




        //  _i=i+cx[17]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[17]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[17]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f17p[kk]=f18[kk];


        //  _i=i+cx[18]; //if(_i<0) _i=M-1; if(_i>M-1) _i=0;
        //  _j=j+cy[18]; //if(_j<0) _j=N-1; if(_j>N-1) _j=0;
        //  _k=k+cz[18]; //if(_k<0) _k=K-1; if(_k>K-1) _k=0;
        //  kk1=I3P(_i,_j,_k);// _k+_j*(K)+_i*(N)*(K);
            f18p[kk]=f17[kk];

        }
/*
        else{
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







__global__ void kernel_macro_0_18(real* uinit,  int M, int N, int K, real *ux_v, real *uy_v, real *uz_v, real *ro_v, int* bc_v, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18){
    

unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;
    
unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);

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

//*/
    if(bc_v[I3(i,j,k)]==IN){
            
        v_x=ux_in;//uinit[I3(0,j,k)];
        v_y=uy_in;
        v_z=uz_in;
        //  ro=ro_v[I3(i,j-1,k)];
        ro=ro_v[I3(i+1,j,k)];
    }
    else if(bc_v[I3(i,j,k)]==OUT){
        ro=1.0;//+(N-float(j))*0.05;
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






__global__ void kernel_collide_0_18(real delta, real *ux_old_v, real gx, real gy, real gz, int M, int N, int K, real omega, real *ux_v, real *uy_v, real *uz_v, real *ro_v,  int* bc_v, 
                                    real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, 
                                    real *f6, real *f7, real *f8, real *f9, 
                                    real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                    real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, 
                                    real *f6p, real *f7p, real *f8p, real *f9p, 
                                    real *f10p, real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p){



unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;
    
unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);

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
        real ro=ro_v[I3(i,j,k)];
        real v_x=ux_v[I3(i,j,k)];
        real v_y=uy_v[I3(i,j,k)];
        real v_z=uz_v[I3(i,j,k)];

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
      if((bc_v[I3(i,j,k)]==IN)||(bc_v[I3(i,j,k)]==OUT)){
            vf0=f_eq0;  vf1=f_eq1;  vf2=f_eq2;  vf3=f_eq3;  vf4=f_eq4;  vf5=f_eq5;  
            vf6=f_eq6;  vf7=f_eq7;  vf8=f_eq8;  vf9=f_eq9;  
            vf10=f_eq10;    vf11=f_eq11;  vf12=f_eq12;  vf13=f_eq13;    vf14=f_eq14;    vf15=f_eq15;    vf16=f_eq16;    vf17=f_eq17; vf18=f_eq18;
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

        if(bc_v[I3(i,j,k)]!=WALL){
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

/*      if(fabsf(ux_old_v[kk])>delta){//*uy_old_v[0]){
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







__global__ void kernel_correction_0_18(real* ux_old_v, real* uy_old_v, int M, int N, int K, real *ux_v, real *uy_v, real *uz_v, real *ro_v, int* bc_v, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18){


unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;
    
unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);



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





__global__ void kernel_init_inst(int M, int N, int K,  real* uxi_v,real*  uyi_v, real* uzi_v,    real* lk1_v,real*  lk2_v, real* lk3_v, real* lk12_v,real*  lk21_v, real* lk31_v,real* lk13_v,real*  lk23_v, real* lk32_v){


unsigned int t1, xIndex, yIndex, zIndex, index_in, index_out, gridIndex ;
// step 1: compute gridIndex in 1-D and 1-D data index "index_in"
unsigned int sizeOfData=(unsigned int) M*N*K;
gridIndex = blockIdx.y * gridDim.x + blockIdx.x ;
index_in = ( gridIndex * BLOCK_DIM_Y + threadIdx.y )*BLOCK_DIM_X + threadIdx.x ;
// step 2: extract 3-D data index via
// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
// where xIndex = i-1, yIndex = j-1, zIndex = k-1
if ( index_in < sizeOfData ){
t1 =  index_in/ K; 
zIndex = index_in - K*t1 ;
xIndex =  t1/ N; 
yIndex = t1 - N * xIndex ;
    
unsigned int i=xIndex, j=yIndex, k=zIndex;
int kk=k+j*(K)+i*(N)*(K);


    uxi_v[kk]=0.0f;
    uyi_v[kk]=0.0f;
    uzi_v[kk]=0.0f;
    lk1_v[kk]=0.0f;
    lk2_v[kk]=0.0f;
    lk3_v[kk]=0.0f;
    lk12_v[kk]=0.0f;
    lk21_v[kk]=0.0f;
    lk31_v[kk]=0.0f;
    lk13_v[kk]=0.0f;
    lk23_v[kk]=0.0f;
    lk32_v[kk]=0.0f;    

}



}


/************************************************************************/
/* GPU KERNELS ENDS!!!!                                                 */
/************************************************************************/


void track_point(int p, real **point, int M, int N, int K, real d_time){
    real s0, t0, s1, t1, q0, q1;


    if(point[0][p]<0.5f) point[0][p]=0.5f;
    if(point[1][p]<0.5f) point[1][p]=0.5f;
    if(point[2][p]<0.5f) point[2][p]=0.5f;
    if(point[0][p]>real(M-1.5f)) point[0][p]=real(M-1.5f);
    if(point[1][p]>real(N-1.5f)) point[1][p]=real(N-1.5f);
    if(point[2][p]>real(K-1.5f)) point[2][p]=real(K-1.5f);


//periodic in XY:
/*
    if(point[0][p]<0.5f) point[0][p]=real(M-1.0-point[0][p]);
    if(point[1][p]<0.5f) point[1][p]=real(N-1.0-point[1][p]);
    if(point[2][p]<0.5f) point[2][p]=0.5f;
    if(point[0][p]>real(M-1.5f)) point[0][p]=M-point[0][p];
    if(point[1][p]>real(N-1.5f)) point[1][p]=N-point[1][p];
    if(point[2][p]>real(K-1.5f)) point[2][p]=real(K-1.5f);  
*/

    real x0=point[0][p];
    real y0=point[1][p];
    real z0=point[2][p];
    int i0=int(x0);
    int j0=int(y0);
    int k0=int(z0);
    int i1=i0+1;
    int j1=j0+1;
    int k1=k0+1;
    


    s1 = x0-i0; s0 = 1.0-s1; t1 = y0-j0; t0 = 1.0-t1;  q1 = z0-k0; q0 = 1.0-q1; 
    
    
    real uxi000=ux[I3(i0,j0,k0)];
    real uyi000=uy[I3(i0,j0,k0)];
    real uzi000=uz[I3(i0,j0,k0)];
    if(bc[I3(i0,j0,k0)]==WALL){
        uxi000=0.0;
        uyi000=0.0;
        uzi000=0.0;
    }
        
    real uxi100=ux[I3(i1,j0,k0)];
    real uyi100=uy[I3(i1,j0,k0)];
    real uzi100=uz[I3(i1,j0,k0)];
    if(bc[I3(i1,j0,k0)]==WALL){
        uxi100=0.0;
        uyi100=0.0;
        uzi100=0.0;
    }

    real uxi110=ux[I3(i1,j1,k0)];
    real uyi110=uy[I3(i1,j1,k0)];
    real uzi110=uz[I3(i1,j1,k0)];
    if(bc[I3(i1,j1,k0)]==WALL){
        uxi110=0.0;
        uyi110=0.0;
        uzi110=0.0;
    }

    real uxi101=ux[I3(i1,j0,k1)];
    real uyi101=uy[I3(i1,j0,k1)];
    real uzi101=uz[I3(i1,j0,k1)];
    if(bc[I3(i1,j0,k1)]==WALL){
        uxi101=0.0;
        uyi101=0.0;
        uzi101=0.0;
    }

    real uxi111=ux[I3(i1,j1,k1)];
    real uyi111=uy[I3(i1,j1,k1)];
    real uzi111=uz[I3(i1,j1,k1)];
    if(bc[I3(i1,j1,k1)]==WALL){
        uxi111=0.0;
        uyi111=0.0;
        uzi111=0.0;
    }

    real uxi010=ux[I3(i0,j1,k0)];
    real uyi010=uy[I3(i0,j1,k0)];
    real uzi010=uz[I3(i0,j1,k0)];
    if(bc[I3(i0,j1,k0)]==WALL){
        uxi010=0.0;
        uyi010=0.0;
        uzi010=0.0;
    }

    real uxi011=ux[I3(i0,j1,k1)];
    real uyi011=uy[I3(i0,j1,k1)];
    real uzi011=uz[I3(i0,j1,k1)];
    if(bc[I3(i0,j1,k1)]==WALL){
        uxi011=0.0;
        uyi011=0.0;
        uzi011=0.0;
    }

    real uxi001=ux[I3(i0,j0,k1)];
    real uyi001=uy[I3(i0,j0,k1)];
    real uzi001=uz[I3(i0,j0,k1)];
    if(bc[I3(i0,j0,k1)]==WALL){
        uxi001=0.0;
        uyi001=0.0;
        uzi001=0.0;
    }



    real vel_x=s0*t0*q0*uxi000+
                s1*t0*q0*uxi100+
                s0*t1*q0*uxi010+
                s1*t1*q0*uxi110+
                s1*t0*q1*uxi101+
                s0*t1*q1*uxi011+
                s0*t0*q1*uxi001+
                s1*t1*q1*uxi111; 
    
    real vel_y=s0*t0*q0*uyi000+
                s1*t0*q0*uyi100+
                s0*t1*q0*uyi010+
                s1*t1*q0*uyi110+
                s1*t0*q1*uyi101+
                s0*t1*q1*uyi011+
                s0*t0*q1*uyi001+
                s1*t1*q1*uyi111; 
    
    real vel_z=s0*t0*q0*uzi000+
                s1*t0*q0*uzi100+
                s0*t1*q0*uzi010+
                s1*t1*q0*uzi110+
                s1*t0*q1*uzi101+
                s0*t1*q1*uzi011+
                s0*t0*q1*uzi001+
                s1*t1*q1*uzi111; 

        

    //if((bc[I3(i0,j0,k0)]!=OUT)&&(bc[I3(i1,j1,k1)]!=OUT)){
        point[0][p]=point[0][p]+10.0*d_time*vel_x;
        point[1][p]=point[1][p]+10.0*d_time*vel_y;
        point[2][p]=point[2][p]+10.0*d_time*vel_z;
    //}
    
    

}




/************************************************************************/
/* mainCUDA                                                             */
/************************************************************************/
int main(int argc, char* argv[])
{
    
    //int M=650, N=58, K=140;
    //int M=850, N=75, K=225;
    //int M=200, N=200, K=50;  //8 T! test
    //int M=90, N=90, K=23;  //4 T! test
    //int M=80, N=80, K=20;  //4 T! test

    //int M=400, N=400, K=100;
    //int M=150, N=30, K=70;
    //int M=300, N=50, K=120;
    

    //int M=160, N=160, K=40;  //4 T! test
    
    //M=80;N=80;K=20;

    //int M=240, N=240, K=60;  //4 T! test
    
//1609* 
    //int M=253, N=253, K=32;
    int M=400, N=100, K=100;
    //int M=388, N=388, K=49; //7.92
    //int M=238, N=238, K=30;


    //int M=308, N=308, K=22;
    //int M=100, N=100, K=12;

    int *_M=new int [1];
    int *_N=new int [1];
    int *_K=new int [1];
    real *xm=new real [1];
    real *ym=new real [1];
    real *zm=new real [1];

    real *uinit_c=new real[(N+1)*(K+1)];
    real gx=0.0;
    real gy=0.0;
    real gz=0.000;
    real beta=1.0;
    real gr=sqrt(gx*gx+gy*gy+gz*gz);
    real scx=1.0/(K-2);
    real sct=sqrt(gr*scx); 
    srand ( time(NULL) );
    

    int Num=80;
    real tau=0.0,tauT=0.0;
    real dt=1.0;
    _M[0]=M; _N[0]=N; _K[0]=K; 
    


    real dh=read_mesh(Num, _M,_N,_K,xm,ym,zm);
    M=_M[0]; N=_N[0]; K=_K[0];

    init_arrays(M,N,K);
    
    
    for(int j=0;j<N;j++)
        for(int k=0;k<K;k++)
            uinit_c[I3(0,j,k)]=0.0;

    for(int j=int(N/3.0);j<N;j++)
        for(int k=0;k<K;k++){

            real d1=abs((j-N/3.0)*1.0/(1.0*(N-N/3.0)/2.0));
            real d2=k*1.00/K*1.00/2.0;
            real d3=abs(j*1.0-N*1.0)/(1.0*(N-N/3.0)/2.0);
            real d4=abs(k*1.00-K*1.00)/K*1.00/2.0;

            real dp1=min(d3,d1);
            real dp2=min(d2,d4);
            real dis=sqrt(sqrt(dp1*dp2));
            
            uinit_c[I3(0,j,k)]=0.338346*0.1*1.4*1.4*3.16027*dis;

        }



    printf("N=%i        ",int(M*N*K));
    
    
    char st1='a';   
    printf("\nLoad Cotrol File?(y/n)>>>");
    scanf("%c",&st1);
    real R;
    double Rr;
    printf("\nReynolds number=");
    scanf("%lf",&Rr);
    R=(real)Rr;
    
    
    tau=real(6.0/3.0/R+0.5);
    

    printf("\ntau=%lf",(double)tau);
        
    real omega=1.0/tau;
    int timesteps=10000;
    printf("\ntimesteps=");
    scanf("%i", &timesteps);

    int timesteps_print_start=10000;
    printf("\ntimesteps print start=");
    scanf("%i",&timesteps_print_start);

    int timesteps_print_period=10000;
    printf("\ntimesteps print period=");
    scanf("%i",&timesteps_print_period);

    set_bc(M, N, K);
    initial_conditions(M,N,K);
    
    if(st1=='y'){
        printf("\nreading control files...\n");
        read_control_file(M,N,K);
        read_avaraged_file(M,N,K);
    }


    int NumP=2;
    //NumP=read_point_data_file(Coords, M,N,K,"point_data.dat");
    Coords=new real[NumP*3+1];
    Coords[I2(0,0)]=0.3*M;
    Coords[I2(1,0)]=0.3*N;
    Coords[I2(2,0)]=0.3*K;
    Coords[I2(0,1)]=0.5*M;
    Coords[I2(1,1)]=0.5*N;
    Coords[I2(2,1)]=0.5*K;

    Points=new real [3*NumP*timesteps+1];
    for(int i=0;i<(3*NumP*timesteps+1);i++)
        Points[i]=0.0;



    if(!InitCUDA()) {
        return 0;
    }

    
    allocate_GPU_mem(M,N,K);
    copy_CPU_mem_to_GPU_mem(M,N,K);

    uinit=allocate_device_mem((N+1)*(K+1), uinit);
    copy_mem_to_device(uinit_c, uinit, (N+1)*(K+1) );
    

    vPoints=allocate_device_mem((3*NumP*timesteps+1), vPoints);
    vCoords=allocate_device_mem((3*NumP+1), vCoords);
    int ss=(3*NumP*timesteps+1);
    copy_mem_to_device(Points, vPoints, ss);
    copy_mem_to_device(Coords, vCoords, (3*NumP+1));
    
    delete [] uinit_c;
    

    unsigned int k1, k2 ;
        // step 1: compute # of threads per block
    unsigned int nthreads = BLOCK_DIM_X * BLOCK_DIM_Y ;
        // step 2: compute # of blocks needed
    unsigned int nblocks = ( M*N*K + nthreads -1 )/nthreads ;
        // step 3: find grid's configuration
    real db_nblocks = (real)nblocks ;
    k1 = (unsigned int) floor( sqrt(db_nblocks) ) ;
    k2 = (unsigned int) ceil( db_nblocks/((real)k1)) ;

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 dimGrid( k2, k1, 1 );

    //structure for wall time
    struct timeval start, end;


    printf("\nAll done.Calculating...\n");
    //start calculations!

    int t=0;
    

    kernel_macro_0_18<<< dimGrid, dimBlock>>>(uinit, M, N, K, ux_v, uy_v, uz_v, ro_v, bc_v, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18);

    copy_main_GPU_to_CPU(M, N, K);
    add_perturbations_conditions(M, N, K, 1.0e-3);
    copy_main_CPU_to_GPU(M, N, K);

    kernel_stat_to_zero<<< dimGrid, dimBlock>>>(M, N, K, uxa_v, uya_v, uza_v, ux_rms_v, uy_rms_v, uz_rms_v, uxi_v, uyi_v, uzi_v, k_ux_v, k_uy_v, k_uz_v, lk1_v, lk2_v, lk3_v, lk12_v, lk13_v, lk23_v, lk21_v, lk31_v, lk32_v, 0.0);

    int min2=K;
    if(min2>M) min2=M;
    if(min2>N) min2=N;

    gettimeofday(&start, NULL);

    for(t=0;t<=timesteps;t++){
    


        real delta=0.002;

        //check for NANS!
        real u_l=0.0;
        copy_device_to_mem(&u_l,&ux_v[I3(M/2,N/2,K/2+1)], 1 );
        if(u_l!=u_l){
            fprintf(stderr,"\n!-------------------!\n");
            fprintf(stderr,"!NANS at timestep %i!",t);
            fprintf(stderr,"\n!-------------------!\n");
            break; //NANs!!!!
        }



        //kernel_init_inst<<< dimGrid, dimBlock>>>(M, N, K,  uxi_v, uyi_v, uzi_v, lk1_v, lk2_v, lk3_v);

        //main loop!!!
        
    
        if(t>0){ //for perturbations to take effect
            kernel_macro_0_18<<< dimGrid, dimBlock>>>(uinit, M, N, K, ux_v, uy_v, uz_v, ro_v, bc_v, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18);
        }

        kernel_collide_0_18<<< dimGrid, dimBlock>>>( delta,ux_old_v, gx, gy, gz, M, N, K, omega, ux_v, uy_v, uz_v, ro_v, bc_v, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f0p, f1p, f2p, f3p, f4p, f5p, f6p, f7p, f8p, f9p, f10p, f11p, f12p, f13p, f14p, f15p, f16p, f17p, f18p);

    

        kernel_wall3D_0_18<<< dimGrid, dimBlock>>>(M, N, K, bc_v, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f0p, f1p,  f2p,  f3p,  f4p,  f5p,  f6p,  f7p,  f8p,  f9p,  f10p, f11p,  f12p,  f13p,  f14p,  f15p,  f16p,  f17p,  f18p);

        kernel_stream3D_0_18_forward<<< dimGrid, dimBlock>>>(M,  N, K, bc_v, f0p, f1p, f2p, f3p, f4p, f5p, f6p, f7p, f8p, f9p, f10p, f11p,f12p, f13p, f14p, f15p, f16p, f17p, f18p, f0, f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9, f10,  f11,  f12,  f13,  f14,  f15,  f16,  f17,  f18);
        
    


        kernel_points_output<<<1,NumP>>>(t, M, N, K, vPoints, NumP, vCoords, ux_v, uy_v, uz_v);




        if((t>timesteps/2)&&(t<=timesteps_print_start)){

            kernel_avaraged_and_RMS<<< dimGrid, dimBlock>>>(M, N, K, ux_v, uy_v, uz_v, uxa_v,  uya_v, uza_v, ux_rms_v, uy_rms_v, uz_rms_v, (real)(1.0*timesteps_print_start-timesteps/2.0));

        }



        if(t>timesteps_print_start){

            kernel_instant_and_turbulent_energy<<< dimGrid, dimBlock>>>(M, N, K, ux_v, uy_v, uz_v,  uxa_v, uya_v, uza_v, uxi_v, uyi_v, uzi_v, k_ux_v, k_uy_v, k_uz_v, 1.0*(timesteps-timesteps_print_start));

            kernel_lk<<< dimGrid, dimBlock>>>( M,  N,  K,  uxi_v,  uyi_v,  uzi_v,  lk1_v,  lk2_v,  lk3_v, lk12_v, lk13_v, lk23_v, lk21_v, lk31_v, lk32_v, 1.0*(timesteps-timesteps_print_start));

            if(t%timesteps_print_period==0){

                printf("Printing intermediate results...");
                copy_main_GPU_to_CPU(M, N, K);
                printf("\n velocity vectors...");
                char fname_pos[200];
                sprintf(fname_pos, "%i_%.02lf_%i_V_vec.pos", min2, (double)R, (t-timesteps_print_start));  
                write_out_file(fname_pos, M,  N, K,'v', dh,xm[0],ym[0],zm[0],1,1,1);
                printf("done\n");

            }

        }


        if((t%1000)==0){
            gettimeofday(&end, NULL);
            real wall_time=((end.tv_sec-start.tv_sec)*1000000u+(end.tv_usec-start.tv_usec))/1.0E6;
            real etime=wall_time*1.0*(timesteps-t)/(1.0*t);
            printf("    [%.03lf%%]   %.03le Remaining time:%.01lfsec.\r",(double)(real(t)*100.0/real(timesteps)),10.0*(double)u_l,(double)etime);
            fflush(stdout);
        }


    }
    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }
    gettimeofday(&end, NULL);
    real etime=((end.tv_sec-start.tv_sec)*1000000u+(end.tv_usec-start.tv_usec))/1.0E6;
    printf("\n\nWall time:%lfsec\n",(double)etime);  

    kernel_lk_calc<<< dimGrid, dimBlock>>>( t,  R,   M,  N,  K,  lk1_v,  lk2_v,  lk3_v,  ux_old_v);
    copy_GPU_mem_to_CPU_mem(M,N,K);
    copy_device_to_mem(Points,vPoints, ss);
    
    printf("Points...\n");
    char fname[200];
    char fname_f[200];
    
 
    sprintf(fname, "%i_%.02lf_points_results.dat", min2, (double)R);  
    write_out_points(fname,NumP,timesteps);
    
    sprintf(fname_f, "%i_%.02lf_general_out.dat", min2, (double)R);  

    printf("control...");
    write_control_file(fname_f,M,N,K);

    
    //pos output
    int flagi=1;//30;
    int flagj=1;
    int flagk=1;
    rotor(ux,uy,uz,rot, M, N, K);

    printf(" velocity vectors...");
    char fname_pos[200];
    sprintf(fname_pos, "%i_%.02lf_V_vec.pos", min2, (double)R);  
    write_out_file(fname_pos, M,  N, K,'v',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);

    printf(" av. velocity vectors...");
    sprintf(fname_pos, "%i_%.02lf_V_vec_av.pos", min2, (double)R);
    write_out_file(fname_pos, M,  N, K,'a',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
    
    printf(" l_Kolmogoroff...");
    sprintf(fname_pos, "%i_%.02lf_lk_iso.pos", min2, (double)R);
    write_out_file(fname_pos, M,  N, K,'l',dh, xm[0],ym[0],zm[0],flagi,flagj,flagk);
    
    printf(" curl...");
    sprintf(fname_pos, "%i_%.02lf_Curl_iso.pos", min2, (double)R);
    write_out_file(fname_pos, M,  N, K,'e',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
//  printf(" curl av...");
//  write_out_file("Curl_a_iso.pos", M,  N, K,'q',dh, xm[0],ym[0],zm[0],flagi,flagj,flagk);
    
    
    printf(" velocity magnitude...");
    sprintf(fname_pos, "%i_%.02lf_V_iso.pos", min2, (double)R);   
    write_out_file(fname_pos, M,  N, K, 'p',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
//  printf(" av. velocity magnitude...");
//  write_out_file("V_iso_av.pos", M,  N, K, 'u',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
    
    printf(" density...");
    sprintf(fname_pos, "%i_%.02lf_Ro_iso.pos", min2, (double)R);
    write_out_file(fname_pos, M,  N, K,'r',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);


    /*
    printf(" entropy...");
    write_out_file("Entropy_iso.pos", M,  N,K, 's',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
    printf(" correction mask...");
    write_out_file("Entropy_mark.pos", M,  N, K,'m',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
    */  

    printf(" Statistical_Data...");
    print_data_in_z_direction(M,N,K,min2,R);


    

    //end calculations!
    printf("\nCalculation done. Coping arrays from GPU to CPU and deallocating them...");

    
    
    delete_arrays_GPU();
    //cudaSafeCall(cudaFree(vPoints));
    //cudaSafeCall(cudaFree(vCoords));
    cudaFree(uinit);
    

    printf("\ndone. Wrighting output files:");
    



    printf("\ntrajectories...\n");
    

    int PN=10;
    
    real** point=new real*[3];
    point[0]=new real[PN];
    point[1]=new real[PN];
    point[2]=new real[PN];
    

    for(int n=0;n<PN;n++){
        real k_point=K/(PN+1)*n+0.5;
        point[0][n]=30.0;
        point[1][n]=N/2;
        point[2][n]=k_point;
    }
    
    FILE *stream;
    stream=fopen( "track.geo", "w" );

    int no=0;
    for(int n=0;n<PN;n++)
    for(int q=0;q<10000;q++){
        
        real d_time=0.5;
        track_point(n,point, M, N, K,d_time);       
        //fprintf( stream, "%lf %lf %lf\n", point[0],point[1],point[2]);
        fprintf( stream, "Point (%i) = {%lf, %lf, %lf, 0.1};\n", (no++), point[0][n], point[1][n], point[2][n]);
    }
    
    fclose(stream);
    delete [] point;


/*
    int flagi=1;//30;
    int flagj=1;
    int flagk=1;

    printf(" velocity vectors...");
    write_out_file("V_vec.pos", M,  N,K,'v',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
//  printf(" av. velocity vectors...");
//  write_out_file("V_vec_av.pos", M,  N,K,'a',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
//  printf(" inst. velocity vectors...");
//  write_out_file("V_vec_inst.pos", M,  N,K,'i',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
//  printf(" l_Kolmogoroff...");
//  write_out_file("lk_iso.pos", M,  N, K,'l',dh, xm[0],ym[0],zm[0],flagi,flagj,flagk);
    printf(" curl...");
    write_out_file("Curl_iso.pos", M,  N, K,'e',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
//  printf(" curl av...");
//  write_out_file("Curl_a_iso.pos", M,  N, K,'q',dh, xm[0],ym[0],zm[0],flagi,flagj,flagk);
    printf(" Temperature...");
    write_out_file("T_iso.pos", M,  N, K, 't',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
//  printf(" X velocity magnitude...");
//  write_out_file("Vx_iso.pos", M,  N, K, 'x',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
    printf(" velocity magnitude...");
    write_out_file("V_iso.pos", M,  N, K, 'p',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
//  printf(" av. velocity magnitude...");
//  write_out_file("V_iso_av.pos", M,  N, K, 'u',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
    printf(" density...");
    write_out_file("Ro_iso.pos", M,  N, K,'r',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);

    printf(" Temperature pointwise...");
    write_out_file_1ord("T1D_iso.pos", M,  N, K, dh, 'T');

    /*
    printf(" entropy...");
    write_out_file("Entropy_iso.pos", M,  N,K, 's',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
    printf(" correction mask...");
    write_out_file("Entropy_mark.pos", M,  N, K,'m',dh,xm[0],ym[0],zm[0],flagi,flagj,flagk);
    */
    printf(" boundary...");
    write_out_file_1ord("BC_iso.pos", M,  N, K, dh, 'B');
    printf("control...");
    write_control_file("general_out.dat",M,N,K);
    write_avaraged_file(M,N,K);
    printf("\ndeleting CPU arrays and quitting...");
    delete_arrays_CPU();
    //CUT_EXIT(argc, argv);
    printf("\nAll done!\n");
    return 0;
}
