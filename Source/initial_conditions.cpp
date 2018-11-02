/*
* This file is part of the Lattice Boltzmann multiple GPU distribution. 
(https://github.com/evstigneevnm/LBM_D3Q19_mGPU).
* Copyright (c) 2017-2018 Evstigneev Nikolay Mikhaylovitch and Ryabkov Oleg Igorevich.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 2 only.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#include "initial_conditions.h"


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



real feq0(int q, real rho, real ux, real uy, real uz)
{
    real vv=cx[q]*ux+cy[q]*uy+cz[q]*uz;
    real cs=1.0;
    
    return real(w[q]*rho*(1.0+3.0*vv/(cs)+9.0/2.0*(vv*vv)/(cs*cs)-3.0/2.0*(ux*ux+uy*uy+uz*uz)/(cs)));

}



void add_perturbations_conditions(macroscopic_variables NV, int Nx, int Ny, int Nz, real amplitude)
{
    
for(int i=0;i<Nx;i++)
    for(int j=0;j<Ny;j++)
        for(int k=0;k<Nz;k++)
        {
            real pert=rand_normal(0.0, amplitude);
            NV.rho[I3(i,j,k)]+=0.0f;
            NV.ux[I3(i,j,k)]+=0.0f;
            NV.uy[I3(i,j,k)]+=0.0f;
            NV.uz[I3(i,j,k)]+=pert;
        }
}





void initial_conditions(microscopic_variables MV, macroscopic_variables NV, int Nx,int Ny,int Nz)
{
    
    for(int i=0;i<Nx;i++)
    {
        for(int j=0;j<Ny;j++)
        {
            for(int k=0;k<Nz;k++)
            {
                NV.rho[I3(i,j,k)]=1.0f;
                NV.ux[I3(i,j,k)]=0.0f;//Uin;
                NV.uy[I3(i,j,k)]=0.0f;
                NV.uz[I3(i,j,k)]=0.0f;


                real x=(1.0*i-1.0*Nx/2.0)/(1.0*Nx);
                real y=(1.0*j-1.0*Ny/2.0)/(1.0*Ny);
                if((x*x+y*y)<0.2*0.2)
                {
                //if((i>2*Nx/3)&&(i<Nx-1))
                //if((j>Ny/3)&&(j<2*Ny/3))
                //  if((k>Nz/3)&&(k<2*Nz/3)){
                //        NV.uz[I3(i,j,k)]=0.05f;
                //  }
                }
                else
                {
                    NV.uz[I3(i,j,k)]=0.0f;//-0.05f;
                }


                real k_critical=3.117;
                real dh=1.0/(1.0*Nz);
                real Lx=dh*Nx*1.0;
                real Ly=dh*Ny*1.0;



                MV.d0[I3(i,j,k)]=feq0(0, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d1[I3(i,j,k)]=feq0(1, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d2[I3(i,j,k)]=feq0(2, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d3[I3(i,j,k)]=feq0(3, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d4[I3(i,j,k)]=feq0(4, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d5[I3(i,j,k)]=feq0(5, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d6[I3(i,j,k)]=feq0(6, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d7[I3(i,j,k)]=feq0(7, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d8[I3(i,j,k)]=feq0(8, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d9[I3(i,j,k)]=feq0(9, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d10[I3(i,j,k)]=feq0(10, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d11[I3(i,j,k)]=feq0(11, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d12[I3(i,j,k)]=feq0(12, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d13[I3(i,j,k)]=feq0(13, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d14[I3(i,j,k)]=feq0(14, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d15[I3(i,j,k)]=feq0(15, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d16[I3(i,j,k)]=feq0(16, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d17[I3(i,j,k)]=feq0(17, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);
                MV.d18[I3(i,j,k)]=feq0(18, NV.rho[I3(i,j,k)], NV.ux[I3(i,j,k)], NV.uy[I3(i,j,k)], NV.uz[I3(i,j,k)]);


            }
        }
    }

}




void get_macroscopic(microscopic_variables MV, macroscopic_variables NV, int Nx,int Ny,int Nz)
{
    for(int i=0;i<Nx;i++)
    {
        for(int j=0;j<Ny;j++)
        {
            for(int k=0;k<Nz;k++)
            {
                NV.rho[I3(i,j,k)]=MV.d0[I3(i,j,k)]+MV.d1[I3(i,j,k)]+MV.d2[I3(i,j,k)]+MV.d3[I3(i,j,k)]+MV.d4[I3(i,j,k)]+MV.d5[I3(i,j,k)]+MV.d6[I3(i,j,k)]+MV.d7[I3(i,j,k)]+MV.d8[I3(i,j,k)]+MV.d9[I3(i,j,k)]+MV.d10[I3(i,j,k)]+MV.d11[I3(i,j,k)]+MV.d12[I3(i,j,k)]+MV.d13[I3(i,j,k)]+MV.d14[I3(i,j,k)]+MV.d15[I3(i,j,k)]+MV.d16[I3(i,j,k)]+MV.d17[I3(i,j,k)]+MV.d18[I3(i,j,k)];

                NV.ux[I3(i,j,k)]=(1.0/NV.rho[I3(i,j,k)])*(cx[0]*MV.d0[I3(i,j,k)]+cx[1]*MV.d1[I3(i,j,k)]+cx[2]*MV.d2[I3(i,j,k)]+cx[3]*MV.d3[I3(i,j,k)]+cx[4]*MV.d4[I3(i,j,k)]+cx[5]*MV.d5[I3(i,j,k)]+cx[6]*MV.d6[I3(i,j,k)]+cx[7]*MV.d7[I3(i,j,k)]+cx[8]*MV.d8[I3(i,j,k)]+cx[9]*MV.d9[I3(i,j,k)]+cx[10]*MV.d10[I3(i,j,k)]+cx[11]*MV.d11[I3(i,j,k)]+cx[12]*MV.d12[I3(i,j,k)]+cx[13]*MV.d13[I3(i,j,k)]+cx[14]*MV.d14[I3(i,j,k)]+cx[15]*MV.d15[I3(i,j,k)]+cx[16]*MV.d16[I3(i,j,k)]+cx[17]*MV.d17[I3(i,j,k)]+cx[18]*MV.d18[I3(i,j,k)]);

                NV.uy[I3(i,j,k)]=(1.0/NV.rho[I3(i,j,k)])*(cy[0]*MV.d0[I3(i,j,k)]+cy[1]*MV.d1[I3(i,j,k)]+cy[2]*MV.d2[I3(i,j,k)]+cy[3]*MV.d3[I3(i,j,k)]+cy[4]*MV.d4[I3(i,j,k)]+cy[5]*MV.d5[I3(i,j,k)]+cy[6]*MV.d6[I3(i,j,k)]+cy[7]*MV.d7[I3(i,j,k)]+cy[8]*MV.d8[I3(i,j,k)]+cy[9]*MV.d9[I3(i,j,k)]+cy[10]*MV.d10[I3(i,j,k)]+cy[11]*MV.d11[I3(i,j,k)]+cy[12]*MV.d12[I3(i,j,k)]+cy[13]*MV.d13[I3(i,j,k)]+cy[14]*MV.d14[I3(i,j,k)]+cy[15]*MV.d15[I3(i,j,k)]+cy[16]*MV.d16[I3(i,j,k)]+cy[17]*MV.d17[I3(i,j,k)]+cy[18]*MV.d18[I3(i,j,k)]);
                
                NV.uz[I3(i,j,k)]=(1.0/NV.rho[I3(i,j,k)])*(cz[0]*MV.d0[I3(i,j,k)]+cz[1]*MV.d1[I3(i,j,k)]+cz[2]*MV.d2[I3(i,j,k)]+cz[3]*MV.d3[I3(i,j,k)]+cz[4]*MV.d4[I3(i,j,k)]+cz[5]*MV.d5[I3(i,j,k)]+cz[6]*MV.d6[I3(i,j,k)]+cz[7]*MV.d7[I3(i,j,k)]+cz[8]*MV.d8[I3(i,j,k)]+cz[9]*MV.d9[I3(i,j,k)]+cz[10]*MV.d10[I3(i,j,k)]+cz[11]*MV.d11[I3(i,j,k)]+cz[12]*MV.d12[I3(i,j,k)]+cz[13]*MV.d13[I3(i,j,k)]+cz[14]*MV.d14[I3(i,j,k)]+cz[15]*MV.d15[I3(i,j,k)]+cz[16]*MV.d16[I3(i,j,k)]+cz[17]*MV.d17[I3(i,j,k)]+cz[18]*MV.d18[I3(i,j,k)]);                
            
            }
        }
    }

}

//see definition of boundaries in Macro.h
#ifndef Face2BC
    #define Face2BC(face) face=='W'?WALL:face=='I'?IN:face=='O'?OUT:face=='P'?PERIODIC:face=='C'?MPI_block:FLUID
#endif

void set_boundaries(int Nx, int Ny, int Nz, control_variables CV, communication_variables COM)
{
    //priority from minimal to maximal:
    // C, {I,O}, P, W

    int j,k,l;

    printf("\n%i %i %i %i %i %i \n", Face2BC(COM.FaceA), Face2BC(COM.FaceB), Face2BC(COM.FaceC), Face2BC(COM.FaceD), Face2BC(COM.FaceE), Face2BC(COM.FaceF) );

    for(int p=6;p>=0;p--)
    for(j=0;j<Nx;j++)
    for(k=0;k<Ny;k++)
    for(l=0;l<Nz;l++)
    {
        int FaceA = Face2BC(COM.FaceA);
        int FaceB = Face2BC(COM.FaceB);
        int FaceC = Face2BC(COM.FaceC);
        int FaceD = Face2BC(COM.FaceD);
        int FaceE = Face2BC(COM.FaceE);
        int FaceF = Face2BC(COM.FaceF);

            

            if(p==FaceE){ 
                CV.bc[I3(j,k,0)]=FaceE;
            }
            
            if(p==FaceF){
                CV.bc[I3(j,k,Nz-1)]=FaceF;
            }

            if(p==FaceA){
                CV.bc[I3(0,k,l)]=FaceA;
                
            }
            if(p==FaceB){
                CV.bc[I3(Nx-1,k,l)]=FaceB;
            }

            if(p==FaceC){
                CV.bc[I3(j,0,l)]=FaceC;
            }
            if(p==FaceD){
                CV.bc[I3(j,Ny-1,l)]=FaceD;
            }
        

    }



}