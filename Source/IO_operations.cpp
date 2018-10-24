#include "IO_operations.h"

void write_control_file(char f_name[], communication_variables COM, microscopic_variables MV, int Nx, int Ny, int Nz)
{

    if (COM.myrank == MASTER) {
        FILE *stream = fopen( f_name, "w" );
        fclose(stream);
    }


    FILE *stream;
    stream=fopen(f_name, "a" );
    MPI_Barrier(MPI_COMM_WORLD);

    
    for (int current_process = 0; current_process < COM.totalrank; current_process++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (COM.myrank != current_process) 
            continue;

        for(int j=0;j<Nx;j++)
        {
            for(int k=0;k<Ny;k++)
            {
                for(int l=0;l<Nz;l++)
                {

                    fprintf( stream, "%.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le %.16le\n", 
                            (double)MV.d0[I3(j,k,l)], (double)MV.d1[I3(j,k,l)], (double)MV.d2[I3(j,k,l)], (double)MV.d3[I3(j,k,l)], (double)MV.d4[I3(j,k,l)], 
                            (double)MV.d5[I3(j,k,l)], (double)MV.d6[I3(j,k,l)], (double)MV.d7[I3(j,k,l)],(double)MV.d8[I3(j,k,l)], (double)MV.d9[I3(j,k,l)],
                            (double)MV.d10[I3(j,k,l)], (double)MV.d11[I3(j,k,l)], (double)MV.d12[I3(j,k,l)], (double)MV.d13[I3(j,k,l)], (double)MV.d14[I3(j,k,l)], 
                            (double)MV.d15[I3(j,k,l)], (double)MV.d16[I3(j,k,l)], (double)MV.d17[I3(j,k,l)], (double)MV.d18[I3(j,k,l)]);

                    
                }   
            }
            printf("proc %i writing %s [%.03f%%]   \r", current_process, f_name, 100.0f*real(j)/real(Nx-1) );
            fflush(stream);
        }
        fflush(stream); 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fclose(stream);
    if(COM.myrank == MASTER){
        printf("\n");
        printf("%s output done\n",f_name);
    }
}




int read_control_file(char f_name[], communication_variables COM, microscopic_variables MV, int Nx, int Ny, int Nz)
{

    int tag=0;
    FILE *stream;
    if(stream=fopen(f_name, "r" ))
    {
        fclose(stream);
    }
    else{ 
        if(COM.myrank==MASTER)
            printf("\n %s file doesn't exist or is not accessible.\n",f_name);
        
        MPI_Barrier(MPI_COMM_WORLD);
        return 0;
    }
    
  
    stream=fopen( f_name, "r" );
    fseek( stream, 0L, SEEK_SET );
    long curposition_recieved=0;
    
    for (int current_process = 0;current_process < COM.totalrank ;current_process++) 
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (COM.myrank != current_process)
        {
        
            MPI_Recv(&curposition_recieved, 1, MPI_LONG, current_process, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Proc #%i recieved position is %li\n",COM.myrank, curposition_recieved);
            fseek( stream, curposition_recieved, SEEK_SET );
            continue;
        }

        double d0c, d1c, d2c, d3c, d4c, d5c, d6c, d7c, d8c, d9c, d10c, d11c, d12c, d13c, d14c, d15c, d16c, d17c, d18c;
        
        for(int j=0;j<Nx;j++)
        {
            for(int k=0;k<Ny;k++)
            {
                for(int l=0;l<Nz;l++)
                {
                            
                        
                    fscanf( stream, "%le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le", &d0c, &d1c, &d2c, &d3c, &d4c, &d5c, &d6c, &d7c, &d8c, &d9c, &d10c, &d11c, &d12c, &d13c, &d14c, &d15c, &d16c, &d17c, &d18c);

                    MV.d0[I3(j,k,l)]=(real)d0c;
                    MV.d1[I3(j,k,l)]=(real)d1c;
                    MV.d2[I3(j,k,l)]=(real)d2c;
                    MV.d3[I3(j,k,l)]=(real)d3c;
                    MV.d4[I3(j,k,l)]=(real)d4c;
                    MV.d5[I3(j,k,l)]=(real)d5c;
                    MV.d6[I3(j,k,l)]=(real)d6c;
                    MV.d7[I3(j,k,l)]=(real)d7c;
                    MV.d8[I3(j,k,l)]=(real)d8c;
                    MV.d9[I3(j,k,l)]=(real)d9c;
                    MV.d10[I3(j,k,l)]=(real)d10c;
                    MV.d11[I3(j,k,l)]=(real)d11c;
                    MV.d12[I3(j,k,l)]=(real)d12c;
                    MV.d13[I3(j,k,l)]=(real)d13c; 
                    MV.d14[I3(j,k,l)]=(real)d14c;
                    MV.d15[I3(j,k,l)]=(real)d15c;
                    MV.d16[I3(j,k,l)]=(real)d16c;
                    MV.d17[I3(j,k,l)]=(real)d17c;
                    MV.d18[I3(j,k,l)]=(real)d18c;
                        

                }   
            }
            printf("proc %i reading %s [%.03f%%]   \r", current_process, f_name, 100.0f*real(j)/real(Nx-1) );
        }
        fflush(stream);  //flush all from disk
        long curposition = ftell (stream ); //get current possition
        printf("proc %i sends current position: %li\n", current_process, curposition);
        for (int l = 0; l < COM.totalrank; l++) 
        {
            if (l != current_process) 
            {
                MPI_Send(&curposition, 1, MPI_LONG, l, tag, MPI_COMM_WORLD); //send this info to all!
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fclose(stream);
        
    return Nx*Ny*Nz;

}





void write_out_file(std::string f_name, communication_variables COM, macroscopic_variables NV, control_variables CV, int Nx, int Ny, int Nz, char type, real dh, real xm, real ym, real zm)
{

    
    FILE *stream;
        
    int what=2;

    real dx=dh;
    real dy=dx;//linesize/real(N);
    real dz=dx;//linesize/real(K);

    std::stringstream ss; 
    ss << f_name;
    ss << COM.myrank;
    ss << ".pos";
    ss >> f_name;
    stream=fopen( f_name.c_str(), "w" );


    fprintf( stream, "View");
    fprintf( stream, " '");
    fprintf( stream, f_name.c_str());
    fprintf( stream, "' {\n");
    fprintf( stream, "TIME{0};\n");
    
    real k_scale=10.0;

 

    for(int j=1;j<Nx-1;j++)
    for(int k=1;k<Ny-1;k++)
    for(int l=1;l<Nz-1;l++)
    if(CV.bc[I3(j,k,l)]!=WALL)
    {
        if(type=='v')
        {
        
            real *ux_l=NV.ux;
            real *uy_l=NV.uy;
            real *uz_l=NV.uz;

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
        else
        {
            real par=0.0;
            real par_mmm=0.0;
            real par_pmm=0.0;
            real par_ppm=0.0;
            real par_ppp=0.0;
            real par_mpp=0.0;
            real par_mmp=0.0;
            real par_pmp=0.0;
            real par_mpm=0.0;
            
            if(type=='p')
            {
                real *ro=NV.rho;
                par_mmm=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l-1)]+ro[I3(j-1,k-1,k)]+ro[I3(j,k-1,l-1)]+ro[I3(j-1,j,l-1)]+ro[I3(j-1,k-1,l-1)]);
                par_pmm=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l-1)]+ro[I3(j+1,k-1,k)]+ro[I3(j,k-1,l-1)]+ro[I3(j+1,j,l-1)]+ro[I3(j+1,k-1,l-1)]);
                par_ppm=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l-1)]+ro[I3(j+1,k+1,k)]+ro[I3(j,k+1,l-1)]+ro[I3(j+1,j,l-1)]+ro[I3(j+1,k+1,l-1)]);
                par_ppp=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l+1)]+ro[I3(j+1,k+1,k)]+ro[I3(j,k+1,l+1)]+ro[I3(j+1,j,l+1)]+ro[I3(j+1,k+1,l+1)]);
                par_mpp=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l+1)]+ro[I3(j-1,k+1,k)]+ro[I3(j,k+1,l+1)]+ro[I3(j-1,j,l+1)]+ro[I3(j-1,k+1,l+1)]);
                par_mmp=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l+1)]+ro[I3(j-1,k-1,k)]+ro[I3(j,k-1,l+1)]+ro[I3(j-1,j,l+1)]+ro[I3(j-1,k-1,l+1)]);
                par_pmp=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l+1)]+ro[I3(j+1,k-1,k)]+ro[I3(j,k-1,l+1)]+ro[I3(j+1,j,l+1)]+ro[I3(j+1,k-1,l+1)]);
                par_mpm=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l-1)]+ro[I3(j-1,k+1,k)]+ro[I3(j,k+1,l-1)]+ro[I3(j-1,j,l-1)]+ro[I3(j-1,k+1,l-1)]);

            }
            if(type=='r')
            {
                real *ro=NV.abs_rot;
                par_mmm=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l-1)]+ro[I3(j-1,k-1,k)]+ro[I3(j,k-1,l-1)]+ro[I3(j-1,j,l-1)]+ro[I3(j-1,k-1,l-1)]);
                par_pmm=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l-1)]+ro[I3(j+1,k-1,k)]+ro[I3(j,k-1,l-1)]+ro[I3(j+1,j,l-1)]+ro[I3(j+1,k-1,l-1)]);
                par_ppm=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l-1)]+ro[I3(j+1,k+1,k)]+ro[I3(j,k+1,l-1)]+ro[I3(j+1,j,l-1)]+ro[I3(j+1,k+1,l-1)]);
                par_ppp=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l+1)]+ro[I3(j+1,k+1,k)]+ro[I3(j,k+1,l+1)]+ro[I3(j+1,j,l+1)]+ro[I3(j+1,k+1,l+1)]);
                par_mpp=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l+1)]+ro[I3(j-1,k+1,k)]+ro[I3(j,k+1,l+1)]+ro[I3(j-1,j,l+1)]+ro[I3(j-1,k+1,l+1)]);
                par_mmp=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l+1)]+ro[I3(j-1,k-1,k)]+ro[I3(j,k-1,l+1)]+ro[I3(j-1,j,l+1)]+ro[I3(j-1,k-1,l+1)]);
                par_pmp=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l+1)]+ro[I3(j+1,k-1,k)]+ro[I3(j,k-1,l+1)]+ro[I3(j+1,j,l+1)]+ro[I3(j+1,k-1,l+1)]);
                par_mpm=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l-1)]+ro[I3(j-1,k+1,k)]+ro[I3(j,k+1,l-1)]+ro[I3(j-1,j,l-1)]+ro[I3(j-1,k+1,l-1)]);

            }
  

            fprintf( stream, "SH(");
            
            fprintf( stream,"%lf,%lf, %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf",
                dh*j-0.5*dh+xm,dh*k-0.5*dh+ym, dh*l-0.5*dh+zm, 
                dh*j+0.5*dh+xm,dh*k-0.5*dh+ym,dh*l-0.5*dh+zm,
                dh*j+0.5*dh+xm,dh*k+0.5*dh+ym,dh*l-0.5*dh+zm,
                dh*j-0.5*dh+xm,dh*k+0.5*dh+ym,dh*l-0.5*dh+zm,
                dh*j-0.5*dh+xm,dh*k-0.5*dh+ym,dh*l+0.5*dh+zm,
                dh*j+0.5*dh+xm,dh*k-0.5*dh+ym,dh*l+0.5*dh+zm,
                dh*j+0.5*dh+xm,dh*k+0.5*dh+ym, dh*l+0.5*dh+zm,
                dh*j-0.5*dh+xm,dh*k+0.5*dh+ym,dh*l+0.5*dh+zm);



            fprintf( stream, "){");
            fprintf( stream,"%le,    %le, %le, %le, %le, %le, %le, %le};\n",par_mmm ,par_pmm,par_ppm,par_mpm,par_mmp ,par_pmp,par_ppp,par_mpp);

        }   
    }
     

    fprintf( stream, "};");

    fclose(stream);
}






void write_out_file_const(std::string f_name, communication_variables COM, macroscopic_variables NV, control_variables CV, int Nx, int Ny, int Nz, char type, real dh, real xm, real ym, real zm)
{
    FILE *stream;
    
    std::stringstream ss; 
    ss << f_name;
    ss << COM.myrank;
    ss << ".pos";
    ss >> f_name;
    stream=fopen( f_name.c_str(), "w" );

    
    fprintf( stream, "View");
    fprintf( stream, " '");
    fprintf( stream, f_name.c_str());
    fprintf( stream, "' {\n");
    fprintf( stream, "TIME{0};\n");
    
    real k_scale=1.0;

 

    for(int j=0;j<Nx;j++)
    for(int k=0;k<Ny;k++)
    for(int l=0;l<Nz;l++)
    {
        fprintf( stream, "SH(");
            
        fprintf( stream,"%lf,%lf, %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf",
            dh*j-0.5*dh+xm,dh*k-0.5*dh+ym, dh*l-0.5*dh+zm, 
            dh*j+0.5*dh+xm,dh*k-0.5*dh+ym,dh*l-0.5*dh+zm,
            dh*j+0.5*dh+xm,dh*k+0.5*dh+ym,dh*l-0.5*dh+zm,
            dh*j-0.5*dh+xm,dh*k+0.5*dh+ym,dh*l-0.5*dh+zm,
            dh*j-0.5*dh+xm,dh*k-0.5*dh+ym,dh*l+0.5*dh+zm,
            dh*j+0.5*dh+xm,dh*k-0.5*dh+ym,dh*l+0.5*dh+zm,
            dh*j+0.5*dh+xm,dh*k+0.5*dh+ym, dh*l+0.5*dh+zm,
            dh*j-0.5*dh+xm,dh*k+0.5*dh+ym,dh*l+0.5*dh+zm);

        real par=0;
        if(type=='b')
            par = (real) CV.bc[I3(j,k,l)];
        else if(type=='p')
            par = (real) NV.rho[I3(j,k,l)];
        fprintf( stream, "){");
        fprintf( stream,"%le,    %le, %le, %le, %le, %le, %le, %le};\n",par ,par,par,par,par ,par,par,par);
    }
    fprintf( stream, "};");

    fclose(stream);

}




void write_out_pos_file(std::string f_name, int what, communication_variables COM, macroscopic_variables NV, control_variables CV, real *ux_l, real *uy_l, real *uz_l)
{
int i,j,k;
    int Nx=COM.Nx, Ny=COM.Ny, Nz=COM.Nz;

    real dx=COM.dh;
    real dy=dx,dz=dx;
    real xm=COM.L0x;
    real ym=COM.L0y;
    real zm=COM.L0z;
    
    if(COM.myrank == MASTER) {
        FILE *stream = fopen( f_name.c_str(), "w" );
        fclose(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    FILE *stream;
    stream=fopen(f_name.c_str(), "a" );
    MPI_Barrier(MPI_COMM_WORLD);
    if (COM.myrank == MASTER) {  
        fprintf( stream, "View");
        fprintf( stream, " '");
        fprintf( stream, "%s", f_name.c_str());
        fprintf( stream, "' {\n");
        fprintf( stream, "TIME{0};\n");
    }
    int stride_local=stride;
    
    for (int current_process = 0;current_process < COM.totalrank ;current_process++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (COM.myrank != current_process) continue;
        int strideA=1, strideB=1, strideC=1, strideD=1, strideE=1, strideF=1;

        if(COM.FaceA=='C') strideA=1;
        if(COM.FaceB=='C') strideB=1;
        if(COM.FaceC=='C') strideC=1;
        if(COM.FaceD=='C') strideD=1;
        if(COM.FaceE=='C') strideE=1;
        if(COM.FaceF=='C') strideF=1;

        for(int j=strideA;j<Nx-strideB;j++){
            for(int k=strideC;k<Ny-strideD;k++){
                for(int l=strideE;l<Nz-strideF;l++){
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
                    if(what==2){

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

                    }

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
                        fprintf(stream, "%f,%f,%f,",par_x_mmm,par_y_mmm,par_z_mmm);
                        fprintf(stream, "%f,%f,%f,",par_x_pmm,par_y_pmm,par_z_pmm);
                        fprintf(stream, "%f,%f,%f,",par_x_ppm,par_y_ppm,par_z_ppm);
                        fprintf(stream, "%f,%f,%f,",par_x_mpm,par_y_mpm,par_z_mpm);
                        fprintf(stream, "%f,%f,%f,",par_x_mmp,par_y_mmp,par_z_mmp);
                        fprintf(stream, "%f,%f,%f,",par_x_pmp,par_y_pmp,par_z_pmp);
                        fprintf(stream, "%f,%f,%f,",par_x_ppp,par_y_ppp,par_z_ppp);
                        fprintf(stream, "%f,%f,%f",par_x_mpp,par_y_mpp,par_z_mpp);
                        fprintf(stream, "};\n");
                    }
                    else if(what==1){
                        fprintf( stream,"{");
                        fprintf(stream, "%f,%f,%f,",par_x,par_y,par_z);
                        fprintf(stream, "%f,%f,%f,",par_x,par_y,par_z);
                        fprintf(stream, "%f,%f,%f,",par_x,par_y,par_z);
                        fprintf(stream, "%f,%f,%f,",par_x,par_y,par_z);
                        fprintf(stream, "%f,%f,%f,",par_x,par_y,par_z);
                        fprintf(stream, "%f,%f,%f,",par_x,par_y,par_z);
                        fprintf(stream, "%f,%f,%f,",par_x,par_y,par_z);
                        fprintf(stream, "%f,%f,%f",par_x,par_y,par_z);
                        fprintf(stream, "};\n");
                    }
                   

                }
            }
            printf("proc %i wrighting %s [%.03f%%]   \r", current_process, f_name.c_str(), 100.0f*real(j)/real(Nx-2) );
            fflush(stdout);
        }
        fflush(stream);  //flush all to disk
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(COM.myrank == MASTER)
        fprintf( stream, "};");

    fclose( stream );
    fflush(stream);
    MPI_Barrier(MPI_COMM_WORLD);    
  
    if(COM.myrank == MASTER)
        printf("%s output done. \n",f_name.c_str());

    

}





void write_out_pos_file(std::string f_name, int what, communication_variables COM, macroscopic_variables NV, control_variables CV, real *ro)
{
int i,j,k;
    int Nx=COM.Nx, Ny=COM.Ny, Nz=COM.Nz;

    real dx=COM.dh;
    real dh=COM.dh;
    real dy=dx,dz=dx;
    real xm=COM.L0x;
    real ym=COM.L0y;
    real zm=COM.L0z;
    if(COM.myrank == MASTER) {
        FILE *stream = fopen( f_name.c_str(), "w" );
        fclose(stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    FILE *stream;
    stream=fopen(f_name.c_str(), "a" );
    MPI_Barrier(MPI_COMM_WORLD);
    if (COM.myrank == MASTER) {  
        fprintf( stream, "View");
        fprintf( stream, " '");
        fprintf( stream, "%s", f_name.c_str());
        fprintf( stream, "' {\n");
        fprintf( stream, "TIME{0};\n");
    }
    int stride_local=stride;
    
    for (int current_process = 0;current_process < COM.totalrank ;current_process++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (COM.myrank != current_process) continue;
        int strideA=1, strideB=1, strideC=1, strideD=1, strideE=1, strideF=1;

        if(COM.FaceA=='C') strideA=2;
        if(COM.FaceB=='C') strideB=2;
        if(COM.FaceC=='C') strideC=2;
        if(COM.FaceD=='C') strideD=2;
        if(COM.FaceE=='C') strideE=2;
        if(COM.FaceF=='C') strideF=2;

        for(int j=strideA;j<Nx-strideB;j++)
        {
            for(int k=strideC;k<Ny-strideD;k++)
            {
                for(int l=strideE;l<Nz-strideF;l++)
                {
                    real par=0.0;
                    real par_mmm=0.0;
                    real par_pmm=0.0;
                    real par_ppm=0.0;
                    real par_ppp=0.0;
                    real par_mpp=0.0;
                    real par_mmp=0.0;
                    real par_pmp=0.0;
                    real par_mpm=0.0;
                    

                    par_mmm=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l-1)]+ro[I3(j-1,k-1,k)]+ro[I3(j,k-1,l-1)]+ro[I3(j-1,j,l-1)]+ro[I3(j-1,k-1,l-1)]);
                    par_pmm=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l-1)]+ro[I3(j+1,k-1,k)]+ro[I3(j,k-1,l-1)]+ro[I3(j+1,j,l-1)]+ro[I3(j+1,k-1,l-1)]);
                    par_ppm=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l-1)]+ro[I3(j+1,k+1,k)]+ro[I3(j,k+1,l-1)]+ro[I3(j+1,j,l-1)]+ro[I3(j+1,k+1,l-1)]);
                    par_ppp=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l+1)]+ro[I3(j+1,k+1,k)]+ro[I3(j,k+1,l+1)]+ro[I3(j+1,j,l+1)]+ro[I3(j+1,k+1,l+1)]);
                    par_mpp=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l+1)]+ro[I3(j-1,k+1,k)]+ro[I3(j,k+1,l+1)]+ro[I3(j-1,j,l+1)]+ro[I3(j-1,k+1,l+1)]);
                    par_mmp=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l+1)]+ro[I3(j-1,k-1,k)]+ro[I3(j,k-1,l+1)]+ro[I3(j-1,j,l+1)]+ro[I3(j-1,k-1,l+1)]);
                    par_pmp=0.125f*(ro[I3(j,k,l)]+ro[I3(j+1,k,l)]+ro[I3(j,k-1,l)]+ro[I3(j,k,l+1)]+ro[I3(j+1,k-1,k)]+ro[I3(j,k-1,l+1)]+ro[I3(j+1,j,l+1)]+ro[I3(j+1,k-1,l+1)]);
                    par_mpm=0.125f*(ro[I3(j,k,l)]+ro[I3(j-1,k,l)]+ro[I3(j,k+1,l)]+ro[I3(j,k,l-1)]+ro[I3(j-1,k+1,k)]+ro[I3(j,k+1,l-1)]+ro[I3(j-1,j,l-1)]+ro[I3(j-1,k+1,l-1)]);

   
          

                    fprintf( stream, "SH(");
                    
                    fprintf( stream,"%lf,%lf, %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf, %lf,%lf,  %lf",
                        dh*j-0.5*dh+xm,dh*k-0.5*dh+ym, dh*l-0.5*dh+zm, 
                        dh*j+0.5*dh+xm,dh*k-0.5*dh+ym,dh*l-0.5*dh+zm,
                        dh*j+0.5*dh+xm,dh*k+0.5*dh+ym,dh*l-0.5*dh+zm,
                        dh*j-0.5*dh+xm,dh*k+0.5*dh+ym,dh*l-0.5*dh+zm,
                        dh*j-0.5*dh+xm,dh*k-0.5*dh+ym,dh*l+0.5*dh+zm,
                        dh*j+0.5*dh+xm,dh*k-0.5*dh+ym,dh*l+0.5*dh+zm,
                        dh*j+0.5*dh+xm,dh*k+0.5*dh+ym, dh*l+0.5*dh+zm,
                        dh*j-0.5*dh+xm,dh*k+0.5*dh+ym,dh*l+0.5*dh+zm);



                    fprintf( stream, "){");
                    fprintf( stream,"%le,    %le, %le, %le, %le, %le, %le, %le};\n",par_mmm ,par_pmm,par_ppm,par_mpm,par_mmp ,par_pmp,par_ppp,par_mpp);

                           

                }
            }
        }
        fflush(stream);  //flush all to disk
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(COM.myrank == MASTER)
        fprintf( stream, "};");

    fclose( stream );
    fflush(stream);
    MPI_Barrier(MPI_COMM_WORLD);    
  
    if(COM.myrank == MASTER)
        printf("%s output done. \n",f_name.c_str());

    

}
