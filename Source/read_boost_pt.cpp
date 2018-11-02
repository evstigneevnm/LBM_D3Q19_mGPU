#include "read_boost_pt.h"


int extractIntegerWords(std::string str) 
{ 
    std::stringstream ss;     

    ss << str; 
  
    std::string temp; 
    int found;
    while (!ss.eof()) { 
        
        ss >> temp; 
        if (std::stringstream(temp) >> found) 
            return found;
        temp = ""; 
    } 
    return -1;
} 


void dispatch_info_data(const char *property_tree_file_name, communication_variables *COM, std::string control_file_2_read, std::string control_file_2_write)
{
    if(COM->myrank==MASTER)
    {
        std::cout << "\n Using configure file: " << property_tree_file_name << std::endl;
        boost::property_tree::ptree all_properties;
        boost::property_tree::info_parser::read_info(property_tree_file_name, all_properties);
        
        control_file_2_read=all_properties.get<std::string>("input_control_file", "");
        control_file_2_write=all_properties.get<std::string>("output_control_file", "");


        int globalNx=all_properties.get<int>("Nx");
        int globalNy=all_properties.get<int>("Ny");
        int globalNz=all_properties.get<int>("Nz");
        real globalLx=all_properties.get<real>("Lx");
        real globalLy=all_properties.get<real>("Ly");
        real globalLz=all_properties.get<real>("Lz");
        real Reynolds=all_properties.get<real>("Reynolds");
        int timesteps=all_properties.get<int>("timesteps");
        char MPI_sections = all_properties.get<char>("MPI_sections");
        unsigned int MPI_fraction = all_properties.get<unsigned int>("MPI_fraction", 100); 
        for(int pr=0;pr<COM->totalrank;pr++)
        {
            std::string block_name = "Block" + std::to_string(pr);
            
            std::string L_fraction_name = block_name + ".L_fraction";
            int L_fraction = all_properties.get<int>(L_fraction_name);
            
            std::string Compute_name = block_name + ".Compute";
            std::string Compute_device = all_properties.get<std::string>(Compute_name);
            int device = 0;
            if(Compute_device == "GPU")
                device = 1;
            if(Compute_device == "PHI")
                device = 2;

            std::string Blade_ID_name = block_name + ".Blade_ID";
            int Blade_ID = all_properties.get<int>(Blade_ID_name, 0);
            
            std::string Device_PCI_ID_name = block_name + ".Device_PCI_ID";
            int Device_PCI_ID = all_properties.get<int>(Device_PCI_ID_name,-1);  //ignored for CPU
            
            std::string L_start_name = block_name + ".L_start";
            int L_start = all_properties.get<int>(L_start_name);

            int Nx=(MPI_sections=='X')?(int(globalNx*L_fraction/MPI_fraction)):globalNx;
            int Ny=(MPI_sections=='Y')?(int(globalNy*L_fraction/MPI_fraction)):globalNy;
            int Nz=(MPI_sections=='Z')?(int(globalNz*L_fraction/MPI_fraction)):globalNz;
            real dh = (MPI_sections=='X')?globalLz/globalNz:(MPI_sections=='Y')?globalLx/globalNx:(MPI_sections=='Z')?globalLy/globalNy:globalLz/globalNz;
            real Lx=(MPI_sections=='X')?(real(globalLx*L_fraction/MPI_fraction)):globalLx;
            real Ly=(MPI_sections=='Y')?(real(globalLy*L_fraction/MPI_fraction)):globalLy;
            real Lz=(MPI_sections=='Z')?(real(globalLz*L_fraction/MPI_fraction)):globalLz;
            real L0x=(MPI_sections=='X')?(real(globalLx*L_start/MPI_fraction)-2*dh*pr):0.0;
            real L0y=(MPI_sections=='Y')?(real(globalLy*L_start/MPI_fraction)-2*dh*pr):0.0;
            real L0z=(MPI_sections=='Z')?(real(globalLz*L_start/MPI_fraction)-2*dh*pr):0.0;            

            //printf("dh = %f, L0z = %f, proc = %i\n",dh, L0z, pr);

            int FaceAproc = -1;
            std::string FaceA = all_properties.get<std::string>(block_name + ".FaceA");
            if(FaceA.size()>1)
            {
                FaceAproc = extractIntegerWords(FaceA);
                FaceA="C";
            }

            int FaceBproc = -1;
            std::string FaceB = all_properties.get<std::string>(block_name + ".FaceB");
            if(FaceB.size()>1)
            {
                FaceBproc = extractIntegerWords(FaceB);
                FaceB="C";
            }

            int FaceCproc = -1;
            std::string FaceC = all_properties.get<std::string>(block_name + ".FaceC");
            if(FaceC.size()>1)
            {
                FaceCproc = extractIntegerWords(FaceC);
                FaceC="C";
            }

            int FaceDproc = -1;
            std::string FaceD = all_properties.get<std::string>(block_name + ".FaceD");
            if(FaceD.size()>1)
            {
                FaceDproc = extractIntegerWords(FaceD);
                FaceD="C";
            }

            int FaceEproc = -1;
            std::string FaceE = all_properties.get<std::string>(block_name + ".FaceE");
            if(FaceE.size()>1)
            {
                FaceEproc = extractIntegerWords(FaceE);
                FaceE="C";
            }

            int FaceFproc = -1;
            std::string FaceF = all_properties.get<std::string>(block_name + ".FaceF");
            if(FaceF.size()>1)
            {
                FaceFproc = extractIntegerWords(FaceF);
                FaceF="C";
            }


            if(pr==MASTER)
            {
                COM->Nx=Nx;
                COM->Ny=Ny;
                COM->Nz=Nz;
                COM->Lx=Lx;
                COM->Ly=Ly;
                COM->Lz=Lz;  
                COM->L0x=L0x;
                COM->L0y=L0y;
                COM->L0z=L0z;  
                COM->MPI_sections=MPI_sections;
                COM->Reynolds=Reynolds;
                COM->timesteps=timesteps;
                COM->Blade_ID = Blade_ID;
                COM->device = device;
                COM->PCI_ID = Device_PCI_ID;
                COM->device_ID = -1;
                COM->FaceA=FaceA.c_str()[0];
                COM->FaceAproc=FaceAproc;
                COM->FaceB=FaceB.c_str()[0];
                COM->FaceBproc=FaceBproc;
                COM->FaceC=FaceC.c_str()[0];
                COM->FaceCproc=FaceCproc;
                COM->FaceD=FaceD.c_str()[0];
                COM->FaceDproc=FaceDproc;                
                COM->FaceE=FaceE.c_str()[0];
                COM->FaceEproc=FaceEproc;                
                COM->FaceF=FaceF.c_str()[0];
                COM->FaceFproc=FaceFproc;                
                COM->dh=dh;
            }
            else
            {
                //send data to other processes
                MPI_Send(&Nx, 1, MPI_INT, pr, 0, MPI_COMM_WORLD); 
                MPI_Send(&Ny, 1, MPI_INT, pr, 1, MPI_COMM_WORLD); 
                MPI_Send(&Nz, 1, MPI_INT, pr, 2, MPI_COMM_WORLD); 
                MPI_Send(&Lx, 1, MPI_real, pr, 3, MPI_COMM_WORLD); 
                MPI_Send(&Ly, 1, MPI_real, pr, 4, MPI_COMM_WORLD); 
                MPI_Send(&Lz, 1, MPI_real, pr, 5, MPI_COMM_WORLD); 
                MPI_Send(&L0x, 1, MPI_real, pr, 6, MPI_COMM_WORLD); 
                MPI_Send(&L0y, 1, MPI_real, pr, 7, MPI_COMM_WORLD); 
                MPI_Send(&L0z, 1, MPI_real, pr, 8, MPI_COMM_WORLD); 
                MPI_Send(&Blade_ID, 1, MPI_INT, pr, 9, MPI_COMM_WORLD);  
                MPI_Send(&device, 1, MPI_INT, pr, 10, MPI_COMM_WORLD);
                MPI_Send(&Device_PCI_ID, 1, MPI_INT, pr, 11, MPI_COMM_WORLD);
                MPI_Send(&FaceA.c_str()[0], 1, MPI_CHAR, pr, 12, MPI_COMM_WORLD);
                MPI_Send(&FaceAproc, 1, MPI_INT, pr, 13, MPI_COMM_WORLD);
                MPI_Send(&FaceB.c_str()[0], 1, MPI_CHAR, pr, 14, MPI_COMM_WORLD);
                MPI_Send(&FaceBproc, 1, MPI_INT, pr, 15, MPI_COMM_WORLD);
                MPI_Send(&FaceC.c_str()[0], 1, MPI_CHAR, pr, 16, MPI_COMM_WORLD);
                MPI_Send(&FaceCproc, 1, MPI_INT, pr, 17, MPI_COMM_WORLD);
                MPI_Send(&FaceD.c_str()[0], 1, MPI_CHAR, pr, 18, MPI_COMM_WORLD);
                MPI_Send(&FaceDproc, 1, MPI_INT, pr, 19, MPI_COMM_WORLD);
                MPI_Send(&FaceE.c_str()[0], 1, MPI_CHAR, pr, 20, MPI_COMM_WORLD);
                MPI_Send(&FaceEproc, 1, MPI_INT, pr, 21, MPI_COMM_WORLD);
                MPI_Send(&FaceF.c_str()[0], 1, MPI_CHAR, pr, 22, MPI_COMM_WORLD);
                MPI_Send(&FaceFproc, 1, MPI_INT, pr, 23, MPI_COMM_WORLD);
                MPI_Send(&Reynolds, 1, MPI_real, pr, 24, MPI_COMM_WORLD);
                MPI_Send(&timesteps, 1, MPI_INT, pr, 25, MPI_COMM_WORLD);
                MPI_Send(&MPI_sections, 1, MPI_CHAR, pr, 26, MPI_COMM_WORLD);
                MPI_Send(&dh, 1, MPI_real, pr, 27, MPI_COMM_WORLD);
            }   


        }

    }
    else
    {
        //not master proc-s just store sent data
        MPI_Recv(&COM->Nx, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);  
        MPI_Recv(&COM->Ny, 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Nz, 1, MPI_INT, MASTER, 2, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Lx, 1, MPI_real, MASTER, 3, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Ly, 1, MPI_real, MASTER, 4, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Lz, 1, MPI_real, MASTER, 5, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->L0x, 1, MPI_real, MASTER, 6, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->L0y, 1, MPI_real, MASTER, 7, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->L0z, 1, MPI_real, MASTER, 8, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Blade_ID, 1, MPI_INT, MASTER, 9, MPI_COMM_WORLD,MPI_STATUS_IGNORE);  
        MPI_Recv(&COM->device, 1, MPI_INT, MASTER, 10, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->PCI_ID, 1, MPI_INT, MASTER, 11, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceA, 1, MPI_CHAR, MASTER, 12, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceAproc, 1, MPI_INT, MASTER, 13, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceB, 1, MPI_CHAR, MASTER, 14, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceBproc, 1, MPI_INT, MASTER, 15, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceC, 1, MPI_CHAR, MASTER, 16, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceCproc, 1, MPI_INT, MASTER, 17, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceD, 1, MPI_CHAR, MASTER, 18, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceDproc, 1, MPI_INT, MASTER, 19, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceE, 1, MPI_CHAR, MASTER, 20, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceEproc, 1, MPI_INT, MASTER, 21, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceF, 1, MPI_CHAR, MASTER, 22, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceFproc, 1, MPI_INT, MASTER, 23, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->Reynolds, 1, MPI_real, MASTER, 24, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->timesteps, 1, MPI_INT, MASTER, 25, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->MPI_sections, 1, MPI_CHAR, MASTER, 26, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->dh, 1, MPI_real, MASTER, 27, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
    }


    MPI_Barrier(MPI_COMM_WORLD);
}



void read_info_data(const char *property_tree_file_name, communication_variables *COM, std::string &control_file_2_read, std::string &control_file_2_write)
{

    if(COM->myrank==MASTER)
    {
        std::cout << "\n Using configure file: " << property_tree_file_name << std::endl;
    }
    boost::property_tree::ptree all_properties;
    boost::property_tree::info_parser::read_info(property_tree_file_name, all_properties);
    
    control_file_2_read=all_properties.get<std::string>("input_control_file", "");
    control_file_2_write=all_properties.get<std::string>("output_control_file", "");

    std::cout << control_file_2_read << "\n";

    int globalNx=all_properties.get<int>("Nx");
    int globalNy=all_properties.get<int>("Ny");
    int globalNz=all_properties.get<int>("Nz");
    real globalLx=all_properties.get<real>("Lx");
    real globalLy=all_properties.get<real>("Ly");
    real globalLz=all_properties.get<real>("Lz");
    real Reynolds=all_properties.get<real>("Reynolds");
    int timesteps=all_properties.get<int>("timesteps");
    char MPI_sections = all_properties.get<char>("MPI_sections");
    unsigned int MPI_fraction = all_properties.get<unsigned int>("MPI_fraction", 100); 
    // for(int pr=0;pr<COM->totalrank;pr++)
    // {
    int pr=COM->myrank;
    std::string block_name = "Block" + std::to_string(pr);
        
    std::string L_fraction_name = block_name + ".L_fraction";
    int L_fraction = all_properties.get<int>(L_fraction_name);
        
    std::string Compute_name = block_name + ".Compute";
    std::string Compute_device = all_properties.get<std::string>(Compute_name);
    int device = 0;
    if(Compute_device == "GPU")
        device = 1;
    if(Compute_device == "PHI")
        device = 2;

    std::string Blade_ID_name = block_name + ".Blade_ID";
    int Blade_ID = all_properties.get<int>(Blade_ID_name, 0);
        
    std::string Device_PCI_ID_name = block_name + ".Device_PCI_ID";
    int Device_PCI_ID = all_properties.get<int>(Device_PCI_ID_name,-1);  //ignored for CPU
        
    std::string L_start_name = block_name + ".L_start";
    int L_start = all_properties.get<int>(L_start_name);

    int Nx=(MPI_sections=='X')?(int(globalNx*L_fraction/MPI_fraction)):globalNx;
    int Ny=(MPI_sections=='Y')?(int(globalNy*L_fraction/MPI_fraction)):globalNy;
    int Nz=(MPI_sections=='Z')?(int(globalNz*L_fraction/MPI_fraction)):globalNz;
    real dh = (MPI_sections=='X')?globalLz/globalNz:(MPI_sections=='Y')?globalLx/globalNx:(MPI_sections=='Z')?globalLy/globalNy:globalLz/globalNz;
    real Lx=(MPI_sections=='X')?(real(globalLx*L_fraction/MPI_fraction)):globalLx;
    real Ly=(MPI_sections=='Y')?(real(globalLy*L_fraction/MPI_fraction)):globalLy;
    real Lz=(MPI_sections=='Z')?(real(globalLz*L_fraction/MPI_fraction)):globalLz;
    real L0x=(MPI_sections=='X')?(real(globalLx*L_start/MPI_fraction)-2*dh*pr):0.0;
    real L0y=(MPI_sections=='Y')?(real(globalLy*L_start/MPI_fraction)-2*dh*pr):0.0;
    real L0z=(MPI_sections=='Z')?(real(globalLz*L_start/MPI_fraction)-2*dh*pr):0.0;            


    int FaceAproc = -1;
    std::string FaceA = all_properties.get<std::string>(block_name + ".FaceA");
    if(FaceA.size()>1)
    {
        FaceAproc = extractIntegerWords(FaceA);
        FaceA="C";
    }

    int FaceBproc = -1;
    std::string FaceB = all_properties.get<std::string>(block_name + ".FaceB");
    if(FaceB.size()>1)
    {
        FaceBproc = extractIntegerWords(FaceB);
        FaceB="C";
    }

    int FaceCproc = -1;
    std::string FaceC = all_properties.get<std::string>(block_name + ".FaceC");
    if(FaceC.size()>1)
    {
        FaceCproc = extractIntegerWords(FaceC);
        FaceC="C";
    }

    int FaceDproc = -1;
    std::string FaceD = all_properties.get<std::string>(block_name + ".FaceD");
    if(FaceD.size()>1)
    {
        FaceDproc = extractIntegerWords(FaceD);
        FaceD="C";
    }

    int FaceEproc = -1;
    std::string FaceE = all_properties.get<std::string>(block_name + ".FaceE");
    if(FaceE.size()>1)
    {
        FaceEproc = extractIntegerWords(FaceE);
        FaceE="C";
    }

    int FaceFproc = -1;
    std::string FaceF = all_properties.get<std::string>(block_name + ".FaceF");
    if(FaceF.size()>1)
    {
        FaceFproc = extractIntegerWords(FaceF);
        FaceF="C";
    }

    COM->Nx=Nx;
    COM->Ny=Ny;
    COM->Nz=Nz;
    COM->Lx=Lx;
    COM->Ly=Ly;
    COM->Lz=Lz;  
    COM->L0x=L0x;
    COM->L0y=L0y;
    COM->L0z=L0z;  
    COM->MPI_sections=MPI_sections;
    COM->Reynolds=Reynolds;
    COM->timesteps=timesteps;
    COM->Blade_ID = Blade_ID;
    COM->device = device;
    COM->PCI_ID = Device_PCI_ID;
    COM->device_ID = -1;
    COM->FaceA=FaceA.c_str()[0];
    COM->FaceAproc=FaceAproc;
    COM->FaceB=FaceB.c_str()[0];
    COM->FaceBproc=FaceBproc;
    COM->FaceC=FaceC.c_str()[0];
    COM->FaceCproc=FaceCproc;
    COM->FaceD=FaceD.c_str()[0];
    COM->FaceDproc=FaceDproc;                
    COM->FaceE=FaceE.c_str()[0];
    COM->FaceEproc=FaceEproc;                
    COM->FaceF=FaceF.c_str()[0];
    COM->FaceFproc=FaceFproc;                
    COM->dh=dh;
   

}