#include "cuda_support.h"


#if __DEVICE_EMULATION__

int InitCUDA(int PCI_ID){return true;}

#else

int InitCUDA(int PCI_ID)
{

    
    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0)
    {
        fprintf(stderr, "There is no compartable device found.\n");
        return -1;
    }
    
    int deviceNumber=0;
    int deviceNumberTemp=0;
    
    if(count>1)
    {

        if(PCI_ID==-1)
        {
            for(i = 0; i < count; i++) 
            {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, i);
                printf( "#%i:   %s, pci-bus id:%i %i %i \n", i, &deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
            }            
            printf("Device number for it to use>>>\n",i);
            scanf("%i", &deviceNumberTemp);
        }
        else
        {
            cudaDeviceProp deviceProp;
            for(int j=0;j<count;j++)
            {
                cudaGetDeviceProperties(&deviceProp, j);
                if(deviceProp.pciBusID==PCI_ID)
                    deviceNumberTemp = j;
            }

            //printf("Using device number %i\n",deviceNumberTemp);
        }
        deviceNumber=deviceNumberTemp;
    
    }
    else
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceNumber);
        printf( "#%i:   %s, pci-bus id:%i %i %i \n", deviceNumber, &deviceProp,deviceProp.pciBusID,deviceProp.pciDeviceID,deviceProp.pciDomainID);
        printf( "       using it...\n");    
    }

    cudaSetDevice(deviceNumber);
    
    return deviceNumber;
}
#endif





// void probe_boundary()
// {




    
// }