#pragma once

#include "Macro.h"
#include "cuda_safe_call.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <stdarg.h>

//TODO: put throw everywhere!!!


int InitCUDA(int GPU_number=-1);



template <class MyType>
void host_2_device_cpy(MyType* device, MyType* host, int Nx, int Ny, int Nz)
{
    int mem_size=sizeof(MyType)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMemcpy(device, host, mem_size, cudaMemcpyHostToDevice));

}

template <class MyType>
void device_2_host_cpy(MyType* host, MyType* device, int Nx, int Ny, int Nz)
{
    int mem_size=sizeof(MyType)*Nx*Ny*Nz;
    CUDA_SAFE_CALL(cudaMemcpy(host, device, mem_size, cudaMemcpyDeviceToHost));
}




template <class MyType>
MyType* device_allocate(int Nx, int Ny, int Nz)
{
    MyType* m_device;
    int mem_size=sizeof(MyType)*Nx*Ny*Nz;
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_device, mem_size));
  

    return m_device;    
}


template <class MyType>
MyType* device_allocate(int size)
{
    MyType* m_device;
    int mem_size=sizeof(MyType)*size;
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_device, mem_size));
  

    return m_device;    
}


template <class MyType>
void device_allocate_all(int Nx, int Ny, int Nz, int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        MyType** value=va_arg(ap, MyType**); /* Increments ap to the next argument. */
        MyType* temp=device_allocate<MyType>(Nx, Ny, Nz);
        value[0]=temp;      
    }
    va_end(ap);

}

template <class MyType>
void device_deallocate_all(int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        MyType* value=va_arg(ap, MyType*); /* Increments ap to the next argument. */
        CUDA_SAFE_CALL(cudaFree(value));
    }
    va_end(ap);
}


// host operations
template <class MyType>
MyType* host_allocate(int Nx, int Ny, int Nz)
{
    
    int size=(Nx)*(Ny)*(Nz);
    MyType* array;
    array=(MyType*)malloc(sizeof(MyType)*size);
    if ( !array )
    {
        fprintf(stderr,"\n unable to allocate memory!\n");
        exit(-1);
    }
    for(int j=0;j<size;j++)
        array[j]=(MyType)0;

    return array;
}

// host operations
template <class MyType>
MyType* host_allocate(int size)
{
    
    MyType* array;
    array=(MyType*)malloc(sizeof(MyType)*size);
    if ( !array )
    {
        fprintf(stderr,"\n unable to allocate memory!\n");
        exit(-1);
    }
    for(int j=0;j<size;j++)
        array[j]=(MyType)0;

    return array;
}


template <class MyType>
void host_allocate_all(int Nx, int Ny, int Nz, int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        MyType** value= va_arg(ap, MyType**); /* Increments ap to the next argument. */
        MyType* temp=host_allocate<MyType>(Nx, Ny, Nz);
        value[0]=temp;      
    }
    va_end(ap);

}

template <class MyType>
void host_deallocate_all(int count, ...)
{

    va_list ap;
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for(int j = 0; j < count; j++)
    {
        MyType* value= va_arg(ap, MyType*); /* Increments ap to the next argument. */
        free(value);
    }
    va_end(ap);

}

