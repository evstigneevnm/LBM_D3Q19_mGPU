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



#pragma once

#include <stdlib.h>
#include <stdexcept>
#include <string>

#include "smart_array_defines.h"
#include "cuda_safe_call.h"
#include "cuda_support.h"


template<typename T, storage_architecure storage>
class smart_array
{
public:
    smart_array()
    {
        data = NULL;
        size = 0;
        std::cout << "constructor" << std::endl;
    }
    smart_array(const smart_array &SA)
    { 
        *this=SA; 
        own=false; 
    }

    ~smart_array()
    {
        if((data!=NULL)&&own){
            if(storage==KEEP_HOST)
            {
                
                free(data);    
                data = NULL;
                std::cout << "distructor host" << std::endl;    
            }
            if(storage==KEEP_DEVICE)
            {
                CUDA_SAFE_CALL(cudaFree(data));
                data = NULL;
                std::cout << "distructor device" << std::endl;  
            }
            size=0;
        }
        std::cout << "distructor void" << std::endl;
    }

    void init(size_t size_l)
    {
        if(data==NULL)
        {            
            if(storage==KEEP_HOST)
            {
                data=host_allocate<T>(size_l);
            }
            if(storage==KEEP_DEVICE)
            {
                data=device_allocate<T>(size_l);
            }
            size=size_l;
            own = true;
        }
    }
    void release()
    {
        if((data!=NULL)&&own)
        {
            if(storage==KEEP_HOST)
            {
                free(data);
                data = NULL;
                std::cout << "released host" << std::endl;
            }
            if(storage==KEEP_DEVICE)
            {
                CUDA_SAFE_CALL(cudaFree(data));
                data = NULL;
                std::cout << "released device" << std::endl;
            }         
            size=0;
        }
    }

    size_t get_size()
    {
        return size;
    }

    //here we keep the main pointer
    T *data;

    __DEVICE_TAG__ T & operator [] (int index) 
    { 
        return data[index]; 
    }

protected:    
    size_t size;
    bool own;
    
    __DEVICE_TAG__ void assign(const smart_array &SA)
    {
        *this = SA; 
        own = false;
    }
    

};



template<typename T>
class clever_arrays: public smart_array<T, KEEP_HOST>
{
public: 
    typedef smart_array<T, KEEP_HOST> host_parent;
    clever_arrays()
    {
        size=0;
    }


    void init(size_t size_l)
    {
        host_parent::init(size_l);
        dev.init(size_l);
        size=size_l;
    }
    void release()
    {
        dev.release();
        host_parent::release();
        size=0;
    }
    void sync_2_device()
    {
        if(size>0)
            host_2_device_cpy<T>(dev.data, host_parent::data, size);
        else
            throw std::runtime_error(std::string("clever array: sync_2_device withought initialization"));
    }
    void sync_2_host()
    {
        if(size>0)
            device_2_host_cpy<T>(host_parent::data, dev.data, size);
        else
            throw std::runtime_error(std::string("clever array: sync_2_host withought initialization"));       
    }

    smart_array<T, KEEP_DEVICE> dev;

protected:
    size_t size;


};