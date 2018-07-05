//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaConstMemAllocator_HPP
#define UMPIRE_CudaConstMemAllocator_HPP

#include <cuda_runtime_api.h>
// #include "umpire/resource/ConstantMemoryResource.hpp"

__constant__ char constant_memory[64*1024];

namespace umpire {
namespace alloc {

/*!
 * \brief Uses cudaMalloc and cudaFree to allocate and deallocate memory on
 *        NVIDIA GPUs.
 */
struct CudaConstMemAllocator {
  /*!
   * \brief Allocate bytes of memory using cudaMalloc
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(size_t size)
  {
    // char type, return current available pointer. check size.
    if (size > 64*1024)
    {
       UMPIRE_ERROR("CudaConstMemAllocator required bytes = " << size << " ) larger than MAX constant size: " << 64*1024 << " bytes" );
    }

    void* ptr = nullptr;
    cudaError_t error = cudaGetSymbolAddress((void**)&ptr, constant_memory);
    UMPIRE_LOG(Debug, "(bytes=" << size << ") returning " << ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaGetSymbolAddress( bytes = " << size << " ) failed with error: " << cudaGetErrorString(error));
    } else {
      return ptr;
    }
  }

  /*!
   * \brief Deallocate memory using cudaFree.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::Exception if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    // Noting need to do.
    return;
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocAllocator_HPP
