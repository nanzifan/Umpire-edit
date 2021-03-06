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
#ifndef UMPIRE_DeviceConstResourceFactory_HPP
#define UMPIRE_DeviceConstResourceFactory_HPP

#include "umpire/resource/MemoryResourceFactory.hpp"

// #include "umpire/resource/DeviceResourceFactory.hpp"

// #include "umpire/resource/ConstantMemoryResource.hpp"
// #include "umpire/alloc/CudaConstMallocAllocator.hpp"
namespace umpire {
namespace resource {


/*!
 * \brief Factory class for constructing MemoryResource objects that use GPU
 * memory.
 */
class DeviceConstResourceFactory :
  public MemoryResourceFactory
{
  bool isValidMemoryResourceFor(const std::string& name);

  std::shared_ptr<MemoryResource> create(const std::string& name, int id);
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_DeviceResourceFactory_HPP
