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
#include <iostream>

#include "umpire/ResourceManager.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  std::cout << "Available allocators: ";
  for (auto s : rm.getAvailableAllocators()){
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  // Device constant memory allocator
  alloc = rm.getAllocator("DEVICE_CONST");

  // allocating device memory
  void* test = alloc.allocate(100);
  alloc.deallocate(test);

  std::cout << "end of test." << std::endl;

  return 0;
}
