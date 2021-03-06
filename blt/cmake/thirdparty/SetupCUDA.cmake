################################
# CUDA
################################
set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")

enable_language(CUDA)

############################################################
# Map Legacy FindCUDA variables to native cmake variables
############################################################
# if we are linking with NVCC, define the link rule here
# Note that some mpi wrappers might have things like -Wl,-rpath defined, which when using 
# FindMPI can break nvcc. In that case, you should set ENABLE_FIND_MPI to Off and control
# the link using CMAKE_CUDA_LINK_FLAGS. -Wl,-rpath, equivalent would be -Xlinker -rpath -Xlinker
if (CUDA_LINK_WITH_NVCC)
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_CUDA_FLAG "-Xlinker -rpath -Xlinker")
  set(CMAKE_CUDA_LINK_EXECUTABLE
    "${CMAKE_CUDA_COMPILER} <CMAKE_CUDA_LINK_FLAGS>  <FLAGS>  <LINK_FLAGS>  <OBJECTS> -o <TARGET>  <LINK_LIBRARIES>")
  # do a no-op for the device links - for some reason the device link library dependencies are only a subset of the 
  # executable link dependencies so the device link fails if there are any missing CUDA library dependencies. Since
  # we are doing a link with the nvcc compiler, the device link step is unnecessary .
  # Frustratingly, nvcc-link errors out if you pass it an empty file, so we have to first compile the empty file. 
  set(CMAKE_CUDA_DEVICE_LINK_LIBRARY "touch <TARGET>.cu ; ${CMAKE_CUDA_COMPILER} <CMAKE_CUDA_LINK_FLAGS> -std=c++11 -dc <TARGET>.cu -o <TARGET>")
  set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE "touch <TARGET>.cu ; ${CMAKE_CUDA_COMPILER} <CMAKE_CUDA_LINK_FLAGS> -std=c++11 -dc <TARGET>.cu -o <TARGET>")
endif()

find_package(CUDA REQUIRED)

message(STATUS "CUDA version:      ${CUDA_VERSION_STRING}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries:    ${CUDA_LIBRARIES}")

# don't propagate host flags - too easy to break stuff!
set (CUDA_PROPAGATE_HOST_FLAGS Off)
if (CMAKE_CXX_COMPILER)
  set (CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else ()
  set (CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
endif ()

if (ENABLE_CLANG_CUDA)
  set (clang_cuda_flags "-x cuda --cuda-gpu-arch=${BLT_CLANG_CUDA_ARCH} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")

  blt_register_library(NAME cuda
                       COMPILE_FLAGS ${clang_cuda_flags}
                       INCLUDES ${CUDA_INCLUDE_DIRS}
                       LIBRARIES ${CUDA_LIBRARIES}
                       DEFINES USE_CUDA)
else ()
  # depend on 'cuda', if you need to use cuda
  # headers, link to cuda libs, and need to run your source
  # through a cuda compiler (nvcc)
  blt_register_library(NAME cuda
                       INCLUDES ${CUDA_INCLUDE_DIRS}
                       LIBRARIES ${CUDA_LIBRARIES}
                       DEFINES USE_CUDA)

endif ()

# depend on 'cuda_runtime', if you only need to use cuda
# headers or link to cuda libs, but don't need to run your source
# through a cuda compiler (nvcc)
blt_register_library(NAME cuda_runtime
                     INCLUDES ${CUDA_INCLUDE_DIRS}
                     LIBRARIES ${CUDA_LIBRARIES}
                     DEFINES USE_CUDA)
