#pragma once

#include <thrust/device_vector.h>
#include "cuda_runtime.h"

//#include <crt/host_defines.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <thrust/host_vector.h>

#ifndef __CUDACC__  
#define __CUDACC__
#endif

//#include <device_functions.h>
#include <device_launch_parameters.h>

//#include <math_functions.h>


#define CALLABLE __device__ __host__
#define KERNEL __global__

#define FMOD(A,B) (A & (B-1))

#define TID UINT
