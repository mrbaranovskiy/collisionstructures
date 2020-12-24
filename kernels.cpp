#pragma once

#include "cuda_common.h"
#include "linearalgebra.h"
#include "kernels.h"

__global__ auto resizeToFit(double output[4], Vector3d* const vertices, Matrix4x4d* axis2modelInv, const int N) -> void
{
	/*__shared__ Matrix4x4d m2axis;
	m2axis = *axis2modelInv;*/

	extern __shared__ double aminz[];
	extern __shared__ double amaxz[];
	extern __shared__ double aminSqr[];
	extern __shared__ double amaxSqr[];

	TID tid = threadIdx.x;
	TID i = blockDim.x * blockIdx.x + tid;

	//doing grid loop using stride method.
	double minz = INFINITY;
	double maxz = -INFINITY;
	double minSqrt = INFINITY;
	double maxSqrt = -INFINITY;

	for (unsigned int s = i; s < N; s += blockDim.x * gridDim.x)
	{
		//const auto point = axis2modelInv->transform(vertices[i]);
		//const auto point = transform(axis2modelInv, &vertices[i]);
		Vector3d point{ 2,3,4 };
		const double z = point.Z;
		minz = thrust::min(minz, z);
		maxz = thrust::max(maxz, z);

		const double sqrt = sqr_len(point.X, point.Y);
		minSqrt = thrust::min(minSqrt, sqrt);
		maxSqrt = thrust::max(maxSqrt, sqrt);
	}

	aminz[tid] = minz;
	amaxz[tid] = maxz;
	aminSqr[tid] = minSqrt;
	amaxSqr[tid] = maxSqrt;

	for (TID halfBlock = blockDim.x / 2; halfBlock > 0; halfBlock >>= 1)
	{
		if (tid < halfBlock && tid + halfBlock < N)
		{
			minz = thrust::min(aminz[tid], aminz[tid + halfBlock]);
			maxz = thrust::max(amaxz[tid], amaxz[tid + halfBlock]);
			minSqrt = thrust::max(aminSqr[tid], aminSqr[tid + halfBlock]);
			maxSqrt = thrust::max(amaxSqr[tid], amaxSqr[tid + halfBlock]);
		}

		__syncthreads();
	}

	output[0] = minz;
	output[1] = maxz;
	output[2] = minSqrt;
	output[3] = maxSqrt;
}



void resizeToFitCPU(double output[4], Vector3d* const vertices, Matrix4x4d* axis2modelInv, const int N)
{
	resizeToFit <<<1, 16>>> (output, vertices, axis2modelInv, N);
}
