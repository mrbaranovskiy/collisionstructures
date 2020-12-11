#pragma once

#include "linearalgebra.cuh"
#include "cuda_common.cuh"
#include <math_functions.h>

using namespace threeshape::linearalgebra;

namespace threeshape
{
	namespace kernels
	{

		/**
		 * \brief
		 * \param output 4-double array defines _minZ _maxZ _minSqrR _maxSqrR
		 * \param vertices
		 * \param axis2modelInv - Inverted axis to model.
		 * \param N
		 */
		KERNEL inline auto resizeToFit(double output[4], const Vector3d* vertices, const Matrix4x4d axis2modelInv, const int N) -> void
		{
			__shared__ Matrix4x4d m2axis = axis2modelInv;

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
				const auto point = m2axis.transform(vertices[i]);
				const double z = point.Z;
				minz = min(minz, z);
				maxz = max(maxz, z);

				const double sqrt = linearalgebra::sqr_len(point.X, point.Y);
				minSqrt = min(minSqrt, sqrt);
				maxSqrt = max(maxSqrt, sqrt);
			}

			aminz[tid] = minz;
			amaxz[tid] = maxz;
			aminSqr[tid] = minSqrt;
			amaxSqr[tid] = maxSqrt;
			
			for (TID halfBlock = blockDim.x / 2; halfBlock > 0; halfBlock >>= 1)
			{
				if(tid < halfBlock && tid+halfBlock < N)
				{
					minz = min(aminz[tid], aminz[tid + halfBlock]);
					maxz = max(amaxz[tid], amaxz[tid + halfBlock]);
					minSqrt = max(aminSqr[tid], aminSqr[tid + halfBlock]);
					maxSqrt = max(amaxSqr[tid], amaxSqr[tid + halfBlock]);
				}

				__syncthreads();
			}

			output[0] = minz;
			output[1] = maxz;
			output[2] = minSqrt;
			output[3] = maxSqrt;
		}
	}
}
