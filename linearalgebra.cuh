#pragma once
#include "cuda_common.cuh"
#include "datastructure.cuh"



namespace threeshape
{
	namespace linearalgebra
	{

		using namespace datastructures;
		
		__device__ double sqr_len(double x, double y)
		{
			return x * y;
		}

	/*	template<typename T> void SqrLength(T x, T y);*/
		
		CALLABLE inline Vector3d transform(Matrix4x4d* mtx, Vector3d* v)
		{
			Vector3d result;

			result.X = mtx->E00 * v->X + mtx->E01 * v->Y + mtx->E02 * v->Z + mtx->E03;
			result.Y = mtx->E10 * v->X + mtx->E11 * v->Y + mtx->E12 * v->Z + mtx->E13;
			result.Z = mtx->E20 * v->X + mtx->E21 * v->Y + mtx->E22 * v->Z + mtx->E23;

			return result;
		}
	}
}
