
//#include <stdio.h>

#include "cuda_common.h"
#include "datastructure.h"
#include "kernels.h"

//extern "C" void resizeToFitCPU(double output[4], Vector3d* const vertices, Matrix4x4d * axis2modelInv, const int N);

__global__ void increment_kernel(int *g_data, int inc_value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	g_data[idx] = g_data[idx] + inc_value;
}

Vector3d RandomVector()
{
	 Vector3d vec {
		1.0f * (rand() % 9) ,
		1.0f * (rand() % 9) ,
		1.0f * (rand() % 9) };

	return vec;
}

 Matrix4x4d BuildRandomMatrix()
{
	 Matrix4x4d mtx;

	mtx.E00 = 1.0f * (rand() % 9);
	mtx.E01 = 1.0f * (rand() % 9);
	mtx.E02 = 1.0f * (rand() % 9);
	mtx.E10 = 1.0f * (rand() % 9);
	mtx.E11 = 1.0f * (rand() % 9);
	mtx.E12 = 1.0f * (rand() % 9);
	mtx.E20 = 1.0f * (rand() % 9);
	mtx.E21 = 1.0f * (rand() % 9);
	mtx.E22 = 1.0f * (rand() % 9);
	mtx.E03 = 1.0f * (rand() % 9);
	mtx.E13 = 1.0f * (rand() % 9);
	mtx.E23 = 1.0f * (rand() % 9);

	return mtx;
	
}

 Vector3d* BuildVectors(const int n)
{
	auto* const ptr = static_cast< Vector3d*>(malloc(
		sizeof( Vector3d) * n));

	for (int i = 0; i < n; i++)
		ptr[i] = RandomVector();

	return ptr;
}

int main(int argc, char *argv[])
{
	cudaDeviceProp deviceProps{};
	cudaSetDevice(1);

	printf("[%s] - Starting...\n", argv[0]);
	const int devID = findCudaDevice(argc, const_cast<const char**>(argv));

	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
	printf("CUDA device [%s]\n", deviceProps.name);

	const int N = 9;
	auto* vectors = BuildVectors(N);

	int blockSize = 3;
	int nBlocks = (N + blockSize - 1) / blockSize;
	int sharedMemory = sizeof(double) * blockSize;

	
	Vector3d* input;
	double* output;
	
	Matrix4x4d* g_mtx;
	Matrix4x4d h_mtx = BuildRandomMatrix();
	
	checkCudaErrors(cudaMallocManaged(&input, sizeof(Vector3d) * N));
	checkCudaErrors(cudaMallocManaged(&output, sizeof(double) * 4));
	checkCudaErrors(cudaMalloc(&g_mtx, sizeof(Matrix4x4d)));

	cudaMemcpy(g_mtx, &h_mtx, sizeof(Matrix4x4d), cudaMemcpyHostToDevice);
	
	g_mtx = &h_mtx;

	memcpy_s(input, sizeof(Vector3d) * N, vectors, sizeof(Vector3d) * N);
	//cudaDeviceSynchronize();

	resizeToFitCPU(output, input, g_mtx, N);
	cudaDeviceSynchronize();

	for(int i = 0; i < 4; i++)
	{
		std::cout << output[i] << std::endl;
	}

	cudaFree(input);
	cudaFree(output);
	cudaFree(g_mtx);

	free(vectors);

	cudaDeviceReset();
	
	exit(true ? EXIT_SUCCESS : EXIT_FAILURE);
}
