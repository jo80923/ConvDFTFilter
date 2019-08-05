#ifndef MATRIXUTIL_CUH
#define MATRIXUTIL_CUH

#include "common_includes.h"
#include "Unity.cuh"


//TODO turn these methods into template methods for use with other types
__global__ void normalize(float* mtx, float min, float max, unsigned long size);
__global__ void randInitMatrix(unsigned long size, float* mtx);
__global__ void multiplyMatrices(float* matrixA, float* matrixB, float* matrixC, long diffDimA, long comDim, long diffDimB);
__global__ void multiplyRowColumn(float *matrixA, float *matrixB, float* resultTranspose, long diffDimA, long diffDimB);
__global__ void computeDFT(unsigned int numElements, float* data);

void executeMultiplyMatrices(float *matrixA, float *matrixB, float* &matrixC, long diffDimA, long comDim, long diffDimB);
std::vector<std::complex<float>> computeDFT(jax::Unity<float>* data);
std::vector<std::complex<float>> computeDFTMatrix(unsigned int length);

#endif /* MATRIXUTIL_CUH */
