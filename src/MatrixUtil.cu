#include "MatrixUtil.cuh"

__global__ void normalize(float *mtx, float min, float max, unsigned long size) {
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  float currentValue = 0.0f;
  float regMin = min;
  float regMax = max;
  while(globalID < size){
    currentValue = 0.0f;
    if (mtx[globalID] != 0) {
      currentValue = mtx[globalID] - regMin;
      currentValue /= (regMax - regMin);
    }
    mtx[globalID] = currentValue;
    globalID += stride;
  }
}
__global__ void randInitMatrix(unsigned long size, float* mtx){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  if(globalID < size){
    mtx[globalID] = ((float)(clock64()%1000))/1000.0f;
  }
}
__global__ void multiplyMatrices(float *matrixA, float *matrixB, float *matrixC, long diffDimA, long comDim, long diffDimB){

  long blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  long currentIndex = globalID;

  if(currentIndex < (diffDimA * diffDimB)){

    long iIndex = currentIndex / diffDimB;
    long jIndex = currentIndex % diffDimB;

    float sum = 0;

    for(int k = 0; k < comDim; k++){

      sum += (matrixA[iIndex * comDim + k] * matrixB[k * diffDimB + jIndex]);
    }

    matrixC[iIndex * diffDimB + jIndex] = sum;
  }
}
__global__ void multiplyRowColumn(float *matrixA, float *matrixB, float* resultTranspose, long diffDimA, long diffDimB){

  long blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  long currentIndex = globalID;

  if(currentIndex < (diffDimA * diffDimB)){

    long iIndex = currentIndex / diffDimB;
    long jIndex = currentIndex % diffDimB;

    resultTranspose[jIndex * diffDimA + iIndex] = (matrixA[iIndex]*matrixB[jIndex]) + 1.0f;


  }
}

__global__ void generateDFTMatrix(unsigned int numElements, float2* dftmtx){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  unsigned int reg_numElements = numElements;
  if(blockId < reg_numElements){
    for(int i = threadIdx.x; i < reg_numElements; i += blockDim.x){
      float angle = 2*M_PI*i*blockId/reg_numElements;
      dftmtx[blockId*numElements + i] = {__cosf(angle),__sinf(angle)};
    }
  }
}
__global__ void computeDFT(unsigned int numElements, float* data, float2* result){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  unsigned int reg_numElements = numElements;
  if(blockId < reg_numElements){
    __shared__ float2 dft_value;
    dft_value = {0.0f,0.0f};
    __syncthreads();
    for(int i = threadIdx.x; i < reg_numElements; i += blockDim.x){
      float angle = 2*M_PI*i*blockId/reg_numElements;
      atomicAdd(&dft_value.x, data[i]*__cosf(angle));
      atomicAdd(&dft_value.y, data[i]*__sinf(angle));
    }
    __syncthreads();
    result[blockId] = dft_value;
  }
}

void executeMultiplyMatrices(float *matrixA, float *matrixB, float* &matrixC, long diffDimA, long comDim, long diffDimB){

  float* matrixADevice, *matrixBDevice, *matrixCDevice;

  CudaSafeCall(cudaMalloc((void**)&matrixADevice, diffDimA*comDim*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&matrixBDevice, comDim*diffDimB*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&matrixCDevice, diffDimA*diffDimB*sizeof(float)));

  CudaSafeCall(cudaMemcpy(matrixADevice, matrixA, diffDimA*comDim*sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(matrixBDevice, matrixB, comDim*diffDimB*sizeof(float), cudaMemcpyHostToDevice));

  dim3 grid, block;

  getFlatGridBlock(diffDimA*diffDimB, grid, block);

  multiplyMatrices<<<grid, block>>>(matrixADevice, matrixBDevice, matrixCDevice, diffDimA, comDim, diffDimB);

  CudaSafeCall(cudaMemcpy(matrixC, matrixCDevice, diffDimA*diffDimB*sizeof(float), cudaMemcpyDeviceToHost));

  CudaSafeCall(cudaFree(matrixADevice));
  CudaSafeCall(cudaFree(matrixBDevice));
  CudaSafeCall(cudaFree(matrixCDevice));

}

std::vector<std::complex<float>> computeDFT(jax::Unity<float>* data){
  assert(data != NULL);

  dim3 grid = {1,1,1};
  getGrid(data->numElements, grid);
  dim3 block = {data->numElements,1,1};
  if(block.x > 1024) block.x = 1024;

  data->transferMemoryTo(jax::gpu);
  float2* result_device = NULL;
  CudaSafeCall(cudaMalloc((void**)&result_device,data->numElements*sizeof(float2)));

  computeDFT<<<grid, block>>>(data->numElements, data->device, result_device);
  CudaCheckError();

  float2* temp = new float2[data->numElements];
  CudaSafeCall(cudaMemcpy(temp, result_device, data->numElements*sizeof(float2), cudaMemcpyDeviceToHost));

  std::vector<std::complex<float>> result_host;
  for(int i = 0; i < data->numElements; ++i){
    result_host.push_back({temp[i].x, temp[i].y});
  }

  return result_host;
}

std::vector<std::complex<float>> computeDFTMatrix(unsigned int length){

  dim3 grid = {1,1,1};
  getGrid(length, grid);
  dim3 block = {length,1,1};
  if(block.x > 1024) block.x = 1024;

  float2* result_device = NULL;
  CudaSafeCall(cudaMalloc((void**)&result_device,length*length*sizeof(float2)));

  generateDFTMatrix<<<grid, block>>>(length, result_device);
  CudaCheckError();

  float2* temp = new float2[length*length];
  CudaSafeCall(cudaMemcpy(temp, result_device, length*length*sizeof(float2), cudaMemcpyDeviceToHost));

  std::vector<std::complex<float>> result_host;
  for(int i = 0; i < length*length; ++i){
    result_host.push_back({temp[i].x, temp[i].y});
  }

  return result_host;
}
