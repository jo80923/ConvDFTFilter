#include "ConvolutionFactory.cuh"



//MAX KERNEL WIDTH OF 1024
__global__ void convolution1D(const float* __restrict__ kernel, unsigned int kernelLength, const float* __restrict__ data, unsigned int dataSize, float* convData){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  unsigned int regKernelLength = kernelLength;
  if(blockId < (regKernelLength + dataSize - 1)){
    __shared__ float convValue;
    convValue = 0.0f;
    __syncthreads();
    for(int i = threadIdx.x; i < dataSize && blockId - i >= 0 && blockId - i < regKernelLength; i += blockDim.x){
      atomicAdd(&convValue, data[i]*kernel[blockId - i]);
    }
    __syncthreads();
    if(threadIdx.x == 0){
      convData[blockId] = convValue;
    }
  }
}

ConvolutionFactory::ConvolutionFactory(){
  this->kernel = NULL;
}
ConvolutionFactory::~ConvolutionFactory(){

}

void ConvolutionFactory::setKernel(jax::Unity<float>* kernel, unsigned int width, unsigned int dimensions){
  this->width = width;
  this->dimensions = dimensions;
  this->kernel = kernel;
}
jax::Unity<float>* ConvolutionFactory::convolve(jax::Unity<float>* data){
  assert(this->kernel != NULL);
  assert(data != NULL);

  unsigned int length = data->numElements;

  this->kernel->transferMemoryTo(jax::gpu);
  data->transferMemoryTo(jax::gpu);

  float* result_device;
  CudaSafeCall(cudaMalloc((void**)&result_device, (this->width+length-1)*sizeof(float)));
  jax::Unity<float>* result = new jax::Unity<float>(result_device, (this->width+length-1), jax::gpu);

  dim3 grid = {1,1,1};
  dim3 block = {this->width + length - 1, 1, 1};
  if(block.x > 1024) block.x = 1024;
  getGrid((this->width+length-1), grid);

  convolution1D<<<block,grid>>>(kernel->device, this->width, data->device, length, result->device);
  CudaCheckError();

  result->transferMemoryTo(jax::cpu);

  return result;

}
