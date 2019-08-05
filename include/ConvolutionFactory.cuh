#ifndef CONVOLUTIONFACTORY_CUH
#define CONVOLUTIONFACTORY_CUH

#include "common_includes.h"
#include "Unity.cuh"

__global__ void convolution1D(const float* __restrict__ kernel, unsigned int kernelLength, const float* __restrict__ data, unsigned int dataSize, float* convData);

class ConvolutionFactory{

unsigned int width;
unsigned int dimensions;
jax::Unity<float>* kernel;


public:

  ConvolutionFactory();
  ~ConvolutionFactory();

  void setKernel(jax::Unity<float>* kernel, unsigned int width, unsigned int dimensions);
  jax::Unity<float>* convolve(jax::Unity<float>* data);
  //TODO implement overloaded convolve for other dimensions and such


};


#endif  /* DFT_CUH */
