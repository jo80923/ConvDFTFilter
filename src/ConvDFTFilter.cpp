#include "common_includes.h"
#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"
#include "Unity.cuh"
#include "ConvolutionFactory.cuh"
#include "MatrixUtil.cuh"


/*
NOTE: CUDA convolution will only be faster than matlab if matrices are large
*/

int main(int argc, char* argv[]){
  std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::startMATLAB();
  matlab::data::ArrayFactory factory;
  std::cout <<"MATLAB engine has started"<<std::endl;

  matlabPtr->eval(u"load data/convolution_data.mat");
  matlab::data::Array pulseMat = matlabPtr->getVariable(u"pulse");
  matlab::data::Array reflectivityMat = matlabPtr->getVariable(u"reflectivity");

  unsigned int pulseLength = pulseMat.getNumberOfElements();
  unsigned int reflectivityLength = reflectivityMat.getNumberOfElements();
  float* pulse_host = new float[pulseLength];
  float* reflectivity_host = new float[reflectivityLength];
  for(int i = 0; i < pulseLength; ++i){
    pulse_host[i] = float(pulseMat[i]);
  }
  for(int i = 0; i < reflectivityLength; ++i){
    reflectivity_host[i] = float(reflectivityMat[i]);
  }
  jax::Unity<float>* pulse = new jax::Unity<float>(pulse_host, pulseLength, jax::cpu);
  jax::Unity<float>* reflectivity = new jax::Unity<float>(reflectivity_host, reflectivityLength, jax::cpu);

  std::cout <<"convolution_data.mat parsing has completed"<<std::endl;

  std::cout<<"\nSTART OF HW PROBLEM 2: SIGNAL CONVOLUTION\n"<<std::endl;

  //run custom cuda convolution

  ConvolutionFactory convolutionFactory;
  convolutionFactory.setKernel(pulse, pulseLength, 1);
  std::cout<<"Starting Custom CUDA Convolution"<<std::endl;
  jax::Unity<float>* convData = convolutionFactory.convolve(reflectivity);
  std::cout<<"Custom CUDA Convolution has returned successfully"<<std::endl;
  matlab::data::TypedArray<double> convDataMat = matlabPtr->feval(u"zeros",{factory.createScalar(1),factory.createScalar(pulseLength + reflectivityLength - 1)});
  for(int i = 0; i < pulseLength + reflectivityLength - 1; ++i){
    convDataMat[i] = convData->host[i];
  }

  //now plot comparison to maltab conv

  std::cout<<"Starting built-in Matlab Convolution for comparison"<<std::endl;
  matlabPtr->eval(u"builtInConv = conv(pulse, reflectivity);");
  std::cout<<"Built-in Matlab Convolution has returned successfully"<<std::endl;
  matlabPtr->setVariable(u"cudaConv",convDataMat);

  matlabPtr->eval(u"figure;sgtitle 'HW Problem 2 : Signal Convolution';\
  subplot(2,1,1);plot(cudaConv);axis([0 255 -0.2 0.2]);title 'Custom CUDA Convolution';\
  subplot(2,1,2);plot(builtInConv);axis([0 255 -0.2 0.2]);title 'Matlab Convolution';");

  std::cout<<"\nEND OF HW PROBLEM 2: SIGNAL CONVOLUTION\n"<<std::endl;

  std::cout<<"\nSTART OF HW PROBLEM 3.1: DFT\n"<<std::endl;

  std::vector<std::complex<float>> dftmtx_reflectivity = computeDFTMatrix(reflectivityLength);
  matlab::data::TypedArray<std::complex<float>> dftmtx_custom = factory.createArray<std::complex<float>>({reflectivityLength,reflectivityLength});
  for(int r = 0; r < reflectivityLength; ++r){
    for(int c = 0; c < reflectivityLength; ++c){
      dftmtx_custom[r][c] = dftmtx_reflectivity[r*reflectivityLength + c];
    }
  }
  matlabPtr->setVariable(u"dftmtx_cuda",dftmtx_custom);
  matlabPtr->eval(u"figure;sgtitle 'HW Problem 3 : DFTMTX';plot(dftmtx_cuda);");

  std::vector<std::complex<float>> freq_reflectivity = computeDFT(reflectivity);
  std::vector<std::complex<float>> freq_pulse = computeDFT(pulse);
  std::vector<std::complex<float>> freq_conv = computeDFT(convData);


  matlab::data::ArrayDimensions pSize = {1,pulseLength};
  matlab::data::ArrayDimensions rSize = {1,reflectivityLength};
  matlab::data::ArrayDimensions cSize = {1,pulseLength + reflectivityLength - 1};

  matlab::data::TypedArray<std::complex<float>> freq_reflectivityMat = factory.createArray<std::complex<float>>(rSize);
  matlab::data::TypedArray<std::complex<float>> freq_pulseMat = factory.createArray<std::complex<float>>(pSize);
  matlab::data::TypedArray<std::complex<float>> freq_convMat = factory.createArray<std::complex<float>>(cSize);
  for(int i = 0; i < reflectivityLength; ++i){
    freq_reflectivityMat[i] = freq_reflectivity[i];
  }
  for(int i = 0; i < pulseLength; ++i){
    freq_pulseMat[i] = freq_pulse[i];
  }
  for(int i = 0; i < pulseLength + reflectivityLength - 1; ++i){
    freq_convMat[i] = freq_conv[i];
  }


  matlabPtr->eval(u"sampleRate = 2;fs = 500;");//2ms and 500hz
  matlabPtr->setVariable(u"freq_reflectivityCUDA",freq_reflectivityMat);
  matlabPtr->setVariable(u"freq_pulseCUDA",freq_pulseMat);
  matlabPtr->setVariable(u"freq_convCUDA",freq_convMat);
  matlabPtr->eval(u"pl = length(pulse);rl = length(reflectivity);cl = pl + rl - 1;");
  matlabPtr->eval(u"reflectivity_dftmtx = dftmtx(rl);pulse_dftmtx = dftmtx(pl);\
  conv_dftmtx = dftmtx(cl);freq_reflectivity = reflectivity_dftmtx*reflectivity;");
  matlabPtr->eval(u"freq_conv = cudaConv*conv_dftmtx;freq_conv = cudaConv*conv_dftmtx;\
  freq_pulse = pulse_dftmtx*pulse;");

  matlabPtr->eval(u"figure;sgtitle 'HW Problem 3.1 : DFT(reflectivity)';");
  matlabPtr->eval(u"f = [-rl/2:rl/2-1]*(fs/rl);subplot(2,1,1);\
  plot(f,abs(fftshift(freq_reflectivityCUDA)));axis([-250 250 0 inf]);\
  xlabel('Frequency (Hz)');ylabel('Magnitude');title 'Custom CUDA DFT';subplot(2,1,2);\
  plot(f,abs(fftshift(freq_reflectivity)));axis([-250 250 0 inf]);\
  xlabel('Frequency (Hz)');ylabel('Magnitude');title 'Matlab DFT';");

  matlabPtr->eval(u"figure;sgtitle 'HW Problem 3.1 : DFT(pulse)';");
  matlabPtr->eval(u"f = [-pl/2:pl/2-1]*(fs/pl);subplot(2,1,1);\
  plot(f,abs(fftshift(freq_pulseCUDA)));axis([-250 250 0 inf]);\
  title 'Custom CUDA DFT';xlabel('Frequency (Hz)');ylabel('Magnitude');\
  subplot(2,1,2);plot(f,abs(fftshift(freq_pulse)));axis([-250 250 0 inf]);\
  title 'Matlab DFT';xlabel('Frequency (Hz)');ylabel('Magnitude');");

  matlabPtr->eval(u"figure;sgtitle 'HW Problem 3.1 : DFT(convolved data)';\
  f = [-cl/2:cl/2-1]*(fs/cl);subplot(2,1,1);plot(f,abs(fftshift(freq_convCUDA)));\
  axis([-250 250 0 inf]);title 'Custom CUDA DFT';xlabel('Frequency (Hz)');ylabel('Magnitude');\
  subplot(2,1,2);plot(f,abs(fftshift(freq_conv)));axis([-250 250 0 inf]);\
  title 'Matlab DFT';xlabel('Frequency (Hz)');ylabel('Magnitude');");


  std::cout<<"\nEND OF HW PROBLEM 3.1: DFT\n"<<std::endl;

  std::cout<<"\nSTART OF HW PROBLEM 3.2: BANDPASS FILTER\n"<<std::endl;

  matlabPtr->eval(u"t = [0:1:256];sig = sin(2*pi*7/1000*t) + sin(2*pi*50/1000*t) + sin(2*pi*110/1000*t);");
  matlab::data::Array compositeSignalMat = matlabPtr->getVariable(u"sig");
  float* compositeSignal_host = new float[256]();
  for(int i = 0; i < 256; ++i){
    compositeSignal_host[i] = (float) compositeSignalMat[i];
  }
  jax::Unity<float>* compositeSignal = new jax::Unity<float>(compositeSignal_host, 256, jax::cpu);

  matlabPtr->eval(u"figure;plot(sig);axis([0 257 -inf inf]);\
  title '7Hz+50Hz+110Hz signal';xlabel('time (ms)');");

  matlab::data::ArrayDimensions sSize = {1,256};
  std::vector<std::complex<float>> freq_compositeSignal = computeDFT(compositeSignal);
  matlab::data::TypedArray<std::complex<float>> freq_compositeSignalMat = factory.createArray<std::complex<float>>(sSize);
  for(int i = 0; i < 256; ++i){
    freq_compositeSignalMat[i] = freq_compositeSignal[i];
  }

  matlabPtr->setVariable(u"freq_compositeSignalCUDA",freq_compositeSignalMat);
  matlabPtr->eval(u"f = [-256/2:256/2-1]*(1000/256);figure;\
  plot(f,abs(fftshift(freq_compositeSignalCUDA)));\
  title 'Signal in Frequncy Domain';set(gca,'XTick', -450:100:450);\
  xlabel('Frequency (Hz)');ylabel('Magnitude');");

  //generate bandpass filter
  float* filter_host = new float[256];
  float f1_c = 40.0f/1000.0f;
  float f2_c = 60.0f/1000.0f;
  float w1_c = 2*M_PI*f1_c;
  float w2_c = 2*M_PI*f2_c;
  int middle = 128;
  for(int i = -128; i < 128; ++i){
    if(i == 0) filter_host[middle] = 2*(f2_c - f1_c);
    else filter_host[i + middle] = sin(w2_c*i)/(M_PI*i) - sin(w1_c*i)/(M_PI*i);
  }
  jax::Unity<float>* filter = new jax::Unity<float>(filter_host, 256, jax::cpu);
  matlab::data::ArrayDimensions fSize = {1,256};
  std::vector<std::complex<float>> freq_filter = computeDFT(filter);
  matlab::data::TypedArray<std::complex<float>> freq_filterMat = factory.createArray<std::complex<float>>(fSize);
  for(int i = 0; i < 256; ++i){
    freq_filterMat[i] = freq_filter[i];
  }
  convolutionFactory.setKernel(filter,filter->numElements,1);
  jax::Unity<float>* filterConv = convolutionFactory.convolve(compositeSignal);
  matlab::data::ArrayDimensions fSSize = {1,511};
  std::vector<std::complex<float>> freq_filteredSignal = computeDFT(filterConv);
  matlab::data::TypedArray<std::complex<float>> freq_filteredSignalMat = factory.createArray<std::complex<float>>(fSSize);
  for(int i = 0; i < 256; ++i){
    freq_filteredSignalMat[i] = freq_filteredSignal[i];
  }
  matlabPtr->setVariable(u"filteredSignal",freq_filteredSignalMat);
  matlabPtr->eval(u"f = [-511/2:511/2-1]*(1000/511);figure;\
  plot(f,abs(fftshift(filteredSignal)));\
  title 'Filtered Signal in Frequency Domain';xlabel('Frequency (Hz)');\
  ylabel('Magnitude');set(gca,'XTick', -450:100:450);");

  std::cout<<"\nEND OF HW PROBLEM 3.2: BANDPASS FILTER\n"<<std::endl;

  matlabPtr->feval<void>(u"pause",1000);
  return 0;
}
