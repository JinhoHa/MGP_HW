#include "LeNet5_cuda.h"

void LeNet5_cuda::predict(int batch) {
    // TODO: Implement conv1
    // TODO: Implement relu
    // TODO: Implement pool1
    // TODO: Implement conv2
    // TODO: Implement relu
    // TODO: Implement pool2
    // TODO: Implement fc1
    // TODO: Implement relu
    // TODO: Implement fc2
    // TODO: Implement relu
    // TODO: Implement fc3

    /* NOTE: unless you want to make a major change to this class structure, 
    *  you need to write your output to the device memory d_output 
    *  so that classify() can handle the rest.
    */
}

void LeNet5_cuda::prepare_device_memory(uint8_t* image) {
  // Alloc Model Parameters
  cudaMalloc((void**)&d_conv1_weight,
             sizeof(double) * conv1_in_channel * conv1_out_channel *
                 conv1_kernel_size * conv1_kernel_size);
  cudaMalloc((void**)&d_conv1_bias, sizeof(double) * conv1_out_channel);
  cudaMalloc((void**)&d_conv2_weight,
             sizeof(double) * conv2_in_channel * conv2_out_channel *
                 conv2_kernel_size * conv2_kernel_size);
  cudaMalloc((void**)&d_conv2_bias, sizeof(double) * conv2_out_channel);
  cudaMalloc((void**)&d_fc1_weight,
             sizeof(double) * fc1_in_channel * fc1_out_channel);
  cudaMalloc((void**)&d_fc1_bias, sizeof(double) * fc1_out_channel);
  cudaMalloc((void**)&d_fc2_weight,
             sizeof(double) * fc2_in_channel * fc2_out_channel);
  cudaMalloc((void**)&d_fc2_bias, sizeof(double) * fc2_out_channel);
  cudaMalloc((void**)&d_fc3_weight,
             sizeof(double) * fc3_in_channel * fc3_out_channel);
  cudaMalloc((void**)&d_fc3_bias, sizeof(double) * fc3_out_channel);

  // Alloc Activations
  cudaMalloc((void**)&d_image,
             sizeof(uint8_t) * batch * input_size * input_size * input_channel);
  cudaMalloc((void**)&d_input,
             sizeof(double) * batch * input_channel * input_size * input_size);
  cudaMalloc((void**)&d_C1_feature_map,
             sizeof(double) * batch * C1_channel * C1_size * C1_size);
  cudaMalloc((void**)&d_S2_feature_map,
             sizeof(double) * batch * S2_channel * S2_size * S2_size);
  cudaMalloc((void**)&d_C3_feature_map,
             sizeof(double) * batch * C3_channel * C3_size * C3_size);
  cudaMalloc((void**)&d_S4_feature_map,
             sizeof(double) * batch * S4_channel * S4_size * S4_size);
  cudaMalloc((void**)&d_C5_layer, sizeof(double) * batch * C5_size);
  cudaMalloc((void**)&d_F6_layer, sizeof(double) * batch * F6_size);
  cudaMalloc((void**)&d_output, sizeof(double) * batch * output_size);

  // Copy Parameters
  cudaMemcpy(d_conv1_weight, conv1_weight,
             sizeof(double) * conv1_in_channel * conv1_out_channel *
                 conv1_kernel_size * conv1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_bias, conv1_bias, sizeof(double) * conv1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_weight, conv2_weight,
             sizeof(double) * conv2_in_channel * conv2_out_channel *
                 conv2_kernel_size * conv2_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_bias, conv2_bias, sizeof(double) * conv2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_weight, fc1_weight,
             sizeof(double) * fc1_in_channel * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_bias, fc1_bias, sizeof(double) * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_weight, fc2_weight,
             sizeof(double) * fc2_in_channel * fc2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_bias, fc2_bias, sizeof(double) * fc2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc3_weight, fc3_weight,
             sizeof(double) * fc3_in_channel * fc3_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc3_bias, fc3_bias, sizeof(double) * fc3_out_channel,
             cudaMemcpyHostToDevice);
  // copy input image
  size_t image_size = batch * input_size * input_size * input_channel;
  cudaMemcpy(d_image, image, image_size * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
}

void LeNet5_cuda::classify(int* predict, int batch) {
  // read logits back to cpu
  cudaMemcpy(output, d_output, sizeof(double) * output_size * batch,
             cudaMemcpyDeviceToHost);
  // Softmax
  softmax(output, predict, batch, output_size);
}

LeNet5_cuda::~LeNet5_cuda() {
  cudaFree(d_conv1_weight);   
  cudaFree(d_conv2_weight);   
  cudaFree(d_conv1_bias);     
  cudaFree(d_conv2_bias);     
  cudaFree(d_fc1_weight);     
  cudaFree(d_fc2_weight);     
  cudaFree(d_fc3_weight);     
  cudaFree(d_fc1_bias);       
  cudaFree(d_fc2_bias);       
  cudaFree(d_fc3_bias);       

  cudaFree(d_image);          
  cudaFree(d_input);          
  cudaFree(d_C1_feature_map); 
  cudaFree(d_S2_feature_map); 
  cudaFree(d_C3_feature_map); 
  cudaFree(d_S4_feature_map); 
  cudaFree(d_C5_layer);      
  cudaFree(d_F6_layer);     
  cudaFree(d_output);       
  cudaFree(d_predict_cuda);   
}
