#include "LeNet5_cuda.h"

__global__
void normalize(uint8_t* image, double* input, int batch, int input_channel, int input_size) {
  // Initialize variables
  double max_int = 255.0L;
  double mean = 0.5L;
  double var = 0.5L;
  // // Normalize
  // for (int i = 0; i < batch * input_channel * input_size * input_size; i++) {
  //   input[i] = image[i] / max_int;       // transforms.ToTensor();
  //   input[i] = (input[i] - mean) / var;  // transforms.Normalize();
  // }
  // cuda
  // blockIdx.y : batch, blockIdx.x : Channel
  // threadIdx.y : input_size, threadIdx.x : input_size
  int taskIdx = blockIdx.y * gridDim.x * blockDim.y * blockDim.x
                + blockIdx.x * blockDim.y * blockDim.x
                + threadIdx.y * blockDim.x
                + threadIdx.x;
  input[taskIdx] = image[taskIdx] / max_int;
  input[taskIdx] = (input[taskIdx] - mean) / var;
}

__global__
void cuda_conv(double* input, double* output, double* weight,
                      double* bias, int B, int H, int W, int IC, int OC,
                      int K) {
  // // Initialize variable
  // int H_OUT = H - (K - 1);
  // int W_OUT = W - (K - 1);
  // Convolution
  // for (int b = 0; b < B; b++)              // mini-batch
  //   for (int oc = 0; oc < OC; oc++) {      // Output Channel
  //     for (int h = 0; h < H_OUT; h++)      // Height
  //       for (int w = 0; w < W_OUT; w++) {  // Width
  //         int output_index =
  //             b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
  //         output[output_index] = bias[oc];
  //         for (int ic = 0; ic < IC; ic++) {
  //           int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
  //           int kernel_base = oc * (IC * K * K) + ic * (K * K);
  //           for (int kh = 0; kh < K; kh++)
  //             for (int kw = 0; kw < K; kw++) {
  //               double val = input[input_base + kh * (W) + kw] *
  //                            weight[kernel_base + kh * (K) + kw];
  //               output[output_index] += val;
  //             }
  //         }
  //       }
  //   }

    // blockIdx.y : mini-batch (b)
    // blockIdx.x : output Channel (oc)
    // threadIdx.y : Height (h)
    // threadIdx.x : Width (w)
    int taskIdx = blockIdx.y * gridDim.x * blockDim.y * blockDim.x
                  + blockIdx.x * blockDim.y * blockDim.x
                  + threadIdx.y * blockDim.x
                  + threadIdx.x;
    double val = bias[blockIdx.x];
    for (int ic=0; ic<IC; ic++) {
      int input_base = blockIdx.y * (IC * H * W) + ic * (H * W)
                       + threadIdx.y * (W) + threadIdx.x;
      int kernel_base = blockIdx.x * (IC * K * K) + ic * (K * K);
      for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) {
          val += input[input_base + kh * (W) + kw] *
                 weight[kernel_base + kh * (K) + kw];
        }
    }
    output[taskIdx] = val;
}

__global__
void cuda_relu(double* feature_map) {
  // for (int i = 0; i < size; i++) feature_map[i] = std::max(feature_map[i], 0.0);
  // blockIdx.x : [batch, channel]
  // threadIdx.x : [H, W]
  int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (feature_map[taskIdx] < 0.0)
    feature_map[taskIdx] = 0.0;
}

__global__
void cuda_pool(double* input, double* output) {
  // // Initilaize variable
  // int scale = 2;
  // int H_OUT = H / scale;
  // int W_OUT = W / scale;
  // // Max Pooling
  // for (int b = 0; b < B; b++)
  //   for (int c = 0; c < C; c++)
  //     for (int h = 0; h < H; h += 2)
  //       for (int w = 0; w < W; w += 2) {
  //         // Init values
  //         int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
  //         int max_sh = 0;
  //         int max_sw = 0;
  //         double max_val = std::numeric_limits<double>::lowest();
  //         // Find maximum
  //         for (int sh = 0; sh < scale; sh++)
  //           for (int sw = 0; sw < scale; sw++) {
  //             double val = input[input_base + sh * (W) + sw];
  //             if (val - max_val > std::numeric_limits<double>::epsilon()) {
  //               max_val = val;
  //               max_sh = sh;
  //               max_sw = sw;
  //             }
  //           }
  //         // Set output with max value
  //         int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
  //                            (h / 2) * W_OUT + (w / 2);
  //         output[output_index] = max_val;
  //       }
  // blockIdx.y : BATCH
  // blockIdx.x : Channel
  // threadIdx.y : output h
  // threadIdx.x : output w
  int taskIdx = blockIdx.y * gridDim.x * blockDim.y * blockDim.x
                + blockIdx.x * blockDim.y * blockDim.x
                + threadIdx.y * blockDim.x
                + threadIdx.x;
  int input_base = blockIdx.y * gridDim.x * (2*blockDim.y) * (2*blockDim.x)
                + blockIdx.x * (2*blockDim.y) * (2*blockDim.x)
                + (2*threadIdx.y) * (2*blockDim.x)
                + (2*threadIdx.x);
  double max_val = 0.0;
  for (int sh = 0; sh < 2; sh++)
    for (int sw = 0; sw < 2; sw++) {
      double val = input[input_base + sh * (2*blockDim.x) + sw];
      if(val > max_val) {
        max_val = val;
      }
    }
  output[taskIdx] = max_val;
}

__global__
void cuda_fc(double* input, double* output, double* weight, double* bias,
                    int IC) {
  // // Fully Connected
  // for (int b = 0; b < B; b++)
  //   for (int oc = 0; oc < OC; oc++) {
  //     output[b * OC + oc] = bias[oc];
  //     for (int ic = 0; ic < IC; ic++)
  //       output[b * OC + oc] += weight[oc * IC + ic] * input[b * IC + ic];
  //   }

  // blockIdx.x : BATCH
  // threadIdx.x : out_channel
  int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
  double val = bias[threadIdx.x];
  for(int ic=0; ic<IC; ic++) {
    val += weight[threadIdx.x * IC + ic] * input[blockIdx.x * IC + ic];
  }
  output[taskIdx] = val;
}

void LeNet5_cuda::predict(int batch) {
  // uint8_t* image;
  // image = new uint8_t[batch * IMG_SIZE];
  // size_t image_size = batch * input_size * input_size * input_channel;
  // cudaMemcpy(image, d_image, image_size * sizeof(uint8_t),
  //            cudaMemcpyDeviceToHost);
  // ToTensor and Normalize
  dim3 DimGrid(input_channel, batch);
  dim3 DimBlock(input_size, input_size);
  normalize<<<DimGrid, DimBlock>>>(d_image, d_input, batch, input_channel, input_size);
  cudaDeviceSynchronize();

  // cudaMemcpy(input, d_input,
  //            batch * input_size * input_size * input_channel * sizeof(double),
  //            cudaMemcpyDeviceToHost);

  // Conv2d
  DimGrid.y = batch; DimGrid.x = conv1_out_channel;
  DimBlock.y = input_size - (conv1_kernel_size - 1);
  DimBlock.x = input_size - (conv1_kernel_size - 1);
  cuda_conv<<<DimGrid, DimBlock>>>(d_input, d_C1_feature_map, d_conv1_weight, d_conv1_bias, batch, input_size,
      input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);
  cudaDeviceSynchronize();

  // cudaMemcpy(C1_feature_map, d_C1_feature_map,
  //            batch * conv1_out_channel * C1_size * C1_size * sizeof(double),
  //            cudaMemcpyDeviceToHost);

  DimGrid.y = 1; DimGrid.x = batch * C1_channel;
  DimBlock.y = 1; DimBlock.x = C1_size * C1_size;
  cuda_relu<<<DimGrid, DimBlock>>>(d_C1_feature_map);
  cudaDeviceSynchronize();

  // cudaMemcpy(C1_feature_map, d_C1_feature_map,
  //            batch * conv1_out_channel * C1_size * C1_size * sizeof(double),
  //            cudaMemcpyDeviceToHost);

  // MaxPool2d
  DimGrid.y = batch; DimGrid.x = C1_channel;
  DimBlock.y = C1_size / 2; DimBlock.x = C1_size / 2;
  cuda_pool<<<DimGrid, DimBlock>>>(d_C1_feature_map, d_S2_feature_map);
  cudaDeviceSynchronize();

  // cudaMemcpy(S2_feature_map, d_S2_feature_map,
  //            batch * C1_channel * (C1_size / 2) * (C1_size / 2) * sizeof(double),
  //            cudaMemcpyDeviceToHost);

  // Conv2d
  DimGrid.y = batch; DimGrid.x = conv2_out_channel;
  DimBlock.y = S2_size - (conv2_kernel_size - 1);
  DimBlock.x = S2_size - (conv2_kernel_size - 1);
  cuda_conv<<<DimGrid, DimBlock>>>(d_S2_feature_map, d_C3_feature_map,
      d_conv2_weight, d_conv2_bias, batch, S2_size,
      S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);
  cudaDeviceSynchronize();

  DimGrid.y = 1; DimGrid.x = batch * C3_channel;
  DimBlock.y = 1; DimBlock.x = C3_size * C3_size;
  cuda_relu<<<DimGrid, DimBlock>>>(d_C3_feature_map);
  cudaDeviceSynchronize();

  // MaxPool2d
  DimGrid.y = batch; DimGrid.x = C3_channel;
  DimBlock.y = C3_size / 2; DimBlock.x = C3_size / 2;
  cuda_pool<<<DimGrid, DimBlock>>>(d_C3_feature_map, d_S4_feature_map);
  cudaDeviceSynchronize();

  // cudaMemcpy(S4_feature_map, d_S4_feature_map,
  //            batch * C3_channel * (C3_size / 2) * (C3_size / 2) * sizeof(double),
  //            cudaMemcpyDeviceToHost);

  // Linear
  DimGrid.y = 1; DimGrid.x = batch;
  DimBlock.y = 1; DimBlock.x = fc1_out_channel;
  cuda_fc<<<DimGrid, DimBlock>>>(d_S4_feature_map, d_C5_layer,
    d_fc1_weight, d_fc1_bias, fc1_in_channel);
  cudaDeviceSynchronize();

  // cudaMemcpy(C5_layer, d_C5_layer,
  //            batch * fc1_out_channel * sizeof(double),
  //            cudaMemcpyDeviceToHost);

  DimGrid.y = 1; DimGrid.x = batch;
  DimBlock.y = 1; DimBlock.x = C5_size;
  cuda_relu<<<DimGrid, DimBlock>>>(d_C5_layer);
  cudaDeviceSynchronize();

  // Linear
  DimGrid.y = 1; DimGrid.x = batch;
  DimBlock.y = 1; DimBlock.x = fc2_out_channel;
  cuda_fc<<<DimGrid, DimBlock>>>(d_C5_layer, d_F6_layer,
    d_fc2_weight, d_fc2_bias, fc2_in_channel);
  cudaDeviceSynchronize();

  DimGrid.y = 1; DimGrid.x = batch;
  DimBlock.y = 1; DimBlock.x = F6_size;
  cuda_relu<<<DimGrid, DimBlock>>>(d_F6_layer);
  cudaDeviceSynchronize();

  // Linear
  DimGrid.y = 1; DimGrid.x = batch;
  DimBlock.y = 1; DimBlock.x = fc3_out_channel;
  cuda_fc<<<DimGrid, DimBlock>>>(d_F6_layer, d_output,
    d_fc3_weight, d_fc3_bias, fc3_in_channel);
  cudaDeviceSynchronize();
  // cudaMemcpy(d_output, output, sizeof(double) * output_size * batch,
  //            cudaMemcpyHostToDevice);

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


void LeNet5_cuda::relu(double* feature_map, int size) {
  // relu
  for (int i = 0; i < size; i++) feature_map[i] = std::max(feature_map[i], 0.0);
}

void LeNet5_cuda::conv(double* input, double* output, double* weight,
                      double* bias, int B, int H, int W, int IC, int OC,
                      int K) {
  // Initialize variable
  int H_OUT = H - (K - 1);
  int W_OUT = W - (K - 1);
  // Convolution
  for (int b = 0; b < B; b++)              // mini-batch
    for (int oc = 0; oc < OC; oc++) {      // Output Channel
      for (int h = 0; h < H_OUT; h++)      // Height
        for (int w = 0; w < W_OUT; w++) {  // Width
          int output_index =
              b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
          output[output_index] = bias[oc];
          for (int ic = 0; ic < IC; ic++) {
            int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
            int kernel_base = oc * (IC * K * K) + ic * (K * K);
            for (int kh = 0; kh < K; kh++)
              for (int kw = 0; kw < K; kw++) {
                double val = input[input_base + kh * (W) + kw] *
                             weight[kernel_base + kh * (K) + kw];
                output[output_index] += val;
              }
          }
        }
    }
}

void LeNet5_cuda::pool(double* input, double* output, int B, int C, int H,
                      int W) {
  // Initilaize variable
  int scale = 2;
  int H_OUT = H / scale;
  int W_OUT = W / scale;
  // Max Pooling
  for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
      for (int h = 0; h < H; h += 2)
        for (int w = 0; w < W; w += 2) {
          // Init values
          int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
          int max_sh = 0;
          int max_sw = 0;
          double max_val = std::numeric_limits<double>::lowest();
          // Find maximum
          for (int sh = 0; sh < scale; sh++)
            for (int sw = 0; sw < scale; sw++) {
              double val = input[input_base + sh * (W) + sw];
              if (val - max_val > std::numeric_limits<double>::epsilon()) {
                max_val = val;
                max_sh = sh;
                max_sw = sw;
              }
            }
          // Set output with max value
          int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
                             (h / 2) * W_OUT + (w / 2);
          output[output_index] = max_val;
        }
}

void LeNet5_cuda::fc(double* input, double* output, double* weight, double* bias,
                    int B, int IC, int OC) {
  // Fully Connected
  for (int b = 0; b < B; b++)
    for (int oc = 0; oc < OC; oc++) {
      output[b * OC + oc] = bias[oc];
      for (int ic = 0; ic < IC; ic++)
        output[b * OC + oc] += weight[oc * IC + ic] * input[b * IC + ic];
    }
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
