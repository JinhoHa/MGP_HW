#include "LeNet5_cuda.h"

void LeNet5_cuda::predict(int batch) {
  uint8_t* image;
  image = new uint8_t[batch * IMG_SIZE];
  size_t image_size = batch * input_size * input_size * input_channel;
  cudaMemcpy(image, d_image, image_size * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);
  // ToTensor and Normalize
  normalize(image, input);
  // Conv2d
  conv(input, C1_feature_map, conv1_weight, conv1_bias, batch, input_size,
      input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);
  relu(C1_feature_map, batch * C1_channel * C1_size * C1_size);
  // MaxPool2d
  pool(C1_feature_map, S2_feature_map, batch, C1_channel, C1_size, C1_size);
  // Conv2d
  conv(S2_feature_map, C3_feature_map, conv2_weight, conv2_bias, batch, S2_size,
      S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);
  relu(C3_feature_map, batch * C3_channel * C3_size * C3_size);
  // MaxPool2d
  pool(C3_feature_map, S4_feature_map, batch, C3_channel, C3_size, C3_size);
  // Linear
  fc(S4_feature_map, C5_layer, fc1_weight, fc1_bias, batch, fc1_in_channel,
    fc1_out_channel);
  relu(C5_layer, batch * C5_size);
  // Linear
  fc(C5_layer, F6_layer, fc2_weight, fc2_bias, batch, fc2_in_channel,
    fc2_out_channel);
  relu(F6_layer, batch * F6_size);
  // Linear
  fc(F6_layer, output, fc3_weight, fc3_bias, batch, fc3_in_channel,
    fc3_out_channel);
  cudaMemcpy(d_output, output, sizeof(double) * output_size * batch,
             cudaMemcpyHostToDevice);

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

void LeNet5_cuda::normalize(const uint8_t* const image, double* input) {
  // Initialize variables
  double max_int = 255.0L;
  double mean = 0.5L;
  double var = 0.5L;
  // Normalize
  for (int i = 0; i < batch * input_channel * input_size * input_size; i++) {
    input[i] = image[i] / max_int;       // transforms.ToTensor();
    input[i] = (input[i] - mean) / var;  // transforms.Normalize();
  }
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
