/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

#define PROJECT_DEF 1

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

#define BLOCK_SIZE 512
#define K_VAL_GRID_SIZE (BLOCK_SIZE * 4)
__constant__ __device__ kValues const_kValues[K_VAL_GRID_SIZE];

__global__ void ComputePhiMagKernelAsync(int numK, float *phiR, float *phiI,
                                         float *phiMag, int offset)
{
  unsigned int t = offset + threadIdx.x + (blockIdx.x * blockDim.x);
  if (t < numK) {
    phiMag[t] = (phiR[t] * phiR[t]) + (phiI[t] * phiI[t]);
  }
}

__global__ void ComputeQKernelAsync(int numK, int numX,
                                    float *x_d, float *y_d, float *z_d,
                                    float *Qr_d, float *Qi_d, int offset)
{
  unsigned int t = threadIdx.x + (blockIdx.x * blockDim.x);

  if (t >= numX)
    return;

  float x_l = x_d[t + offset];
  float y_l = y_d[t + offset];
  float z_l = z_d[t + offset];
  float Qracc = 0.0f;
  float Qiacc = 0.0f;
  float phi = 0.0f;

  float expArg;
  int idx = 0;

  for (; idx < numK; idx++) {
    /* using thread coarsening technique */
    expArg = PIx2 * (const_kValues[idx].Kx * x_l +
                     const_kValues[idx].Ky * y_l +
                     const_kValues[idx].Kz * z_l);

    phi = const_kValues[idx].PhiMag;

    Qracc += phi * cos(expArg);
    Qiacc += phi * sin(expArg);
  }

  Qr_d[t + offset] += Qracc;
  Qi_d[t + offset] += Qiacc;

}

void ComputePhiMagGPUAsync(int numK, int streamSz, float* phiR_d, float* phiI_d,
                           float* phiMag_d, cudaStream_t stream, int offset)
{
  unsigned int numBlocks = ((streamSz - 1) / BLOCK_SIZE) + 1;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  ComputePhiMagKernelAsync<<<dimGrid, dimBlock, 0, stream>>>(numK,
    phiR_d, phiI_d, phiMag_d, offset);
}

void ComputeQGPUAsync(int numK, int streamSz, struct kValues *kVals,
                      float *x_d, float *y_d, float *z_d,
                      float *Qr_d, float *Qi_d,
                      cudaStream_t stream, int offset)
{
  unsigned int kV_size_to_cover = K_VAL_GRID_SIZE;
  unsigned int n_iter = ((numK - 1) / K_VAL_GRID_SIZE) + 1;
  struct kValues *kV_ptr = kVals;

  unsigned int numBlocks = ((streamSz - 1) / BLOCK_SIZE) + 1;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);


  for (int iter = 0; iter < n_iter; iter++) {
    kV_size_to_cover = MIN(K_VAL_GRID_SIZE, numK - (iter * K_VAL_GRID_SIZE));
    if (kV_size_to_cover) {
      cudaMemcpyToSymbol(const_kValues, kV_ptr,
        kV_size_to_cover * sizeof(struct kValues), 0);

      ComputeQKernelAsync<<<dimGrid, dimBlock, 0, stream>>>(kV_size_to_cover,
        streamSz, x_d, y_d, z_d, Qr_d, Qi_d, offset);
      cudaDeviceSynchronize();
    }
    kV_ptr += kV_size_to_cover;
  }

}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}
