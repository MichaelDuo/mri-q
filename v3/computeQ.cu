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

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

#define BLOCK_SIZE 128
#define KVALS_SH_SIZE BLOCK_SIZE

__global__ void ComputePhiMagKernel(int numK, float *phiR, float *phiI, float *phiMag){
  int t = threadIdx.x + (blockIdx.x * blockDim.x);
  if (t < numK){
    float real = phiR[t];
    float imag = phiI[t];
    phiMag[t] = (real * real) + (imag * imag);
  }
}

__global__ void ComputeQKernel(int numK, int numX, struct kValues *kVals_d, float *x_d, float *y_d, float *z_d, float *Qr_d, float *Qi_d) {
  unsigned int t = threadIdx.x + (blockIdx.x * blockDim.x);

  if (t >= numX)
    return;

  __shared__ struct kValues sh_kValues[KVALS_SH_SIZE];

  float x_l = x_d[t];
  float y_l = y_d[t];
  float z_l = z_d[t];
  float Qracc = 0.0f;
  float Qiacc = 0.0f;
  float phi = 0.0f;

  float expArg, cosArg, sinArg;
  int temp;
  int nBlk = ((numK - 1) / KVALS_SH_SIZE) + 1;

  for (int idx_o = 0; idx_o < nBlk; idx_o++) {
    temp = threadIdx.x + (idx_o * KVALS_SH_SIZE);

    if (temp < numK) {
      sh_kValues[threadIdx.x].Kx = kVals_d[temp].Kx;
      sh_kValues[threadIdx.x].Ky = kVals_d[temp].Ky;
      sh_kValues[threadIdx.x].Kz = kVals_d[temp].Kz;
      sh_kValues[threadIdx.x].PhiMag = kVals_d[temp].PhiMag;
    } else {
      sh_kValues[threadIdx.x].Kx = 0;
      sh_kValues[threadIdx.x].Ky = 0;
      sh_kValues[threadIdx.x].Kz = 0;
      sh_kValues[threadIdx.x].PhiMag = 0;
    }
    __syncthreads();

    for (int idx = 0; idx < KVALS_SH_SIZE; idx++) {
      expArg = PIx2 * (sh_kValues[idx].Kx * x_l +
                       sh_kValues[idx].Ky * y_l +
                       sh_kValues[idx].Kz * z_l);

      cosArg = cos(expArg);
      sinArg = sin(expArg);

      phi = sh_kValues[idx].PhiMag;
      Qracc += phi * cosArg;
      Qiacc += phi * sinArg;
    }
    __syncthreads();
  }

  Qr_d[t] = Qracc;
  Qi_d[t] = Qiacc;
}

void ComputePhiMagGPU(int numK, float* phiR_d, float* phiI_d,
                      float* phiMag_d)
{
  int numBlocks = ((numK - 1) / BLOCK_SIZE) + 1;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  /*printf("PhiMag numBlocks : %d\n", numBlocks);*/
  ComputePhiMagKernel<<<dimGrid, dimBlock>>>(numK, phiR_d, phiI_d, phiMag_d);
}

void ComputeQGPU(int numK, int numX, struct kValues *kVals_d, float *x_d, float *y_d, float *z_d, float *Qr_d, float *Qi_d) {
  int numBlocks = ((numX - 1) / BLOCK_SIZE) + 1;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  ComputeQKernel<<<dimGrid, dimBlock>>>(numK, numX, kVals_d, x_d, y_d, z_d, Qr_d, Qi_d);
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
