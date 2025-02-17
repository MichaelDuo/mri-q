/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * C code for creating the Q data structure for fast convolution-based
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

 #include <stdio.h>
 #include <math.h>
 #include <stdlib.h>
 #include <string.h>
 #include <sys/time.h>
 #include <malloc.h>
 
 #include "parboil.h"
 
 #include "file.h"
 #include "computeQ.cu"
 
 int main (int argc, char *argv[])
 {
   int numX, numK;		/* Number of X and K values */
   int original_numK;		/* Number of K values in input file */
   float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
   float *x, *y, *z;		/* X coordinates (3D vectors) */
   float *phiR, *phiI;		/* Phi values (complex) */
   float *phiMag;		/* Magnitude of Phi */
   float *Qr, *Qi;		/* Q signal (complex) */
   struct kValues* kVals;
 
   float *phiR_d, *phiI_d, *phiMag_d;
   float *Qr_d, *Qi_d;
   float *x_d, *y_d, *z_d;
 
   struct pb_Parameters *params;
   struct pb_TimerSet timers;
 
   pb_InitializeTimerSet(&timers);
 
   params = pb_ReadParameters(&argc, argv);
   if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
   {
 
     fprintf(stderr, "Expecting one input filename\n");
     exit(-1);
   }
 
   pb_SwitchToTimer(&timers, pb_TimerID_IO);
   inputData(params->inpFiles[0],
       &original_numK, &numX,
       &kx, &ky, &kz,
       &x, &y, &z,
       &phiR, &phiI);
 
   if (argc < 2)
     numK = original_numK;
   else
   {
     int inputK;
     char *end;
     inputK = strtol(argv[1], &end, 10);
     if (end == argv[1])
     {
       fprintf(stderr, "Expecting an integer parameter\n");
       exit(-1);
     }
 
     numK = MIN(inputK, original_numK);
   }
 
   printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
          numX, original_numK, numK);
 
   pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
 
   createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);
  
   cudaSetDevice(1);
 
   pb_SwitchToTimer(&timers, pb_TimerID_COPY);

   cudaMalloc((void** )&phiR_d, sizeof(float) * numK);
   cudaMalloc((void** )&phiI_d, sizeof(float) * numK);
   cudaMalloc((void** )&phiMag_d, sizeof(float) * numK);
   cudaDeviceSynchronize();

    int n = 0, i = 0;
    unsigned int offset = 0;
    const unsigned int n_streams_phimag = 4;

    const unsigned int stream_size = ceil(numK / n_streams_phimag);

    cudaStream_t stream[n_streams_phimag];

    for (i = 0; i < n_streams_phimag; i++) {
      cudaStreamCreate(&stream[i]);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COPY_ASYNC);
    
    cudaMemcpy(phiR_d, phiR, sizeof(float) * numK, cudaMemcpyHostToDevice);
    cudaMemcpy(phiI_d, phiI, sizeof(float) * numK, cudaMemcpyHostToDevice);
    cudaMemset(phiMag_d, 0, sizeof(float) * numK);

    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    for (n = 0; n < n_streams_phimag; n++) {
      offset = n * stream_size;
      ComputePhiMagGPUAsync(numK, stream_size, phiR_d, phiI_d, phiMag_d,
        stream[n], offset);
    }
    pb_SwitchToTimer(&timers, pb_TimerID_COPY_ASYNC);

    cudaMemcpy(phiMag, phiMag_d, sizeof(float) * numK, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for (i = 0; i < n_streams_phimag; i++) {
      cudaStreamDestroy(stream[i]);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
 
   cudaFree(phiMag_d);
   cudaFree(phiI_d);
   cudaFree(phiR_d);
   cudaDeviceSynchronize();
 
   pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
 
   kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
   int k;
   for (k = 0; k < numK; k++) {
     kVals[k].Kx = kx[k];
     kVals[k].Ky = ky[k];
     kVals[k].Kz = kz[k];
     kVals[k].PhiMag = phiMag[k];
   }
  
   pb_SwitchToTimer(&timers, pb_TimerID_COPY);
 
   cudaMalloc((void** )&Qr_d, sizeof(float) * numX);
   cudaMalloc((void** )&Qi_d, sizeof(float) * numX);
   cudaMalloc((void** )&x_d, sizeof(float) * numX);
   cudaMalloc((void** )&y_d, sizeof(float) * numX);
   cudaMalloc((void** )&z_d, sizeof(float) * numX);
   cudaDeviceSynchronize();
 
   {
 
     int n = 0, i = 0;
     unsigned int offset = 0;
     const unsigned int n_streams_q = 4;
     const unsigned int stream_size = ceil(numX / n_streams_q);
     cudaStream_t stream[n_streams_q];
 
     for (i = 0; i < n_streams_q; i++) {
       cudaStreamCreate(&stream[i]);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_COPY_ASYNC);

    cudaMemcpy(x_d, x, sizeof(float) * numX, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, sizeof(float) * numX, cudaMemcpyHostToDevice);
    cudaMemcpy(z_d, z, sizeof(float) * numX, cudaMemcpyHostToDevice);

    cudaMemset(Qr_d, 0, sizeof(float) * numX);
    cudaMemset(Qi_d, 0, sizeof(float) * numX);
 
     pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
     for (n = 0; n < n_streams_q; n++) {
       offset = n * stream_size;
       ComputeQGPUAsync(numK, stream_size, kVals, x_d, y_d, z_d, Qr_d, Qi_d,
         stream[n], offset);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_COPY_ASYNC);

    cudaMemcpy(Qr, Qr_d, sizeof(float) * numX, cudaMemcpyDeviceToHost);
    cudaMemcpy(Qi, Qi_d, sizeof(float) * numX, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
 
     for (i = 0; i < n_streams_q; i++) {
       cudaStreamDestroy(stream[i]);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_COPY);
   } 
 
   cudaFree(z_d);
   cudaFree(y_d);
   cudaFree(x_d);
   cudaFree(Qi_d);
   cudaFree(Qr_d);
   cudaDeviceSynchronize();
   cudaDeviceReset();
 
   if (params->outFile)
   {
     pb_SwitchToTimer(&timers, pb_TimerID_IO);
     outputData(params->outFile, Qr, Qi, numX);
     pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
   }
 
 
   free (kx);
   free (ky);
   free (kz);
   free (x);
   free (y);
   free (z);
   free (phiR);
   free (phiI);
   free (phiMag);
   free (kVals);
   free (Qr);
   free (Qi);
 
   pb_SwitchToTimer(&timers, pb_TimerID_NONE);
   pb_PrintTimerSet(&timers);
   pb_FreeParameters(params);
 
   return 0;
 }
 