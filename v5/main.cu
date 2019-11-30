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
 
 #define FATAL(msg, ...) \
     do {\
         fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
         exit(-1);\
     } while(0)
 
 
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
 
   /* Read command line */
   params = pb_ReadParameters(&argc, argv);
   if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
   {
 
     fprintf(stderr, "Expecting one input filename\n");
     exit(-1);
   }
 
   /* Read in data */
   pb_SwitchToTimer(&timers, pb_TimerID_IO);
   inputData(params->inpFiles[0],
       &original_numK, &numX,
       &kx, &ky, &kz,
       &x, &y, &z,
       &phiR, &phiI);
 
   /* Reduce the number of k-space samples if a number is given
    * on the command line */
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
   // Allocating memory
   cudaMalloc((void** )&phiR_d, sizeof(float) * numK);
   cudaMalloc((void** )&phiI_d, sizeof(float) * numK);
   cudaMalloc((void** )&phiMag_d, sizeof(float) * numK);
   cudaDeviceSynchronize();
 
     /* Allocate pinned memory */
     float *phiR_p, *phiI_p, *phiMag_p;
 
     cudaHostAlloc((void **)&phiR_p, sizeof(float) * numK, cudaHostAllocDefault);
     cudaHostAlloc((void **)&phiI_p, sizeof(float) * numK, cudaHostAllocDefault);
     cudaHostAlloc((void **)&phiMag_p, sizeof(float) * numK, cudaHostAllocDefault);
     for (int i = 0; i < numK; i++) {
       phiR_p[i] = phiR[i];
       phiI_p[i] = phiI[i];
     }
 
 
     int n = 0, i = 0;
     unsigned int offset = 0;
     /* choosing an appropriate number of streams */
     /* least number of samples in the given dataset
        is 2048, BLOCK_SIZE chosen in 512. 2048/4 = 512
        so that each stream has sufficient data to work on */
     const unsigned int n_streams_phimag = 4;
 
     /* divide input data into segments based on number of streams*/
     const unsigned int stream_size = ceil(numK / n_streams_phimag);
 
     cudaStream_t stream[n_streams_phimag];
 
     /* Create CUDA streams */
     for (i = 0; i < n_streams_phimag; i++) {
       cudaStreamCreate(&stream[i]);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_COPY_ASYNC);
     /* Copy data to the device asynchronously */
     for (n = 0; n < n_streams_phimag; n++) {
       offset = n * stream_size;
 
       cudaMemcpyAsync(&phiR_d[offset], &phiR_p[offset],
         sizeof(float) * stream_size, cudaMemcpyHostToDevice);
       cudaMemcpyAsync(&phiI_d[offset], &phiI_p[offset],
         sizeof(float) * stream_size, cudaMemcpyHostToDevice);
       cudaMemsetAsync(&phiMag_d[offset], 0,
         sizeof(float) * stream_size, stream[n]);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
     /* Call the kernels with respective streams */
     for (n = 0; n < n_streams_phimag; n++) {
       offset = n * stream_size;
       ComputePhiMagGPUAsync(numK, stream_size, phiR_d, phiI_d, phiMag_d,
         stream[n], offset);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_COPY_ASYNC);
     /* Copy data from the device asynchronously */
     for (n = 0; n < n_streams_phimag; n++) {
       offset = n * stream_size;
 
       cudaMemcpyAsync(&phiMag_p[offset], &phiMag_d[offset],
         sizeof(float) * stream_size, cudaMemcpyDeviceToHost);
     }
 
     /* Wait for all streams to finish */
     cudaDeviceSynchronize();
 
     /* Delete the streams */
     for (i = 0; i < n_streams_phimag; i++) {
       cudaStreamDestroy(stream[i]);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_COPY);
     for (i = 0; i < numK; i++)
       phiMag[i] = phiMag_p[i];
 
     cudaFreeHost(phiR_p);
     cudaFreeHost(phiI_p);
     cudaFreeHost(phiMag_p);
     pb_SwitchToTimer(&timers, pb_TimerID_COPY);
 
   /* Freeing up no longer needed memory on GPU */
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
 
   /* Allocating memory on GPU */
   cudaMalloc((void** )&Qr_d, sizeof(float) * numX);
   cudaMalloc((void** )&Qi_d, sizeof(float) * numX);
   cudaMalloc((void** )&x_d, sizeof(float) * numX);
   cudaMalloc((void** )&y_d, sizeof(float) * numX);
   cudaMalloc((void** )&z_d, sizeof(float) * numX);
   cudaDeviceSynchronize();
 
   {
     /* Allocate pinned memory */
     float *x_p, *y_p, *z_p, *Qr_p, *Qi_p;
 
     cudaHostAlloc((void **)&x_p, sizeof(float) * numX, cudaHostAllocDefault);
     cudaHostAlloc((void **)&y_p, sizeof(float) * numX, cudaHostAllocDefault);
     cudaHostAlloc((void **)&z_p, sizeof(float) * numX, cudaHostAllocDefault);
     cudaHostAlloc((void **)&Qr_p, sizeof(float) * numX, cudaHostAllocDefault);
     cudaHostAlloc((void **)&Qi_p, sizeof(float) * numX, cudaHostAllocDefault);
     for (int i = 0; i < numX; i++) {
       x_p[i] = x[i];
       y_p[i] = y[i];
       z_p[i] = z[i];
       Qr_p[i] = Qr[i];
       Qi_p[i] = Qi[i];
     }
 
     int n = 0, i = 0;
     unsigned int offset = 0;
     const unsigned int n_streams_q = 4;
     const unsigned int stream_size = ceil(numX / n_streams_q);
     cudaStream_t stream[n_streams_q];
 
     /* Create CUDA streams */
     for (i = 0; i < n_streams_q; i++) {
       cudaStreamCreate(&stream[i]);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_COPY_ASYNC);
     /* Copy data to the device asynchronously */
     for (n = 0; n < n_streams_q; n++) {
       offset = n * stream_size;
 
       cudaMemcpy(&x_d[offset], &x_p[offset],
         sizeof(float) * stream_size, cudaMemcpyHostToDevice);
       cudaMemcpy(&y_d[offset], &y_p[offset],
         sizeof(float) * stream_size, cudaMemcpyHostToDevice);
       cudaMemcpy(&z_d[offset], &z_p[offset],
         sizeof(float) * stream_size, cudaMemcpyHostToDevice);
       cudaMemsetAsync(&Qr_d[offset], 0,
         sizeof(float) * stream_size, stream[n]);
       cudaMemsetAsync(&Qi_d[offset], 0,
         sizeof(float) * stream_size, stream[n]);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
     /* Call the kernels with respective streams */
     for (n = 0; n < n_streams_q; n++) {
       offset = n * stream_size;
       ComputeQGPUAsync(numK, stream_size, kVals, x_d, y_d, z_d, Qr_d, Qi_d,
         stream[n], offset);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_COPY_ASYNC);
     /* Copy data from the device asynchronously */
     for (n = 0; n < n_streams_q; n++) {
       offset = n * stream_size;
 
       cudaMemcpy(&Qr_p[offset], &Qr_d[offset],
         sizeof(float) * stream_size, cudaMemcpyDeviceToHost);
       cudaMemcpy(&Qi_p[offset], &Qi_d[offset],
         sizeof(float) * stream_size, cudaMemcpyDeviceToHost);
     }
     cudaDeviceSynchronize();
 
     /* Delete the streams */
     for (i = 0; i < n_streams_q; i++) {
       cudaStreamDestroy(stream[i]);
     }
 
     pb_SwitchToTimer(&timers, pb_TimerID_COPY);
     for (i = 0; i < numX; i++) {
       Qr[i] = Qr_p[i];
       Qi[i] = Qi_p[i];
     }
     cudaFreeHost(x_p);
     cudaFreeHost(y_p);
     cudaFreeHost(z_p);
     cudaFreeHost(Qr_p);
     cudaFreeHost(Qi_p);
   } 
 
   /* Freeing up no longer needed memory on GPU */
   cudaFree(z_d);
   cudaFree(y_d);
   cudaFree(x_d);
   cudaFree(Qi_d);
   cudaFree(Qr_d);
   cudaDeviceSynchronize();
   cudaDeviceReset();
 
   if (params->outFile)
   {
     /* Write Q to file */
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
 