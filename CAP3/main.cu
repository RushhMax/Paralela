#include <stdio.h>
#include <cuda_runtime.h>

__global__ void Hilo_elemento(float* C, float* A, float* B, int N) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int colum = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < N && colum < N) {
        int ind = fila * N + colum;
        C[ind] = A[ind] + B[ind];
    }
}

__global__ void Hilo_fila(float* C, float* A, float* B, int N) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < N) {
        for (int colum = 0; colum < N; colum++) {
            int ind = fila * N + colum;
            C[ind] = A[ind] + B[ind];
        }
    }
}

__global__ void Hilo_columna(float* C, float* A, float* B, int N) {
    int colum = blockIdx.x * blockDim.x + threadIdx.x;

    if (colum < N) {
        for (int fila = 0; fila < N; fila++) {
            int ind = fila * N + colum;
            C[ind] = A[ind] + B[ind];
        }
    }
}


void Host_stub(float* hC, float* hA, float* hB, int N, void (*kernel)(float*, float*, float*, int), dim3 grid, dim3 block) {
    float *dA, *dB, *dC;
    int tam = N * N * sizeof(float);

    cudaMalloc((void**)&dA, tam);
    cudaMalloc((void**)&dB, tam);
    cudaMalloc((void**)&dC, tam);

    cudaMemcpy(dA, hA, tam, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, tam, cudaMemcpyHostToDevice);

    kernel<<<grid, block>>>(dC, dA, dB, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, tam, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void imprimir_Matriz(float* M, int N) {
    for (int i = 0; i < N * N; i++) {
        printf("%6.1f ", M[i]);
        if ((i + 1) % N == 0) printf("\n");
    }
    printf("\n");
}


float measureKernelTime(void (*kernel)(float*, float*, float*, int), float* hA, float* hB, float* hC, int N, dim3 grid, dim3 block, int repetitions = 10) {
    float *dA, *dB, *dC;
    int tam = N * N * sizeof(float);

    cudaMalloc(&dA, tam);
    cudaMalloc(&dB, tam);
    cudaMalloc(&dC, tam);

    cudaMemcpy(dA, hA, tam, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, tam, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repetitions; i++) {
        kernel<<<grid, block>>>(dC, dA, dB, N);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repetitions; 
}

int main() {
    printf("\n=== VERIFICACIÓN DE RESULTADO (N = 4) ===\n");

    int Ntest = 4;
    float Atest[16], Btest[16], Ctest[16];

    for (int i = 0; i < 16; i++) {
        Atest[i] = 1.0f;
        Btest[i] = 2.0f;
    }

    // ---- Kernel: 1 hilo por elemento ----
    dim3 blockElem(16, 16);
    dim3 gridElem(1, 1);

    Host_stub(
        Ctest,
        Atest,
        Btest,
        Ntest,
        Hilo_elemento,
        gridElem,
        blockElem
    );

    printf("Kernel: 1 hilo por elemento\n");
    imprimir_Matriz(Ctest, Ntest);

    // ---- Kernel: 1 hilo por fila ----
    dim3 blockfila(256);
    dim3 gridfila(1);

    Host_stub(
        Ctest,
        Atest,
        Btest,
        Ntest,
        Hilo_fila,
        gridfila,
        blockfila
    );

    printf("Kernel: 1 hilo por fila\n");
    imprimir_Matriz(Ctest, Ntest);

    // ---- Kernel: 1 hilo por columumna ----
    dim3 blockcolum(256);
    dim3 gridcolum(1);

    Host_stub(
        Ctest,
        Atest,
        Btest,
        Ntest,
        Hilo_columna,
        gridcolum,
        blockcolum
    );

    printf("Kernel: 1 hilo por columumna\n");
    imprimir_Matriz(Ctest, Ntest);


    int tams[] = {512, 1024, 2048, 4096};
    int numtams = 4;

    printf("\n=== ANALISIS CUDA: SUMA DE MATRICES ===\n");
    printf("N\tElemento(ms)\tFila(ms)\tcolumumna(ms)\n");

    for (int s = 0; s < numtams; s++) {

        int N = tams[s];
        size_t bytes = (size_t)N * N * sizeof(float);

        float* A = (float*)malloc(bytes);
        float* B = (float*)malloc(bytes);
        float* C = (float*)malloc(bytes);

        if (!A || !B || !C) {
            printf("Memoria insuficiente para N = %d\n", N);
            break;
        }

        // Inicialización
        for (int i = 0; i < N * N; i++) {
            A[i] = 1.0f;
            B[i] = 2.0f;
        }

        // ---- 1 hilo por elemento ----
        dim3 block1(16, 16);
        dim3 grid1((N + 15) / 16, (N + 15) / 16);

        float time_elemen = measureKernelTime(
            Hilo_elemento, A, B, C, N, grid1, block1, 10
        );

        // ---- 1 hilo por fila ----
        dim3 block2(256);
        dim3 grid2((N + 255) / 256);

        float time_fila = measureKernelTime(
            Hilo_fila, A, B, C, N, grid2, block2, 10
        );

        // ---- 1 hilo por columumna ----
        dim3 block3(256);
        dim3 grid3((N + 255) / 256);

        float time_colum = measureKernelTime(
            Hilo_columna, A, B, C, N, grid3, block3, 10
        );

        printf("%d\t%.3f\t\t%.3f\t\t%.3f\n",
               N, time_elemen, time_fila, time_colum);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}