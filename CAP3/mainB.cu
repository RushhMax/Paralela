#include <stdio.h>
#include <cuda_runtime.h>


__global__ void multi_Matriz_Vector(float* A, float* B, float* C, int N) {

    int fila = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < N) {
        float suma = 0.0f;
        for (int j = 0; j < N; j++) {
            suma += B[fila * N + j] * C[j];
        }
        A[fila] = suma;
    }
}

void Host_Stub(float* hA, float* hB, float* hC, int N) {

    float *dA, *dB, *dC;
    int tam_Matriz = N * N * sizeof(float);
    int tam_Vector = N * sizeof(float);

    cudaMalloc((void**)&dA, tam_Vector);
    cudaMalloc((void**)&dB, tam_Matriz);
    cudaMalloc((void**)&dC, tam_Vector);

    cudaMemcpy(dB, hB, tam_Matriz, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, tam_Vector, cudaMemcpyHostToDevice);

    int hilos_bloque = 256;
    int bloque_malla = (N + hilos_bloque - 1) / hilos_bloque;

    multi_Matriz_Vector<<<bloque_malla, hilos_bloque>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hA, dA, tam_Vector, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}


void imp_Vector(float* V, int N) {
    for (int i = 0; i < N; i++) {
        printf("%6.1f\n", V[i]);
    }
    printf("\n");
}

float calculate_time(float* hB, float* hC, float* hA,
                        int N, int repetitions = 10) {

    float *dA, *dB, *dC;
    int tam_Matriz = N * N * sizeof(float);
    int tam_Vector = N * sizeof(float);

    cudaMalloc(&dA, tam_Vector);
    cudaMalloc(&dB, tam_Matriz);
    cudaMalloc(&dC, tam_Vector);

    cudaMemcpy(dB, hB, tam_Matriz, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, tam_Vector, cudaMemcpyHostToDevice);

    int hilos_bloque = 256;
    int bloque_malla = (N + hilos_bloque - 1) / hilos_bloque;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repetitions; i++) {
        multi_Matriz_Vector<<<bloque_malla, hilos_bloque>>>(dA, dB, dC, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hA, dA, tam_Vector, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repetitions; 
}

int main() {

    int N = 4;

    // Matriz B (4x4)
    float B[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16
    };

    // Vector C
    float C[4] = {1, 1, 1, 1};

    // Vector resultado A
    float A[4];

    // Multiplicación matriz-vector
    Host_Stub(A, B, C, N);

    // Imprimir resultado
    printf("Resultado A = B * C:\n");
    imp_Vector(A, N);

    printf("\n=== ANALISIS MATRIX–VECTOR CUDA ===\n");
    printf("N\tTiempo promedio (ms)\n");

    int sizes[] = {512, 1024, 2048, 4096};
    int numSizes = 4;

    for (int s = 0; s < numSizes; s++) {

        int Ntest = sizes[s];
        size_t tam_Matriz = (size_t)Ntest * Ntest * sizeof(float);
        size_t tam_Vector = Ntest * sizeof(float);

        float* Btest = (float*)malloc(tam_Matriz);
        float* Ctest = (float*)malloc(tam_Vector);
        float* Atest = (float*)malloc(tam_Vector);

        if (!Btest || !Ctest || !Atest) {
            printf("Memoria insuficiente para N=%d\n", Ntest);
            break;
        }

        // Inicialización simple
        for (int i = 0; i < Ntest * Ntest; i++)
            Btest[i] = 1.0f;

        for (int i = 0; i < Ntest; i++)
            Ctest[i] = 1.0f;

        float timeMs = calculate_time(Btest, Ctest, Atest, Ntest, 10);

        printf("%d\t%.4f\n", Ntest, timeMs);

        free(Btest);
        free(Ctest);
        free(Atest);
    }

    return 0;
}
