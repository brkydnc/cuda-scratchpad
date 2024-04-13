#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

#define WIDTH 4
#define BLOCK_SIZE 2
#define BLOCKS (WIDTH / BLOCK_SIZE)
#define SIZE (sizeof(float) * WIDTH * WIDTH)

__device__ float* submatrix(float* A, int row, int column) {
    return A + WIDTH * BLOCK_SIZE * row + BLOCK_SIZE * column;
}

__device__ float get(float* A, int row, int col) {
    return A[WIDTH * row + col];
}

__device__ void set(float* A, int row, int col, float val) {
    A[WIDTH * row + col] = val;
}

__global__ void multiply(float* A, float* B, float *C) {
    __shared__ float M[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float N[BLOCK_SIZE][BLOCK_SIZE];

    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0.0;

    for (int i = 0; i < BLOCKS; i++) {
        float* Asub = submatrix(A, blockIdx.y, i);
        float* Bsub = submatrix(B, i, blockIdx.x);

        M[row][col] = get(Asub, row, col);
        N[row][col] = get(Bsub, row, col);

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++)
            sum += M[row][j] + N[j][col];

        __syncthreads();
    }

    float* Csub = submatrix(C, blockIdx.y, blockIdx.x);
    set(Csub, row, col, sum);
}

void cublasMultiply(float* B, float *A, float *C) {
    const float alpha { 1.0f };
    const float beta { 0.0f };
    cublasHandle_t handle;

    cublasCreate(&handle);

    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        WIDTH, WIDTH, WIDTH,
        &alpha,
        A, WIDTH,
        B, WIDTH,
        &beta,
        C, WIDTH
    );

    cublasDestroy(handle);
}

void fill(float* A) {
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, 123);
    curandGenerateUniform(rng, A, WIDTH * WIDTH);
    cudaDeviceSynchronize();
    curandDestroyGenerator(rng);
}

void display(float *A) {
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%f,", A[WIDTH * i + j]);
        }
        printf("\n");
    }
}

int main() {
    float *A, *B, *C, *D;
    float *A_d, *B_d, *C_d, *D_d;

    A = (float*) malloc(SIZE);
    B = (float*) malloc(SIZE);
    C = (float*) malloc(SIZE);
    D = (float*) malloc(SIZE);

    cudaError_t error_A = cudaMalloc(&A_d, SIZE);
    cudaError_t error_B = cudaMalloc(&B_d, SIZE);
    cudaError_t error_C = cudaMalloc(&C_d, SIZE);
    cudaError_t error_D = cudaMalloc(&D_d, SIZE);

    if (error_A || error_B || error_C) {
        printf("(Error A): %s: %s\n", cudaGetErrorName(error_A), cudaGetErrorString(error_A));
        printf("(Error B): %s: %s\n", cudaGetErrorName(error_B), cudaGetErrorString(error_B));
        printf("(Error C): %s: %s\n", cudaGetErrorName(error_C), cudaGetErrorString(error_C));
        printf("(Error C): %s: %s\n", cudaGetErrorName(error_D), cudaGetErrorString(error_D));
    }

    fill(A_d);
    fill(B_d);

    cudaMemcpy(A, A_d, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, B_d, SIZE, cudaMemcpyDeviceToHost);

    printf("Matrix A:\n");
    display(A);
    printf("\n");
    printf("Matrix B:\n");
    display(B);

    dim3 gridDim(BLOCKS, BLOCKS);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    multiply<<<gridDim, blockDim>>>(A_d, B_d, C_d);

    cublasMultiply(A_d, B_d, D_d);

    cudaMemcpy(C, C_d, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(D, D_d, SIZE, cudaMemcpyDeviceToHost);

    printf("\n");
    printf("Matrix C (multiply):\n");
    display(C);
    printf("\n");
    printf("Matrix D (cuBLAS):\n");
    display(D);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(D_d);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
