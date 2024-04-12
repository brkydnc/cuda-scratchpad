#include <stdio.h>
#include <time.h>
#include <curand.h>

#define WIDTH 1024
#define BLOCK_SIZE 32
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

void fill(float* A) {
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, clock());
    curandGenerateUniform(rng, A, WIDTH * WIDTH);
    cudaDeviceSynchronize();
    curandDestroyGenerator(rng);
}

int main() {
    float *A_d, *B_d, *C_d, *C;

    C = (float*) malloc(SIZE);

    cudaError_t error_A = cudaMalloc(&A_d, SIZE);
    cudaError_t error_B = cudaMalloc(&B_d, SIZE);
    cudaError_t error_C = cudaMalloc(&C_d, SIZE);

    if (error_A || error_B || error_C) {
        printf("(Error A): %s: %s\n", cudaGetErrorName(error_A), cudaGetErrorString(error_A));
        printf("(Error B): %s: %s\n", cudaGetErrorName(error_B), cudaGetErrorString(error_B));
        printf("(Error C): %s: %s\n", cudaGetErrorName(error_C), cudaGetErrorString(error_C));
    }

    fill(A_d);
    fill(B_d);

    dim3 gridDim(BLOCKS, BLOCKS);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    multiply<<<gridDim, blockDim>>>(A_d, B_d, C_d);

    cudaMemcpy(C, C_d, SIZE, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(C);

    return 0;
}
