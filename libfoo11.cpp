#include <stdio.h>
#include <hip/hip_runtime.h>
#include "libfoo.h"

/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void vector_square(T* C_d, const T* A_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

extern "C" {
/*
 * Host wrapper function for vector_square
 */
void vector_square(unsigned blocks, unsigned threadsPerBlock, float* C_d, const float* A_d, size_t N) {
    printf("Launch vector_square inside shared library %s\n", __FILE__);
    hipLaunchKernelGGL(vector_square<float>, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);
}

}

