#include <dlfcn.h>
#include <stdio.h>
#include "hip/hip_runtime.h"
#include "libfoo.h"

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void vector_square_local(T* C_d, const T* A_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}


#define RUN_KERNEL(name) \
    { \
        printf("info: launch 'vector_square' kernel from %s\n", name); \
        void (*vs_kernel)(unsigned, unsigned, float*, const float*, size_t) = nullptr; \
        void* libhandle = dlopen(name, RTLD_NOW | RTLD_LOCAL); \
        if (!libhandle) { \
            printf("%s NOT loaded!\n", name); \
            exit(-1); \
        } \
        vs_kernel = (void (*)(unsigned, unsigned, float*, const float*, size_t)) dlsym(libhandle, "vector_square"); \
        if (!vs_kernel) { \
            printf("vector_square NOT located!\n"); \
            exit(-1); \
        } \
        vs_kernel(blocks, threadsPerBlock, C_d, A_d, N); \
    } \


int main(int argc, char* argv[]) {
    float *A_d, *C_d;
    float *A_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(float);
    static int device = 0;
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s\n", props.name);
#ifdef __HIP_PLATFORM_HCC__
    printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);
#endif
    printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
    }

    printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    CHECK(hipMalloc(&A_d, Nbytes));
    CHECK(hipMalloc(&C_d, Nbytes));

    printf("info: copy Host2Device\n");
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    printf("info: launch 'vector_square' kernel locally\n");
    hipLaunchKernelGGL(vector_square_local<float>, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

    for (int i = 0; i < 2; ++i) {
        RUN_KERNEL("libfoo1.so")
        RUN_KERNEL("libfoo2.so")
        RUN_KERNEL("libfoo3.so")
        RUN_KERNEL("libfoo4.so")
        RUN_KERNEL("libfoo5.so")
        RUN_KERNEL("libfoo6.so")
        RUN_KERNEL("libfoo7.so")
        RUN_KERNEL("libfoo8.so")
        RUN_KERNEL("libfoo9.so")
        RUN_KERNEL("libfoo10.so")
        RUN_KERNEL("libfoo11.so")
        RUN_KERNEL("libfoo12.so")
        RUN_KERNEL("libfoo13.so")
        RUN_KERNEL("libfoo14.so")
        RUN_KERNEL("libfoo15.so")
        RUN_KERNEL("libfoo16.so")
        RUN_KERNEL("libfoo17.so")
        RUN_KERNEL("libfoo18.so")
        RUN_KERNEL("libfoo19.so")
        RUN_KERNEL("libfoo20.so")
        RUN_KERNEL("libfoo21.so")
        RUN_KERNEL("libfoo22.so")
        RUN_KERNEL("libfoo23.so")
        RUN_KERNEL("libfoo24.so")
        RUN_KERNEL("libfoo25.so")
        RUN_KERNEL("libfoo26.so")
        RUN_KERNEL("libfoo27.so")
        RUN_KERNEL("libfoo28.so")
        RUN_KERNEL("libfoo29.so")
        RUN_KERNEL("libfoo30.so")
        RUN_KERNEL("libfoo31.so")
        RUN_KERNEL("libfoo32.so")
        RUN_KERNEL("libfoo33.so")
        RUN_KERNEL("libfoo34.so")
        RUN_KERNEL("libfoo35.so")
        RUN_KERNEL("libfoo36.so")
        RUN_KERNEL("libfoo37.so")
        RUN_KERNEL("libfoo38.so")
        RUN_KERNEL("libfoo39.so")
        RUN_KERNEL("libfoo40.so")
        RUN_KERNEL("libfoo41.so")
        RUN_KERNEL("libfoo42.so")
        RUN_KERNEL("libfoo43.so")
        RUN_KERNEL("libfoo44.so")
        RUN_KERNEL("libfoo45.so")
        RUN_KERNEL("libfoo46.so")
        RUN_KERNEL("libfoo47.so")
        RUN_KERNEL("libfoo48.so")
        RUN_KERNEL("libfoo49.so")
        RUN_KERNEL("libfoo50.so")
        RUN_KERNEL("libfoo51.so")
        RUN_KERNEL("libfoo52.so")
        RUN_KERNEL("libfoo53.so")
        RUN_KERNEL("libfoo54.so")
        RUN_KERNEL("libfoo55.so")
        RUN_KERNEL("libfoo56.so")
        RUN_KERNEL("libfoo57.so")
        RUN_KERNEL("libfoo58.so")
        RUN_KERNEL("libfoo59.so")
        RUN_KERNEL("libfoo60.so")
        RUN_KERNEL("libfoo61.so")
        RUN_KERNEL("libfoo62.so")
        RUN_KERNEL("libfoo63.so")
        RUN_KERNEL("libfoo64.so")
        printf("==========\n");
    }

    printf("info: copy Device2Host\n");
    CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    printf("info: check result\n");
    for (size_t i = 0; i < N; i++) {
        if (C_h[i] != A_h[i] * A_h[i]) {
            CHECK(hipErrorUnknown);
        }
    }
    printf("PASSED!\n");
}
