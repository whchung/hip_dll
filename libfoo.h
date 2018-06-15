#include <stdio.h>
#include <hip/hip_runtime.h>

#ifndef __LIBFOO_H__
#define __LIBFOO_H__

extern "C" {
/*
 * Host wrapper function for vector_square
 */
void vector_square(unsigned blocks, unsigned threadsPerBlock, float* C_d, const float* A_d, size_t N);

}
#endif
