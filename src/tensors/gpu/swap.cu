#include "cuda_helpers.h"
#include "swap.h"
void copyCpuToGpu(const char * in, char * gpuOut);
void copyGpuToGpu(const char * in, char * gpuOut);

namespace marian {
    namespace swapper {

#ifdef CUDA_FOUND
        void copyCpuToGpu(char * gpuOut, const char * in, size_t count, const marian::DeviceId& deviceId) {
            CUDA_CHECK(cudaSetDevice(deviceId.no));
            CUDA_CHECK(cudaMemcpy(gpuOut, in, count, cudaMemcpyHostToDevice));
        }

        void copyGpuToGpu(char * gpuOut, const char * in, size_t count, const marian::DeviceId& deviceId) {
            CUDA_CHECK(cudaSetDevice(deviceId.no));
            CUDA_CHECK(cudaMemcpy(gpuOut, in, count, cudaMemcpyDeviceToDevice));
        }
#else
        void copyCpuToGpu(char * gpuOut, const char * in, size_t count, const marian::DeviceId& deviceId) {
            ABORT("Copy from CPU to GPU memory is only available with CUDA.");
        }

        void copyGpuToGpu(char * gpuOut, const char * in, size_t count, const marian::DeviceId& deviceId) {
            ABORT("Copy from GPU to GPU memory is only available with CUDA.");
        }
#endif
    }
}
