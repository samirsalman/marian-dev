#pragma once
#include <stdlib.h>
#include "common/definitions.h"
#include "common/logging.h"
namespace marian {
    namespace swapper {
        void copyCpuToGpu(char * gpuOut, const char * in, size_t count, const marian::DeviceId& deviceId);
        void copyGpuToGpu(char * gpuOut, const char * in, size_t count, const marian::DeviceId& deviceId);
    }
}
