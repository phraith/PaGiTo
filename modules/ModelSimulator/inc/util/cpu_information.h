#ifndef MODEL_SIMULATOR_UTIL_CPU_INFORMATION_HELPER_H
#define MODEL_SIMULATOR_UTIL_CPU_INFORMATION_HELPER_H

#include <vector>
#include <set>

namespace cpu_info
{
    typedef struct cpu_info_t
    {
        std::set<int> core_ids;

        int shared_cache_level;

        int numa_node;
        int l1_cache_size;
        int l1d_cache_size;
        int l2_cache_size;
        int l3_cache_size;

        /* Only used when on windows:
            - only 64 Cores can be handled by one bitmask -> multiple groups
              when using more than 64 cores on windows
        */

        int group;
    }
    cpu_info_t;

    std::vector<cpu_info_t> GetCpuInfo();
}

#endif