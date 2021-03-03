#include "util/cpu_information.h"

#include <iostream>
#include <cstdlib>
#include <iterator>
#include <map>

#include <Windows.h>

namespace cpu_info
{
    static PCSTR szCacheType[] = {
        "Unified",
        "Instruction",
        "Data",
        "Trace"
    };

    void FormatMask(KAFFINITY Mask, PSTR sz)
    {
        sz += sprintf(sz, "%p {", (PVOID)Mask);

        ULONG i = sizeof(KAFFINITY) * 8;
        do
        {
            if (_bittest((PLONG)&Mask, --i))
            {
                sz += sprintf(sz, " #%u,", i);
            }
        } while (i);

        *--sz = '}';
    }

    std::set<int> IdsFromMask(KAFFINITY Mask)
    {
        std::set<int> ids;
        ULONG i = sizeof(KAFFINITY) * 8;
        do
        {
            if (_bittest((PLONG)&Mask, --i))
            {
                ids.emplace(i);
            }
        }
        while (i);

        return ids;
    }

    void SetCacheData(CACHE_RELATIONSHIP cache, std::map<KAFFINITY, cpu_info_t>& mask_to_info)
    {
        std::set<int> setCores = IdsFromMask(cache.GroupMask.Mask);
        
        for (auto it = mask_to_info.begin(); it != mask_to_info.end(); ++it)
        {
            auto& info = it->second;
            if ((it->first & cache.GroupMask.Mask) == cache.GroupMask.Mask && cache.GroupMask.Group == info.group)
            {
                if (cache.Level == 1)
                {
                    if (szCacheType[cache.Type % RTL_NUMBER_OF(szCacheType)] == szCacheType[2] && info.l1d_cache_size == -1)
                    {
                        info.l1d_cache_size = cache.CacheSize / 1024;
                    }
                    else if (szCacheType[cache.Type % RTL_NUMBER_OF(szCacheType)] == szCacheType[1] && info.l1_cache_size == -1)
                    {
                        info.l1_cache_size = cache.CacheSize / 1024;
                    }
                }
                else if (cache.Level == 2 && info.l2_cache_size == -1)
                {
                    info.l2_cache_size = cache.CacheSize / 1024;
                }
                else if (cache.Level == 3 && info.l3_cache_size == -1)
                {
                    info.l3_cache_size = cache.CacheSize / 1024;
                }
            }
        }
    }


    void InsertCacheMaskOfLevel(CACHE_RELATIONSHIP cache, int level, std::map<KAFFINITY, cpu_info_t> &mask_to_info)
    {
        std::set<int> setCores = IdsFromMask(cache.GroupMask.Mask);

        if (level != cache.Level)
            return;

        if (mask_to_info.find(cache.GroupMask.Mask) == mask_to_info.end())
            mask_to_info[cache.GroupMask.Mask] = cpu_info_t{ setCores , level, -1, -1, -1, -1, -1, cache.GroupMask.Group };

    }

    void SetNumaData(NUMA_NODE_RELATIONSHIP numa, std::map<KAFFINITY, cpu_info_t> &mask_to_info)
    {
        std::set<int> setCores = IdsFromMask(numa.GroupMask.Mask);

        for (auto it = mask_to_info.begin(); it != mask_to_info.end(); ++it)
        {
            auto& info = it->second;
            if ((it->first & numa.GroupMask.Mask) == numa.GroupMask.Mask && numa.GroupMask.Group == info.group)
            {
                info.numa_node = numa.NodeNumber;
            }
        }
    }

    void RemoveLogicalCpus(const std::set<std::tuple<int, std::vector<int>>> &procs, std::map<KAFFINITY, cpu_info_t> &mask_to_info)
    {
        for (auto it = mask_to_info.begin(); it != mask_to_info.end(); ++it)
        {
            auto& info = it->second;
            for (const auto& proc : procs)
            {
                if (std::get<0>(proc) != info.group)
                    continue;

                std::vector<int> set_cores = std::get<1>(proc);
                for (int i = 1; i < set_cores.size(); ++i)
                {
                    info.core_ids.erase(set_cores[i]);
                }
            }
        }
    }

    void SetProcessorData(PROCESSOR_RELATIONSHIP proc, std::set<std::tuple<int, std::vector<int>>>& procs)
    {
        if (WORD GroupCount = proc.GroupCount)
        {
            PGROUP_AFFINITY GroupMask = proc.GroupMask;
            do
            {
                std::set<int> set_cores = IdsFromMask(GroupMask->Mask);
                procs.emplace(GroupMask->Group, std::vector<int>(set_cores.begin(), set_cores.end()));
            } while (GroupMask++, --GroupCount);
        }
    }

	std::vector<cpu_info_t> GetCpuInfo()
	{
        std::map<KAFFINITY, cpu_info_t> mask_to_info;
        std::set<std::tuple<int, std::vector<int>>> procs;

        DWORD return_len = 0;
        char* buffer = nullptr; 

        GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer, &return_len);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
        {
            buffer = new char[return_len / sizeof(char)];
            GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer, &return_len);
        }

        char *ptr = buffer;

        while (ptr < buffer + return_len)
        {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX pi = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;

            if (!pi)
                continue;

            switch (pi->Relationship)
            {
            case RelationCache:
                InsertCacheMaskOfLevel(pi->Cache, 3, mask_to_info);
                break;
            case RelationProcessorCore:
                SetProcessorData(pi->Processor, procs);
                break;
            default:
                break;
            }

            ptr += pi->Size;
        }

        ptr = buffer;
        
        while (ptr < buffer + return_len)
        {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX pi = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;
            
            if (!pi)
                continue;

            switch (pi->Relationship)
            {
            case RelationNumaNode:
                SetNumaData(pi->NumaNode, mask_to_info);
                break;
            case RelationCache:
                SetCacheData(pi->Cache, mask_to_info);
                break;
            default:
                break;
            }

            ptr += pi->Size;
        }

        delete[] buffer;

        RemoveLogicalCpus(procs, mask_to_info);

        std::vector<cpu_info_t> cores;
        for (auto it = mask_to_info.begin(); it != mask_to_info.end(); ++it) {
            cores.push_back(it->second);
        }

        return cores;
	}
}