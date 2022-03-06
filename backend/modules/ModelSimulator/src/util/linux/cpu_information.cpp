#include "util/cpu_information.h"

#include <fstream>
#include <map>
#include <string>
#include <regex>
#include <set>

#include <experimental/filesystem>

using namespace std::experimental::filesystem;
namespace CpuInfo {


    std::vector<std::string> Split(const std::string &s, const std::string &delimiter) {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        std::vector<std::string> res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
            token = s.substr(pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            res.push_back(token);
        }

        res.push_back(s.substr(pos_start));
        return res;
    }


    int GetCacheLevel(const std::string &cache_dir) {
        std::ifstream cache_level_f(cache_dir + "/level");
        if (cache_level_f.good()) {
            std::string level_str;
            std::getline(cache_level_f, level_str);

            return std::stoi(level_str);
        }
        return -1;
    }

    int GetCacheSize(const std::string &cache_dir) {
        std::ifstream cache_size_f(cache_dir + "/size");
        if (cache_size_f.good()) {
            std::string size_str;
            std::getline(cache_size_f, size_str);

            return std::stoi(size_str);
        }
        return -1;
    }

    std::string GetCacheType(const std::string &cache_dir) {
        std::ifstream cache_type_f(cache_dir + "/type");
        if (cache_type_f.good()) {
            std::string type_str;
            std::getline(cache_type_f, type_str);

            return type_str;
        }
        return "";
    }

    std::string GetCacheMap(const std::string &cache_dir) {
        std::ifstream cache_map_f(cache_dir + "/shared_cpu_map");
        if (cache_map_f.good()) {
            std::string cpu_map;
            std::getline(cache_map_f, cpu_map);

            return cpu_map;
        }
        return "";
    }

    std::vector<std::string> GetCores(const std::string &cpu_core_dir) {
        std::ifstream cpu_core_f(cpu_core_dir + "/topology/core_cpus_list");
        if (cpu_core_f.good()) {
            std::vector<std::string> core_split;
            std::string core_str;
            std::getline(cpu_core_f, core_str);

            return Split(core_str, ",");
        }
        return {};
    }

    int GetNumaNode(const std::string &cpu_core_dir) {
        std::ifstream cpu_node_f(cpu_core_dir + "/topology/physical_package_id");
        if (cpu_node_f.good()) {
            std::string node_str;
            std::getline(cpu_node_f, node_str);

            return std::stoi(node_str);
        }
        return -1;
    }

    int FindCacheSize(int level, std::string type, const std::string &cache_dir) {
        for (const auto &cpu_cache_dir_entry: directory_iterator(cache_dir)) {
            std::vector<std::string> cache_split;
            std::string ccd_string = cpu_cache_dir_entry.path().u8string();
            CpuInfo::Split(ccd_string, "/");

            if (!std::regex_match(cache_split.back(), std::regex("index[0-9]+")))
                continue;

            std::string cache_type_dir = cache_dir + cache_split.back();

            if (GetCacheLevel(cache_type_dir) != level || GetCacheType(cache_type_dir) != type)
                continue;

            return GetCacheSize(cache_type_dir);
        }
        return -1;
    }

    CpuInfo::cpu_info_t ConstructNewCpuInfo(const std::string &cache_dir, const int level, int core_id, int numa_node) {

        int l1_cache_size = FindCacheSize(1, "Instruction", cache_dir);
        int l1d_cache_size = FindCacheSize(1, "Data", cache_dir);
        int l2_cache_size = FindCacheSize(2, "Unified", cache_dir);
        int l3_cache_size = FindCacheSize(3, "Unified", cache_dir);
        return {{}, level, numa_node,
                l1_cache_size, l1d_cache_size,
                l2_cache_size, l3_cache_size, -1};
    }

    std::vector<CpuInfo::cpu_info_t> GetCpuInfo() {

        std::map<std::string, CpuInfo::cpu_info_t> mask_to_info;

        std::string cpu_dir{"/sys/devices/system/cpu/"};
        for (const auto &cpu_dir_entry: directory_iterator(cpu_dir)) {
            std::string cd_string = cpu_dir_entry.path().u8string();
            std::vector<std::string> dir_split = Split(cd_string, "/");

            if (!std::regex_match(dir_split.back(), std::regex("cpu[0-9]+")))
                continue;

            std::string cpu_core_dir = cpu_dir + dir_split.back();
            std::vector<std::string> core_split = GetCores(cpu_core_dir);
            int numa_node = GetNumaNode(cpu_core_dir);
            std::string cache_dir = cpu_core_dir + "/cache/";
            for (const auto &cpu_cache_dir_entry: directory_iterator(cache_dir)) {
                std::string ccd_string = cpu_cache_dir_entry.path().u8string();
                std::vector<std::string> cache_split = Split(ccd_string, "/");

                if (!std::regex_match(cache_split.back(), std::regex("index[0-9]+")))
                    continue;

                int level = GetCacheLevel(cache_dir + cache_split.back());
                if (level == 3) {
                    std::string cpu_map = GetCacheMap(cache_dir + cache_split.back());

                    auto it = mask_to_info.find(cpu_map);

                    if (it != mask_to_info.end())
                        it->second.core_ids.emplace(std::stoi(core_split.at(0)));
                    else
                        mask_to_info[cpu_map] = ConstructNewCpuInfo(cache_dir, level, std::stoi(core_split.at(0)),
                                                                    numa_node);
                }
            }
        }

        std::vector<CpuInfo::cpu_info_t> cache_groups;
        cache_groups.reserve(mask_to_info.size());
        for (auto &it: mask_to_info) {
            cache_groups.push_back(it.second);
        }
        return cache_groups;
    }
}