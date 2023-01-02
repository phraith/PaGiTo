//
// Created by Phil on 27.03.2022.
//

#include "binary_serialization_utility.h"

using namespace GisaxsTransformationContainer;

std::vector<LineProfileContainer>
BinarySerializationUtility::ReadLineProfiles(const std::vector<std::byte> &data) {
    if (data.size() < sizeof(int)) {
        return {};
    }
    size_t consumed_bytes = 0;
    int lp_count = *reinterpret_cast<int32_t const *>(&data[consumed_bytes]);
    consumed_bytes += sizeof(int);

    std::vector<LineProfileContainer> line_profiles;
    for (int i = 0; i < lp_count; ++i) {
        if (data.size() < consumed_bytes + sizeof(int)) {
            return {};
        }

        int lp_px_count = *reinterpret_cast<int32_t const *>(&data[consumed_bytes]);
        if (data.size() < consumed_bytes + lp_px_count * sizeof(double) + lp_px_count * sizeof(int)) {
            return {};
        }

        consumed_bytes += sizeof(int);
        std::vector<double> lp_intensities(lp_px_count);
        std::copy(reinterpret_cast<const double *>(&data[consumed_bytes]),
                  reinterpret_cast<const double *>(&data[consumed_bytes] + lp_px_count * sizeof(double)),
                  &lp_intensities[0]);

        std::vector<MyType> casted_intensities(lp_px_count);
        std::transform(lp_intensities.begin(), lp_intensities.end(), casted_intensities.begin(),
                       [](double x) { return static_cast<MyType>(x); });

        consumed_bytes += lp_px_count * sizeof(double);

        std::vector<int> lp_offsets(lp_px_count);
        std::copy(reinterpret_cast<const int *>(&data[consumed_bytes]),
                  reinterpret_cast<const int *>(&data[consumed_bytes] + lp_px_count * sizeof(int)),
                  &lp_offsets[0]);

        consumed_bytes += lp_px_count * sizeof(int);
        line_profiles.emplace_back(LineProfileContainer{casted_intensities, lp_offsets});
    }

    if (consumed_bytes != data.size()) {
        return {};
    }

    return line_profiles;
}

std::vector<SimulationTargetData>
BinarySerializationUtility::ReadSimulationTargetData(const std::vector<std::byte> &data) {
    if (data.size() < sizeof(int)) {
        return {};
    }
    size_t consumed_bytes = 0;
    int target_count = *reinterpret_cast<int32_t const *>(&data[consumed_bytes]);
    consumed_bytes += sizeof(int);


    std::vector<SimulationTargetData> simulation_target_data;
    for (int i = 0; i < target_count; ++i) {
        if (data.size() < consumed_bytes + sizeof(int)) {
            return {};
        }

        int start_x = *reinterpret_cast<int32_t const *>(&data[consumed_bytes]);
        consumed_bytes += sizeof(int);

        int start_y = *reinterpret_cast<int32_t const *>(&data[consumed_bytes]);
        consumed_bytes += sizeof(int);

        int end_x = *reinterpret_cast<int32_t const *>(&data[consumed_bytes]);
        consumed_bytes += sizeof(int);

        int end_y = *reinterpret_cast<int32_t const *>(&data[consumed_bytes]);
        consumed_bytes += sizeof(int);

        SimulationTargetDefinition simulation_target_definition;
        simulation_target_definition.start = {start_x, start_y};
        simulation_target_definition.end = {end_x, end_y};

        int lp_px_count = *reinterpret_cast<int32_t const *>(&data[consumed_bytes]);
        consumed_bytes += sizeof(int);
        if (data.size() < consumed_bytes + lp_px_count * sizeof(double)) {
            return {};
        }

        std::vector<double> lp_intensities(lp_px_count);
        std::copy(reinterpret_cast<const double *>(&data[consumed_bytes]),
                  reinterpret_cast<const double *>(&data[consumed_bytes] + lp_px_count * sizeof(double)),
                  &lp_intensities[0]);

        std::vector<MyType> casted_intensities(lp_px_count);
        std::transform(lp_intensities.begin(), lp_intensities.end(), casted_intensities.begin(),
                       [](double x) { return static_cast<MyType>(x); });

        consumed_bytes += lp_px_count * sizeof(double);
        simulation_target_data.emplace_back(SimulationTargetData{casted_intensities, simulation_target_definition});
    }

    return simulation_target_data;
}

