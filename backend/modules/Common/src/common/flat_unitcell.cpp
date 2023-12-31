//
// Created by Phil on 13.01.2022.
//

#include <utility>
#include "common/flat_unitcell.h"

FlatUnitcellV2::FlatUnitcellV2(GisaxsTransformationContainer::FlatShapeContainer shape_container, Vector3<int> repetitions,
                               Vector3<MyType> translation)
        :
        parameters_(std::move(shape_container.parameters)),
        parameter_indices_(std::move(shape_container.parameter_indices)),
        upper_bounds_(std::move(shape_container.upper_bounds)),
        lower_bounds_(std::move(shape_container.lower_bounds)),
        shape_types_(std::move(shape_container.shape_types)),
        positions_(std::move(shape_container.positions)),
        position_indices_(std::move(shape_container.position_indices)),
        repetitions_(repetitions),
        translation_(translation) {
}

const std::vector<Vector2<MyType>> &FlatUnitcellV2::LowerBounds() const {
    return lower_bounds_;
}

const std::vector<ShapeTypeV2> &FlatUnitcellV2::ShapeTypes() const {
    return shape_types_;
}

const std::vector<Vector2<MyType>> &FlatUnitcellV2::UpperBounds() const {
    return upper_bounds_;
}

const std::vector<int> &FlatUnitcellV2::ParameterIndices() const {
    return parameter_indices_;
}

const std::vector<Vector2<MyType>> &FlatUnitcellV2::Parameters() const {
    return parameters_;
}

const std::vector<Vector3<MyType>> &FlatUnitcellV2::Positions() const {
    return positions_;
}

const std::vector<int> &FlatUnitcellV2::PositionIndices() const {
    return position_indices_;
}

const Vector3<int> &FlatUnitcellV2::Repetitions() const {
    return repetitions_;
}

const Vector3<MyType> &FlatUnitcellV2::Translation() const {
    return translation_;
}

std::vector<int> FlatUnitcellV2::LocationCounts() const {
    std::vector<int> location_counts;
    for (int j = 0; j < ShapeTypes().size(); ++j) {
        int loc_start_idx = PositionIndices().at(j);
        int loc_end_idx = (j + 1 < PositionIndices().size())
                          ? PositionIndices().at(
                        j + 1) : Positions().size();
        location_counts.emplace_back(loc_end_idx - loc_start_idx);
    }
    return location_counts;
}
