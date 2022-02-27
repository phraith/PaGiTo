//
// Created by Phil on 13.01.2022.
//

#include <utility>
#include "common/flat_unitcell.h"

FlatUnitcellV2::FlatUnitcellV2(std::vector<MyType2> parameters, std::vector<int> parameter_indices,
                               std::vector<MyType2> upper_bounds, std::vector<MyType2> lower_bounds,
                               std::vector<ShapeTypeV2> shape_types, std::vector<MyType3> positions,
                               std::vector<int> postion_indices, MyType3I repetitons, MyType3 translation)
        :
        parameters_(std::move(parameters)),
        parameter_indices_(std::move(parameter_indices)),
        upper_bounds_(std::move(upper_bounds)),
        lower_bounds_(std::move(lower_bounds)),
        shape_types_(std::move(shape_types)),
        positions_(std::move(positions)),
        position_indices_(std::move(postion_indices)),
        repetitons_(repetitons),
        translation_(translation) {
}

const std::vector<MyType2> &FlatUnitcellV2::LowerBounds() const{
    return lower_bounds_;
}

const std::vector<ShapeTypeV2> &FlatUnitcellV2::ShapeTypes() const{
    return shape_types_;
}

const std::vector<MyType2> &FlatUnitcellV2::UpperBounds() const{
    return upper_bounds_;
}

const std::vector<int> &FlatUnitcellV2::ParameterIndices() const{
    return parameter_indices_;
}

const std::vector<MyType2> &FlatUnitcellV2::Parameters() const{
    return parameters_;
}

const std::vector<MyType3> &FlatUnitcellV2::Positions() const{
    return positions_;
}

const std::vector<int> &FlatUnitcellV2::PositionIndices() const{
    return position_indices_;
}

const MyType3I &FlatUnitcellV2::Repetitons() const{
    return repetitons_;
}

const MyType3 &FlatUnitcellV2::Translation() const{
    return translation_;
}
