//
// Created by Phil on 13.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_FLAT_UNITCELL_H
#define GISAXSMODELINGFRAMEWORK_FLAT_UNITCELL_H

#include <vector>
#include "standard_vector_types.h"
#include "standard_defs.h"


class FlatUnitcellV2 {
public:
    FlatUnitcellV2(std::vector<MyType2> parameters, std::vector<int> parameter_indices,
                   std::vector<MyType2> upper_bounds, std::vector<MyType2> lower_bounds,
                   std::vector<ShapeTypeV2> shape_types, std::vector<MyType3> positions,
                   std::vector<int> postion_indices, MyType3I repetitons, MyType3 translation);

    const std::vector<MyType2> &Parameters();

    const std::vector<int> &ParameterIndices();

    const std::vector<MyType2> &UpperBounds();

    const std::vector<MyType2> &LowerBounds();

    const std::vector<ShapeTypeV2> &ShapeTypes();

    const std::vector<MyType3> &Positions();

    const std::vector<int> &PositionIndices();

    const MyType3 &Translation();

    const MyType3I &Repetitons();

private:
    const std::vector<MyType2> parameters_;
    const std::vector<int> parameter_indices_;
    const std::vector<MyType2> upper_bounds_;
    const std::vector<MyType2> lower_bounds_;
    const std::vector<ShapeTypeV2> shape_types_;
    const std::vector<MyType3> positions_;
    const std::vector<int> position_indices_;

    MyType3I repetitons_;
    MyType3 translation_;
};


#endif //GISAXSMODELINGFRAMEWORK_FLAT_UNITCELL_H
