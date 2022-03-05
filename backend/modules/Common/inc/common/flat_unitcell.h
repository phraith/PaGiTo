//
// Created by Phil on 13.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_FLAT_UNITCELL_H
#define GISAXSMODELINGFRAMEWORK_FLAT_UNITCELL_H

#include <vector>
#include "standard_vector_types.h"
#include "standard_defs.h"
#include "parameter_definitions/transformation_container.h"


class FlatUnitcellV2 {
public:
    FlatUnitcellV2(GisaxsTransformationContainer::FlatShapeContainer shape_container, MyType3I repetitions, MyType3 translation);

    const std::vector<MyType2> &Parameters() const;

    const std::vector<int> &ParameterIndices() const;

    const std::vector<MyType2> &UpperBounds() const;

    const std::vector<MyType2> &LowerBounds() const;

    const std::vector<ShapeTypeV2> &ShapeTypes() const;

    const std::vector<MyType3> &Positions() const;

    const std::vector<int> &PositionIndices() const;

    const MyType3 &Translation() const;

    const MyType3I &Repetitons() const;

    std::vector<int> LocationCounts() const;

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
