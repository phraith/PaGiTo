//
// Created by Phil on 11.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_UNITCELL_V2_H
#define GISAXSMODELINGFRAMEWORK_UNITCELL_V2_H

#include <vector>
#include "standard_vector_types.h"
#include "flat_unitcell.h"
#include <common/standard_defs.h>
#include <memory>


class Shape {
public:
    Shape(std::vector<MyType3> positions);

    const std::vector<MyType3> &Positions() const;

    virtual ~Shape() = default;

    virtual std::vector<MyType2> PackParameters() const = 0;

    virtual std::vector<MyType2> PackUpperBounds() const = 0;

    virtual std::vector<MyType2> PackLowerBounds() const = 0;

    virtual ShapeTypeV2 Type() const = 0;

private:
    const std::vector<MyType3> positions_;
};

class Bounds {
public:
    Bounds(MyType lower, MyType upper);

    MyType2 Pack();

    [[nodiscard]] MyType Upper() const;

    [[nodiscard]] MyType Lower() const;

private:
    MyType lower_;
    MyType upper_;
};


class BoundedDistribution {
public:
    BoundedDistribution(MyType mean, Bounds mean_bounds, MyType stddev, Bounds stddev_bounds);

    MyType2 PackDistribution() const;

    MyType2 PackUpperBounds() const;

    MyType2 PackLowerBounds() const;

private:
    MyType mean_;
    Bounds mean_bounds_;
    MyType stddev_;
    Bounds stddev_bounds_;
};

class Sphere : public Shape {
public:
    explicit Sphere(const BoundedDistribution &radius, std::vector<MyType3> positions);

    std::vector<MyType2> PackParameters() const override;

    std::vector<MyType2> PackUpperBounds() const override;

    std::vector<MyType2> PackLowerBounds() const override;

    ShapeTypeV2 Type() const override;

private:
    BoundedDistribution radius_;
};

class Cylinder : public Shape {
public:
    explicit Cylinder(const BoundedDistribution &radius, const BoundedDistribution &height, std::vector<MyType3> positions);

    std::vector<MyType2> PackParameters() const override;

    std::vector<MyType2> PackUpperBounds() const override;

    std::vector<MyType2> PackLowerBounds() const override;

    ShapeTypeV2 Type() const override;

private:
    BoundedDistribution radius_;
    BoundedDistribution height_;
};

class UnitcellV2 {
public:
    explicit UnitcellV2(std::vector<std::unique_ptr<Shape>> shapes, MyType3I repetitions, MyType3 translation);

    const std::vector<std::unique_ptr<Shape>> &Shapes() const;
    const MyType3 &Translation() const;
    const MyType3I &Repetitions() const;

    std::vector<MyType> CurrentParams() const;

    int RvsCount() const;

    int ShapeCount() const;

    std::vector<ShapeTypeV2> Types() const;

    std::vector<MyType3> Locations() const;

    std::vector<int> LocationCounts() const;

private:
    const std::vector<std::unique_ptr<Shape>> shapes_;
    MyType3 translation_;
    MyType3I repetitions_;
};


#endif //GISAXSMODELINGFRAMEWORK_UNITCELL_V2_H
