//
// Created by Phil on 11.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_UNITCELL_V2_H
#define GISAXSMODELINGFRAMEWORK_UNITCELL_V2_H

#include <vector>
#include "flat_unitcell.h"
#include <common/standard_defs.h>
#include <memory>


class Shape {
public:
    Shape(std::vector<Vector3<MyType>> positions);

    const std::vector<Vector3<MyType>> &Positions() const;

    virtual ~Shape() = default;

    virtual std::vector<Vector2<MyType>> PackParameters() const = 0;

    virtual std::vector<Vector2<MyType>> PackUpperBounds() const = 0;

    virtual std::vector<Vector2<MyType>> PackLowerBounds() const = 0;

    virtual ShapeTypeV2 Type() const = 0;

private:
    const std::vector<Vector3<MyType>> positions_;
};

class Bounds {
public:
    Bounds(MyType lower, MyType upper);

    Vector2<MyType> Pack();

    [[nodiscard]] MyType Upper() const;

    [[nodiscard]] MyType Lower() const;

private:
    MyType lower_;
    MyType upper_;
};


class BoundedDistribution {
public:
    BoundedDistribution(MyType mean, Bounds mean_bounds, MyType stddev, Bounds stddev_bounds);

    Vector2<MyType> PackDistribution() const;

    Vector2<MyType> PackUpperBounds() const;

    Vector2<MyType> PackLowerBounds() const;

private:
    MyType mean_;
    Bounds mean_bounds_;
    MyType stddev_;
    Bounds stddev_bounds_;
};

class Sphere : public Shape {
public:
    explicit Sphere(const BoundedDistribution &radius, std::vector<Vector3<MyType>> positions);

    std::vector<Vector2<MyType>> PackParameters() const override;

    std::vector<Vector2<MyType>> PackUpperBounds() const override;

    std::vector<Vector2<MyType>> PackLowerBounds() const override;

    ShapeTypeV2 Type() const override;

private:
    BoundedDistribution radius_;
};

class Cylinder : public Shape {
public:
    explicit Cylinder(const BoundedDistribution &radius, const BoundedDistribution &height, std::vector<Vector3<MyType>> positions);

    std::vector<Vector2<MyType>> PackParameters() const override;

    std::vector<Vector2<MyType>> PackUpperBounds() const override;

    std::vector<Vector2<MyType>> PackLowerBounds() const override;

    ShapeTypeV2 Type() const override;

private:
    BoundedDistribution radius_;
    BoundedDistribution height_;
};

class UnitcellV2 {
public:
    explicit UnitcellV2(std::vector<std::unique_ptr<Shape>> shapes, Vector3<int> repetitions, Vector3<MyType> translation);

    const std::vector<std::unique_ptr<Shape>> &Shapes() const;
    const Vector3<MyType> &Translation() const;
    const Vector3<int> &Repetitions() const;

    std::vector<MyType> CurrentParams() const;

    int RvsCount() const;

    int ShapeCount() const;

    std::vector<ShapeTypeV2> Types() const;

    std::vector<Vector3<MyType>> Locations() const;

    std::vector<int> LocationCounts() const;

private:
    const std::vector<std::unique_ptr<Shape>> shapes_;
    Vector3<MyType> translation_;
    Vector3<int> repetitions_;
};


#endif //GISAXSMODELINGFRAMEWORK_UNITCELL_V2_H
