//
// Created by Phil on 11.01.2022.
//

#include "common/unitcell_v2.h"

#include <utility>

BoundedDistribution::BoundedDistribution(MyType mean, Bounds mean_bounds, MyType stddev, Bounds stddev_bounds)
        :
        mean_(mean),
        mean_bounds_(mean_bounds),
        stddev_(stddev),
        stddev_bounds_(stddev_bounds) {}

Vector2<MyType> BoundedDistribution::PackDistribution() const {
    return Vector2<MyType>{mean_, stddev_};
}

Vector2<MyType> BoundedDistribution::PackUpperBounds() const {
    return Vector2<MyType>{mean_bounds_.Upper(), stddev_bounds_.Upper()};
}

Vector2<MyType> BoundedDistribution::PackLowerBounds() const {
    return Vector2<MyType>{mean_bounds_.Lower(), stddev_bounds_.Upper()};
}

Sphere::Sphere(const BoundedDistribution &radius, std::vector<Vector3<MyType>> positions)
        :
        Shape(std::move(positions)),
        radius_(radius) {}

std::vector<Vector2<MyType>> Sphere::PackParameters() const {
    return std::vector<Vector2<MyType>>{radius_.PackDistribution()};
}

std::vector<Vector2<MyType>> Sphere::PackUpperBounds() const {
    return std::vector<Vector2<MyType>>{radius_.PackUpperBounds()};
}

std::vector<Vector2<MyType>> Sphere::PackLowerBounds() const {
    return std::vector<Vector2<MyType>>{radius_.PackLowerBounds()};
}

ShapeTypeV2 Sphere::Type() const {
    return ShapeTypeV2::sphere;
}

Bounds::Bounds(MyType lower, MyType upper)
        :
        lower_(lower),
        upper_(upper) {}

Vector2<MyType> Bounds::Pack() {
    return Vector2<MyType>{lower_, upper_};
}

MyType Bounds::Upper() const {
    return upper_;
}

MyType Bounds::Lower() const {
    return lower_;
}

const std::vector<std::unique_ptr<Shape>> &UnitcellV2::Shapes() const{
    return shapes_;
}


UnitcellV2::UnitcellV2(std::vector<std::unique_ptr<Shape>> shapes, Vector3<int> repetitions, Vector3<MyType> translation)
        :
        shapes_(std::move(shapes)),
        repetitions_(repetitions),
        translation_(translation) {

}

const Vector3<MyType> &UnitcellV2::Translation() const{
    return translation_;
}

const Vector3<int> &UnitcellV2::Repetitions() const{
    return repetitions_;
}

int UnitcellV2::RvsCount() const{
    int count = 0;
    for (auto & shape : shapes_) {
        count += shape->PackParameters().size();
    }
    return count;
}

std::vector<MyType> UnitcellV2::CurrentParams() const{
    std::vector<MyType> params;
    for (auto &shape : shapes_) {
        auto parameters = shape->PackParameters();
        for (auto &parameter : parameters) {
            params.emplace_back(parameter.x);
            params.emplace_back(parameter.y);
        }
    }
    return params;
}

int UnitcellV2::ShapeCount() const{
    return shapes_.size();
}

std::vector<ShapeTypeV2> UnitcellV2::Types() const{
    std::vector<ShapeTypeV2> types;
    for (auto &shape:shapes_) {
        types.emplace_back(shape->Type());
    }
    return types;
}

std::vector<Vector3<MyType>> UnitcellV2::Locations() const{
    std::vector<Vector3<MyType>> locations;
    for (auto &shape: shapes_) {
        for (const auto &position : shape->Positions()) {
            locations.emplace_back(position);
        }
    }
    return locations;
}

std::vector<int> UnitcellV2::LocationCounts() const{
    std::vector<int> out;
    for (auto &shape: shapes_) {
        out.emplace_back(shape->Positions().size());
    }
    return out;
}

Shape::Shape(std::vector<Vector3<MyType>> positions)
        :
        positions_(std::move(positions)) {


}

const std::vector<Vector3<MyType>> &Shape::Positions() const {
    return positions_;
}

ShapeTypeV2 Cylinder::Type() const {
    return ShapeTypeV2::cylinder;
}

std::vector<Vector2<MyType>> Cylinder::PackLowerBounds() const {
    return std::vector<Vector2<MyType>>{radius_.PackLowerBounds(), height_.PackLowerBounds()};
}

std::vector<Vector2<MyType>> Cylinder::PackUpperBounds() const {
    return std::vector<Vector2<MyType>>{radius_.PackUpperBounds(), height_.PackUpperBounds()};
}

Cylinder::Cylinder(const BoundedDistribution &radius, const BoundedDistribution &height,
                   std::vector<Vector3<MyType>> positions)
        :
        Shape(std::move(positions)),
        radius_(radius),
        height_(height) {

}

std::vector<Vector2<MyType>> Cylinder::PackParameters() const {
    return std::vector<Vector2<MyType>>{radius_.PackDistribution(), height_.PackDistribution()};
}
