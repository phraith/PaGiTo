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

MyType2 BoundedDistribution::PackDistribution() const {
    return MyType2{mean_, stddev_};
}

MyType2 BoundedDistribution::PackUpperBounds() const {
    return MyType2{mean_bounds_.Upper(), stddev_bounds_.Upper()};
}

MyType2 BoundedDistribution::PackLowerBounds() const {
    return MyType2{mean_bounds_.Lower(), stddev_bounds_.Upper()};
}

Sphere::Sphere(const BoundedDistribution &radius, std::vector<MyType3> positions)
        :
        Shape(std::move(positions)),
        radius_(radius) {}

std::vector<MyType2> Sphere::PackParameters() const {
    return std::vector<MyType2>{radius_.PackDistribution()};
}

std::vector<MyType2> Sphere::PackUpperBounds() const {
    return std::vector<MyType2>{radius_.PackUpperBounds()};
}

std::vector<MyType2> Sphere::PackLowerBounds() const {
    return std::vector<MyType2>{radius_.PackLowerBounds()};
}

ShapeTypeV2 Sphere::Type() const {
    return ShapeTypeV2::sphere;
}

Bounds::Bounds(MyType lower, MyType upper)
        :
        lower_(lower),
        upper_(upper) {}

MyType2 Bounds::Pack() {
    return MyType2{lower_, upper_};
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


UnitcellV2::UnitcellV2(std::vector<std::unique_ptr<Shape>> shapes, MyType3I repetitions, MyType3 translation)
        :
        shapes_(std::move(shapes)),
        repetitions_(repetitions),
        translation_(translation) {

}

const MyType3 &UnitcellV2::Translation() const{
    return translation_;
}

const MyType3I &UnitcellV2::Repetitions() const{
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

std::vector<MyType3> UnitcellV2::Locations() const{
    std::vector<MyType3> locations;
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

Shape::Shape(std::vector<MyType3> positions)
        :
        positions_(std::move(positions)) {


}

const std::vector<MyType3> &Shape::Positions() const {
    return positions_;
}

ShapeTypeV2 Cylinder::Type() const {
    return ShapeTypeV2::cylinder;
}

std::vector<MyType2> Cylinder::PackLowerBounds() const {
    return std::vector<MyType2>{radius_.PackLowerBounds(), height_.PackLowerBounds()};
}

std::vector<MyType2> Cylinder::PackUpperBounds() const {
    return std::vector<MyType2>{radius_.PackUpperBounds(), height_.PackUpperBounds()};
}

Cylinder::Cylinder(const BoundedDistribution &radius, const BoundedDistribution &height,
                   std::vector<MyType3> positions)
        :
        Shape(std::move(positions)),
        radius_(radius),
        height_(height) {

}

std::vector<MyType2> Cylinder::PackParameters() const {
    return std::vector<MyType2>{radius_.PackDistribution(), height_.PackDistribution()};
}
