#include "common/unitcell.h"
#include <algorithm>
#include <iterator>
#include <cassert>
#include <iostream>
Unitcell::Unitcell(MyType3I repetitions, MyType3 distances)
	:
	repetitions_(repetitions),
	distances_(distances),
	rvs_(0)
{
}

Unitcell::Unitcell(const Unitcell& unitcell)
	:
	left_bounds_(unitcell.left_bounds_),
	right_bounds_(unitcell.right_bounds_),
	types_(unitcell.types_),
	repetitions_(unitcell.repetitions_),
	distances_(unitcell.distances_),
	rvs_(unitcell.rvs_)
{
	for (const auto& shape : unitcell.shapes_)
	{
		shapes_.emplace_back(shape->clone());
	}
}

Unitcell::~Unitcell()
{
}

void Unitcell::InsertShape(std::unique_ptr<Shape> shape, ShapeType type)
{
	rvs_ += shape->ParamsCount();

	for (const auto& elem : shape->Bounds())
	{
		left_bounds_.emplace_back(elem.x);
		right_bounds_.emplace_back(elem.y);
	}

	shapes_.emplace_back(std::move(shape));
	types_.emplace_back(type);
}

const std::vector<std::unique_ptr<Shape>> & Unitcell::Shapes() const
{
	return shapes_;
}

const MyType3I& Unitcell::Repetitions() const
{
	return repetitions_;
}

const MyType3& Unitcell::Distances() const
{
	return distances_;
}

int Unitcell::RvsCount() const
{
	return rvs_;
}

int Unitcell::ShapeCount() const
{
	return shapes_.size();
}

const std::vector<double>& Unitcell::LeftBounds() const
{
	return left_bounds_;
}

const std::vector<double>& Unitcell::RightBounds() const
{
	return right_bounds_;
}

const std::vector<ShapeType>& Unitcell::Types() const
{
	return types_;
}

std::vector<MyType> Unitcell::CurrentParams() const
{
	std::vector<MyType> current_params;
	for (const auto& shape : shapes_)
	{
		for (MyType2& distr : shape->Params())
		{
			current_params.emplace_back(distr.x);
			current_params.emplace_back(distr.y);
		}
	}
	return current_params;
}

std::vector<MyType3> Unitcell::Locations() const
{
	std::vector<MyType3> locations;
	for (const auto& shape : shapes_)
	{
		for (const auto & location : shape->Locations())
			locations.emplace_back(location);
	}
	return locations;
}

std::vector<int> Unitcell::LocationCounts() const
{
	std::vector<int> location_counts;
	for (const auto& shape : shapes_)
	{
		location_counts.emplace_back(shape->Locations().size());
	}

	return location_counts;
}

std::vector<FittedShape> Unitcell::FittedShapes() const
{
	std::vector<FittedShape> fitted_shapes;
	for (const auto& shape : shapes_)
	{
		fitted_shapes.emplace_back(FittedShape{ ShapeTypeToString(shape->Type()), shape->FittedParams() });
	}
	return fitted_shapes;
}

void Unitcell::UpdateRvs(const std::vector<MyType>& dv) const
{
	assert(dv.size() == 2 * RvsCount());

	int rvs_idx = 0;

	for (const auto& shape : Shapes())
	{
		switch (shape->Type())
		{
		case ShapeType::kCylinder:
		{
			Cylinder* cylinder = dynamic_cast<Cylinder*> (shape.get());
			cylinder->UpdateDistributions(MyType2{ dv[rvs_idx], dv[rvs_idx + 1] }, MyType2{ dv[rvs_idx + 2], dv[rvs_idx + 3] });
			break;
		}
		case ShapeType::kSphere:
		{
			Sphere* sphere = dynamic_cast<Sphere*> (shape.get());
			sphere->UpdateDistributions(MyType2{ dv[rvs_idx], dv[rvs_idx + 1] });
			break;
		}
		case ShapeType::kTrapezoid:
		{
			Trapezoid* trapezoid = dynamic_cast<Trapezoid*> (shape.get());
			trapezoid->UpdateDistributions(MyType2{ dv[rvs_idx], dv[rvs_idx + 1] }, MyType2{ dv[rvs_idx + 2], dv[rvs_idx + 3] }, MyType2{ dv[rvs_idx + 4], dv[rvs_idx + 5] });
			break;
		}
		default:
			break;
		}

		rvs_idx += 2 * shape->ParamsCount();
	}
}


Cylinder::Cylinder(MyType2 radii_distr, MyType2 height_distr, std::vector<MyType3> locations)
	:
	radii_distr_(radii_distr),
	height_distr_(height_distr),
	radii_mean_bounds_({ radii_distr.x,radii_distr.x }),
	radii_stddev_bounds_({ radii_distr.y, radii_distr.y }),
	height_mean_bounds_({ height_distr.x, height_distr.x }),
	height_stddev_bounds_({ height_distr.y, height_distr.y }),
	locations_(locations)
{
}

Cylinder::Cylinder(MyType2 radii_mean_bounds, MyType2 radii_stddev_bounds, MyType2 height_mean_bounds, MyType2 height_stddev_bounds, std::vector<MyType3> locations)
	:
	radii_distr_({ radii_mean_bounds.x, radii_stddev_bounds.x}),
	height_distr_({ height_mean_bounds.x,  height_stddev_bounds.y}),
	radii_mean_bounds_(radii_mean_bounds),
	radii_stddev_bounds_(radii_stddev_bounds),
	height_mean_bounds_(height_mean_bounds),
	height_stddev_bounds_(height_stddev_bounds),
	locations_(locations)
{
}

Cylinder::~Cylinder()
{
}

void Cylinder::UpdateDistributions(MyType2 radii_distr, MyType2 height_distr)
{
	radii_distr_ = radii_distr;
	height_distr_ = height_distr;
}

int Cylinder::ParamsCount() const
{
	return 2;
}

std::unique_ptr<Shape> Cylinder::clone() const
{
	return std::make_unique<Cylinder>(*this);
}

std::vector<MyType2> Cylinder::Params() const
{
	return std::vector<MyType2> {radii_distr_, height_distr_};
}

std::vector<MyType3> Cylinder::Locations() const
{
	return locations_;
}

std::vector<FittedParameter> Cylinder::FittedParams() const
{
	return std::vector <FittedParameter> {FittedParameter{ radii_distr_.x, radii_distr_.y, "radius" }, FittedParameter{ height_distr_.x, height_distr_.y, "height" }};
}

std::vector<MyType2> Cylinder::Bounds() const
{
	return std::vector<MyType2> {radii_mean_bounds_, radii_stddev_bounds_, height_mean_bounds_, height_stddev_bounds_};
}

ShapeType Cylinder::Type() const
{
	return ShapeType::kCylinder;
}


Sphere::Sphere(MyType2 radii_distr, std::vector<MyType3> locations)
	:
	radii_distr_(radii_distr),
	radii_mean_bounds_({ radii_distr.x, radii_distr.x }),
	radii_stddev_bounds_({ radii_distr.y, radii_distr.y }),
	locations_(locations)
{
}

Sphere::Sphere(MyType2 radii_mean_bounds, MyType2 radii_stddev_bounds, std::vector<MyType3> locations)
	:
	radii_distr_({ 0, 0 }),
	radii_mean_bounds_(radii_mean_bounds),
	radii_stddev_bounds_(radii_stddev_bounds),
	locations_(locations)
{
}

Sphere::~Sphere()
{
}

void Sphere::UpdateDistributions(MyType2 radii_distr)
{
	radii_distr_ = radii_distr;
}

int Sphere::ParamsCount() const
{
	return 1;
}

std::unique_ptr<Shape> Sphere::clone() const
{
	return std::make_unique<Sphere>(*this);
}

std::vector<MyType2> Sphere::Params() const
{
	return std::vector<MyType2> {radii_distr_};
}

std::vector<MyType3> Sphere::Locations() const
{
	return locations_;
}

std::vector<FittedParameter> Sphere::FittedParams() const
{
	return std::vector <FittedParameter> {FittedParameter{ radii_distr_.x, radii_distr_.y, "radius" }};
}

std::vector<MyType2> Sphere::Bounds() const
{
	return std::vector<MyType2> {radii_mean_bounds_, radii_stddev_bounds_};
}

ShapeType Sphere::Type() const
{
	return ShapeType::kSphere;
}

Trapezoid::Trapezoid(MyType2 beta_distr, MyType2 L_distr, MyType2 h_distr, std::vector<MyType3> locations)
	:
	beta_distr_(beta_distr),
	L_distr_(L_distr),
	h_distr_(h_distr),
	beta_mean_bounds_({ beta_distr.x,beta_distr.x }),
	beta_stddev_bounds_({ beta_distr.y, beta_distr.y }),
	L_mean_bounds_({ L_distr.x, L_distr.x }),
	L_stddev_bounds_({ L_distr.y, L_distr.y }),
	h_mean_bounds_({ h_distr.x, h_distr.x }),
	h_stddev_bounds_({ h_distr.y, h_distr.y }),
	locations_(locations)
{
	
}

Trapezoid::Trapezoid(MyType2 beta_mean_bounds, MyType2 beta_stddev_bounds, MyType2 L_mean_bounds, MyType2 L_stddev_bounds, MyType2 h_mean_bounds, MyType2 h_stddev_bounds, std::vector<MyType3> locations)
	:
beta_distr_(MyType2{0,0}),
L_distr_(MyType2{ 0,0 }),
h_distr_(MyType2{ 0,0 }),
beta_mean_bounds_(beta_mean_bounds),
beta_stddev_bounds_(beta_stddev_bounds),
L_mean_bounds_(L_mean_bounds),
L_stddev_bounds_(L_stddev_bounds),
h_mean_bounds_(h_mean_bounds),
h_stddev_bounds_(h_stddev_bounds),
locations_(locations)
{

}

Trapezoid::~Trapezoid()
{
}

void Trapezoid::UpdateDistributions(MyType2 beta_distr, MyType2 L_distr, MyType2 h_distr)
{
	beta_distr_ = beta_distr;
	L_distr_ = L_distr;
	h_distr_ = h_distr;
}

int Trapezoid::ParamsCount() const
{
	return 3;
}

std::unique_ptr<Shape> Trapezoid::clone() const
{
	return std::make_unique<Trapezoid>(*this);
}

std::vector<MyType2> Trapezoid::Params() const
{
	return std::vector<MyType2> {beta_distr_, L_distr_, h_distr_};
}

std::vector<MyType3> Trapezoid::Locations() const
{
	return locations_;
}

std::vector<FittedParameter> Trapezoid::FittedParams() const
{
	return std::vector <FittedParameter> {FittedParameter{ beta_distr_.x, beta_distr_.y, "beta" }, 
		FittedParameter{ L_distr_.x, L_distr_.y, "length" },
		FittedParameter{ h_distr_.x, h_distr_.y, "height" }};
}

std::vector<MyType2> Trapezoid::Bounds() const
{
	return std::vector<MyType2> {beta_mean_bounds_, beta_stddev_bounds_, L_mean_bounds_, L_stddev_bounds_, h_mean_bounds_, h_stddev_bounds_};
}

ShapeType Trapezoid::Type() const
{
	return ShapeType::kTrapezoid;
}
