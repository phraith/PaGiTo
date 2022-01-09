#ifndef MODEL_SIMULATOR_UTIL_UNITCELL_H
#define MODEL_SIMULATOR_UTIL_UNITCELL_H

#include <vector>
#include "standard_vector_types.h"
#include <common/standard_defs.h>
#include <memory>

class Shape
{
public:
    virtual ~Shape() {};
    virtual int ParamsCount() const = 0;
    virtual std::unique_ptr<Shape> clone() const = 0;
    virtual ShapeType Type() const = 0;
    virtual std::vector<MyType2> Params() const = 0;
    virtual std::vector<MyType3> Locations() const = 0;
    virtual std::vector<FittedParameter> FittedParams() const = 0;
    virtual std::vector<MyType2> Bounds() const = 0;
};

class Cylinder : public Shape
{
public:
    Cylinder(MyType2 radii_distr, MyType2 height_distr, std::vector<MyType3> locations);
    Cylinder(MyType2 radii_mean_bounds, MyType2 radii_stddev_bounds, MyType2 height_mean_bounds, MyType2 height_stddev_bounds, std::vector<MyType3> locations);
    ~Cylinder();
    void UpdateDistributions(MyType2 radii_distr, MyType2 height_distr);
    int ParamsCount() const;
    std::unique_ptr<Shape> clone() const;
    std::vector<MyType2> Params() const;
    std::vector<MyType3> Locations() const;
    std::vector<FittedParameter> FittedParams() const;
    std::vector<MyType2> Bounds() const;

    ShapeType Type() const;
private:
    MyType2 radii_distr_;
    MyType2 radii_mean_bounds_;
    MyType2 radii_stddev_bounds_;

    MyType2 height_distr_;
    MyType2 height_mean_bounds_;
    MyType2 height_stddev_bounds_;

    std::vector<MyType3> locations_;
};

class Trapezoid : public Shape
{
public:
    Trapezoid(MyType2 beta_distr, MyType2 L_distr, MyType2 h_distr, std::vector<MyType3> locations);
    Trapezoid(MyType2 beta_mean_bounds, MyType2 beta_stddev_bounds, MyType2 L_mean_bounds, MyType2 L_stddev_bounds, MyType2 h_mean_bounds, MyType2 h_stddev_bounds, std::vector<MyType3> locations);
    ~Trapezoid();
    void UpdateDistributions(MyType2 beta_distr, MyType2 L_distr, MyType2 h_distr);
    int ParamsCount() const;
    std::unique_ptr<Shape> clone() const;
    std::vector<MyType2> Params() const;
    std::vector<MyType3> Locations() const;
    std::vector<FittedParameter> FittedParams() const;
    std::vector<MyType2> Bounds() const;

    ShapeType Type() const;
private:
    MyType2 beta_distr_;
    MyType2 beta_mean_bounds_;
    MyType2 beta_stddev_bounds_;

    MyType2 L_distr_;
    MyType2 L_mean_bounds_;
    MyType2 L_stddev_bounds_;

    MyType2 h_distr_;
    MyType2 h_mean_bounds_;
    MyType2 h_stddev_bounds_;

    std::vector<MyType3> locations_;
};

class Sphere : public Shape
{
public:
    Sphere(MyType2 radii_distr, std::vector<MyType3> locations);
    Sphere(MyType2 radii_mean_bounds, MyType2 radii_stddev_bounds, std::vector<MyType3> locations);
    ~Sphere();
    void UpdateDistributions(MyType2 radii_distr);
    int ParamsCount() const;
    std::unique_ptr<Shape> clone() const;
    std::vector<MyType2> Params() const;
    std::vector<MyType3> Locations() const;
    std::vector<FittedParameter> FittedParams() const;
    std::vector<MyType2> Bounds() const;

    ShapeType Type() const;
private:
    MyType2 radii_distr_;
    MyType2 radii_mean_bounds_;
    MyType2 radii_stddev_bounds_;

    std::vector<MyType3> locations_;
};

class Unitcell
{
public:
    Unitcell(MyType3I repetitions, MyType3 distances);
    Unitcell(const Unitcell& unitcell);
    ~Unitcell();
    void InsertShape(std::unique_ptr<Shape> shape, ShapeType type);

    const std::vector<std::unique_ptr<Shape>> &Shapes() const;
    const std::vector<ShapeType>& Types() const;
    const MyType3I& Repetitions() const;
    const MyType3& Distances() const;
    
    int RvsCount() const;
    int ShapeCount() const;
    void UpdateRvs(const std::vector<MyType>& dv) const;

    const std::vector<double> &LeftBounds() const;
    const std::vector<double>& RightBounds() const;

    std::vector<MyType> CurrentParams() const;
    std::vector<MyType3> Locations() const;
    std::vector<int> LocationCounts() const;
    std::vector<FittedShape> FittedShapes() const;


private:
    std::vector<std::unique_ptr<Shape>> shapes_;

    std::vector<double> left_bounds_;
    std::vector<double> right_bounds_;

    std::vector<ShapeType> types_;

    MyType3I repetitions_;
    MyType3 distances_;
    
    int rvs_;
};

#endif