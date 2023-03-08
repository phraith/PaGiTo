//
// Created by phili on 23.10.2022.
//

#include "gtest/gtest.h"
#include "cmaes/cmaes_optimizer.h"
#include <vector>

class CmaesTest : public ::testing::TestWithParam<int>
{
};

double QuadraticFunction(const std::vector<double> &parameters)
{
    double x1 = parameters[0];
    double x2 = parameters[1];
    return std::pow((x1 - 3.0), 2) + std::pow((10.0 * (x2 + 2.0)), 2);
}

std::vector<double> QuadraticFunctionBulk(const std::vector<std::vector<double>> &dv)
{
    auto res = std::vector<double>(dv.size());
    for (int i = 0; i < dv.size(); ++i)
    {
        res.at(i) = QuadraticFunction(dv.at(i));
    }
    return res;
}

double EllipsoidFunction(const std::vector<double> &parameters)
{

    if (parameters.size() < 2)
    {
        throw std::exception();
    }

    double sum = 0;
    for (int i = 0; i < parameters.size(); ++i)
    {
        sum += std::pow(1000.0, (double)i / (parameters.size() - 1)) * std::pow(parameters.at(i), 2);
    }
    return sum;
}

std::vector<double> EllipsoidFunctionBulk(const std::vector<std::vector<double>> &dv)
{
    auto res = std::vector<double>(dv.size());
    for (int i = 0; i < dv.size(); ++i)
    {
        res.at(i) = EllipsoidFunction(dv.at(i));
    }
    return res;
}

TEST(CmaesTest, EllipsoidFunction)
{
    int dim = 5;
    std::vector<double> initial(dim, 3);
    CmaesOptimizer o(EllipsoidFunctionBulk, initial, std::vector<double>(), std::vector<double>(), 2.0, 1000);
    std::shared_ptr<Solution> best_solution = o.Optimize();

    EXPECT_NE(nullptr, best_solution);
    for (int i = 0; i < dim; ++i)
    {
        EXPECT_NEAR(0.0, best_solution->Parameters()[i], 1e-7);
    }
}

TEST(CmaesTest, QuadraticFunction)
{
    std::vector<double> initial;
    initial.emplace_back(0);
    initial.emplace_back(0);
    CmaesOptimizer o(QuadraticFunctionBulk, initial, std::vector<double>(), std::vector<double>(), 1.3, 1000);
    std::shared_ptr<Solution> best_solution = o.Optimize();

    EXPECT_NE(nullptr, best_solution);
    EXPECT_FLOAT_EQ(3.0, best_solution->Parameters()[0]);
    EXPECT_FLOAT_EQ(-2.0, best_solution->Parameters()[1]);
}
