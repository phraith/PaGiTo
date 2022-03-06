#include "gtest/gtest.h"
#include "gisaxs_cpu_core.h"

#include <vector>
#include <fstream>

class ModelSimulatorTest : public ::testing::TestWithParam<int> {};

TEST(ModelSimulatorTest, CylinderFormFactorTest)
{
    //placeholder test
    std::complex<MyType> qpar {1, 0};
    std::complex<MyType> q {1, 0};
    std::complex<MyType> qz {1, 0};
    int first_parameter_index = 0;
    int first_random_index = 0;
    std::vector<Vector2<MyType>> parameters = {{1, 1}, {2,2} };
    std::vector<MyType> randoms = {0.5, 0.5};

    std::complex<MyType> ff = GisaxsCpuCore::CalculateCylinderFF(qpar, q, qz, first_parameter_index, first_random_index, parameters, randoms);
    EXPECT_FLOAT_EQ(ff.real(), 0.742069304);
    EXPECT_FLOAT_EQ(ff.imag(), 10.4642315);
}

//INSTANTIATE_TEST_SUITE_P(ModelSimulatorTester,
//    ModelSimulatorTest,
//    ::testing::Values(0));