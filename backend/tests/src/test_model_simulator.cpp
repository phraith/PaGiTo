#include "gtest/gtest.h"

#include <vector>
#include <fstream>

class ModelSimulatorTest : public ::testing::TestWithParam<int> {};

TEST_P(ModelSimulatorTest, TestTest1)
{
    EXPECT_EQ(4, 4);
}

TEST_P(ModelSimulatorTest, TestTest2)
{
    auto x = GetParam();
    EXPECT_EQ(x, x);
}

INSTANTIATE_TEST_SUITE_P(ModelSimulatorTester,
    ModelSimulatorTest,
    ::testing::Values(0, 1024, 131072, 21039213));