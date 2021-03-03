#include "gtest/gtest.h"

#include <vector>
#include <fstream>

#include <common/experimental_model.h>
#include <common/sample.h>
#include <common/layer.h>
#include <common/detector.h>

#include "standard_vector_types.h"

class ModelSimulatorTest : public ::testing::TestWithParam<int> {};

TEST_P(ModelSimulatorTest, ParameterMap)
{
    // std::ifstream ifs("../../config/cylinders-hcp.json");
    // json gisaxs_in = json::parse(ifs);
    // const auto& detector_data = gisaxs_in.at("instrumentation").at("detector");
    // const auto& computation = gisaxs_in.at("computation");
    // const auto& scattering = gisaxs_in.at("scattering");

    // double pixelsize = detector_data.at("pixelsize");
    // double sdd = detector_data.at("sdd");
    // Vec2D directbeam = { detector_data.at("directbeam")[0], detector_data.at("directbeam")[1] };
    // Vec2D resolution = { computation.at("resolution")[0], computation.at("resolution")[1] };
    // double min_alphai = scattering.at("alphai").at("min");

    //Detector d(pixelsize, resolution, directbeam);

    //std::vector<double> current_values = { 1, 2, 3 };

    //std::vector<FittingParameter> params;
    //params.emplace_back("radius", Vec2D{ 0, 1 }, Vec2D{ 0, 0.5 });
    //params.emplace_back("width", Vec2D{ 0, 1 }, Vec2D{ 0, 0.5 });
    //params.emplace_back("height", Vec2D{ 0, 1 }, Vec2D{ 0, 0.5 });

    //std::unique_ptr<ImageData> i = std::make_unique<ImageData>(std::vector<my_type2>(4096, 1), std::vector<int>(4096, 1));
    //
    //ExperimentalModel m(Detector{ 0.172, {300, 600}, {1,1} }, BeamConfiguration{ 0.1, {1,1}, 0.5, 0.1 }, Sample{ Layer {0, 0, -1, 0, 0} }, 0.5);

    //const auto& name_value_map = m.ParamsAsNameValueMap(current_values);

    //EXPECT_EQ(name_value_map.at("radius"), 1);
    //EXPECT_EQ(name_value_map.at("width"), 2);
    //EXPECT_EQ(name_value_map.at("height"), 3);

    //EXPECT_THROW(m.ParamsAsNameValueMap({1,2,3,4}), std::invalid_argument);
}

TEST_P(ModelSimulatorTest, PropagationCoefficients)
{
    // double pi = 3.14159265359;

    // int nz = 600;
    
    // Sample l(Layer(4.88E-06, 7.37E-08, -1, 0, 0));
    // l.InsertLayer(0, 0, 0, 0);

    // double alpha_i = 0.3 * (pi / 180.0);
    // std::vector<double> alpha_fs(nz);

    // double stepsize = (0.0145 + alpha_i) / nz;
    // double current_alpha_f = -alpha_i - stepsize;
    // std::for_each(alpha_fs.begin(), alpha_fs.end(), [&](double& alpha_f) { alpha_f = current_alpha_f + stepsize; current_alpha_f += stepsize; });

    // double k0 = 2 * pi / 0.123984;

    // std::vector<MyType4> dwbas = l.PropagationCoeffs(alpha_i, alpha_fs, k0, 0);
}

INSTANTIATE_TEST_SUITE_P(ModelSimulatorTester,
    ModelSimulatorTest,
    ::testing::Values(0, 1024, 131072, 21039213));