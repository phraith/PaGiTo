#include <iostream>
#include "core/model_fitter_v2.h"
#include "common/standard_defs.h"
#include "core/fitting_description.h"
#include "parameter_definitions/transformation_container.h"
#include "common/timer.h"
#include "common/binary_serialization_utility.h"
//#include "majordomo_utility.h"
#include "cmaes/cmaes_optimizer.h"

#include <utility>
#include <spdlog/spdlog.h>
#include <barrier>


ModelFitterV2::ModelFitterV2(const std::string &broker_address)
        :
//        client_(std::make_shared<majordomo::RabbitMqClient>(broker_address)),
        broker_address_(broker_address) {

}

ModelFitterV2::~ModelFitterV2() = default;

std::string ModelFitterV2::ServiceName() const {
    return "fitting";
}

RequestResult
ModelFitterV2::HandleRequest(const std::string &request, std::vector<std::byte> image_data, const std::string &origin) {


    json data = json::parse(request);

//    json sim_data = data["sim_data"];
//    json fit_data = data["fit_data"];
//
//    int evolutions = fit_data["evolutions"];
//    int individuals = fit_data["individuals"];
//    int populations = fit_data["populations"];


    json sim_data = data;


    int evolutions = 50;
    int individuals = 5;
    int populations = 50;
    auto simulation_target_data = BinarySerializationUtility::ReadSimulationTargetData(image_data);
    ImageData img_data(simulation_target_data);
    auto fit_job = std::make_shared<FitJob>(sim_data, img_data, evolutions, individuals, populations);
    int dim = fit_job->BaseShapes().parameters.size();

    std::function<double(const std::vector<double> &dv)> func = [=](const std::vector<double> &dv) {
//        auto baseShapes = fit_job->BaseShapes();
//        auto test_client = majordomo::RabbitMqClient(broker_address_);
//        std::vector<Vector2<MyType>> parameters;
//        for (int i = 0; i < dv.size(); i += 2) {
//            parameters.emplace_back(Vector2<MyType>{static_cast<MyType>(dv[i]), static_cast<MyType>(dv[i + 1])});
//        }
//
//        baseShapes.parameters = parameters;
//
//        auto updatedJson = GisaxsTransformationContainer::UpdateShapes(fit_job->SimulationData(), baseShapes);
//        spdlog::info("fitness {}", std::hash<std::thread::id>{}(std::this_thread::get_id()));
//        test_client.Send("simulation", updatedJson.dump(), image_data);
//        zmq::multipart_t res = test_client.Recv("simulation");
//
//        auto fitness_bytes = res.pop();
//
//        if (!fitness_bytes.empty()) {
//            MyType* bt = fitness_bytes.data<MyType>();
//            MyType final = *bt;
//            return final;
//        }
        return -1.f;
    };

    std::vector<double> initial(2 * fit_job->BaseShapes().parameters.size());
    for (int i = 0; i < fit_job->BaseShapes().parameters.size(); ++i) {
        const auto &param = fit_job->BaseShapes().parameters[i];
        initial[i] = param.x;
        initial[i + 1] = param.y;
    }

    std::vector<double> upper(2 * fit_job->BaseShapes().upper_bounds.size());
    for (int i = 0; i < fit_job->BaseShapes().upper_bounds.size(); ++i) {
        const auto &param = fit_job->BaseShapes().upper_bounds[i];
        upper[i] = param.x;
        upper[i + 1] = param.y;
    }

    std::vector<double> lower(2 * fit_job->BaseShapes().lower_bounds.size());
    for (int i = 0; i < fit_job->BaseShapes().lower_bounds.size(); ++i) {
        const auto &param = fit_job->BaseShapes().lower_bounds[i];
        lower[i] = param.x;
        lower[i + 1] = param.y;
    }

    CmaesOptimizer o(func, initial, lower, upper, 2.0, 1000);
    std::shared_ptr<Solution> best_solution = o.Optimize();


    return {std::vector<std::byte>{}};
}

std::vector<double> ModelFitterV2::ConvertFlat(const std::vector<Vector2<MyType>> &input) {
    std::vector<double> converted_vector;
    for (const auto &element: input) {
        converted_vector.emplace_back(element.x);
        converted_vector.emplace_back(element.y);
    }

    return converted_vector;
}

double ModelFitterV2::Fitness(const std::vector<double> &parameters) {
    return 0;
}


