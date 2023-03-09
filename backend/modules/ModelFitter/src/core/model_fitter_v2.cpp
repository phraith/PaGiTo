#include <iostream>
#include "core/model_fitter_v2.h"
#include "common/standard_defs.h"
#include "core/fitting_description.h"
#include "parameter_definitions/transformation_container.h"
#include "common/timer.h"
#include "common/binary_serialization_utility.h"
//#include "majordomo_utility.h"
#include "cmaes/cmaes_optimizer.h"
#include "rabbitmq_connection.h"

#include <utility>
#include <spdlog/spdlog.h>
#include <barrier>


ModelFitterV2::ModelFitterV2(std::shared_ptr<RabbitMqConnection> connection)
        :
        connection_(connection) {
    std::tuple<std::string, uint16_t> consumer_data = connection_->QueueDeclare();
    consumer_channel_ = std::get<1>(consumer_data);
    consumer_queue_name_ = std::get<0>(consumer_data);
    publisher_channel_ = connection_->NextChannel();

    connection_->RegisterConsumer(consumer_queue_name_, consumer_channel_);
}

ModelFitterV2::~ModelFitterV2() = default;

std::string ModelFitterV2::ServiceName() const {
    return "fitting";
}

RequestResult
ModelFitterV2::HandleRequest(const std::string &request) {




//    json sim_data = data["sim_data"];
//    json fit_data = data["fit_data"];
//
//    int evolutions = fit_data["evolutions"];
//    int individuals = fit_data["individuals"];
//    int populations = fit_data["populations"];




    int evolutions = 50;
    int individuals = 5;
    int populations = 50;

    int message_size = *reinterpret_cast<int32_t const *>(&request[0]);
    std::string config(static_cast<const char *>(&request[sizeof(int32_t)]), message_size);
    int image_size = *reinterpret_cast<int32_t const *>(&request[sizeof(int) + message_size]);

    std::vector<std::byte> image_data(image_size);
    if (image_size > 0) {
        std::copy(reinterpret_cast<const std::byte *>(&request[2 * sizeof(int) + message_size]),
                  reinterpret_cast<const std::byte *>(&request[2 * sizeof(int)] + message_size + image_size),
                  &image_data[0]);
    }
    json sim_data = json::parse(config);
    auto simulation_target_data = BinarySerializationUtility::ReadSimulationTargetData(image_data);
    ImageData img_data(simulation_target_data);
    auto fit_job = std::make_shared<FitJob>(sim_data, img_data, evolutions, individuals, populations);
    int dim = fit_job->BaseShapes().parameters.size();

    std::function<std::vector<double>(const std::vector<std::vector<double>> &input_vectors)> func = [=](
            const std::vector<std::vector<double>> &input_vectors) {

        std::vector<double> results(input_vectors.size());
        for (int k = 0; k < input_vectors.size(); ++k) {
            auto baseShapes = fit_job->BaseShapes();
            std::vector<double> dv = input_vectors.at(k);

            int parameters_count = dv.size() / 2;

            std::vector<Vector2<MyType>> parameters;
            for (int i = 0; i < parameters_count; ++i) {
                parameters.emplace_back(Vector2<MyType>{static_cast<MyType>(dv[i]), static_cast<MyType>(dv[i + parameters_count])});
            }

            baseShapes.parameters = parameters;
            auto updatedJson = GisaxsTransformationContainer::UpdateShapes(fit_job->SimulationData(), baseShapes);
            std::string m = updatedJson.dump();
            int sim_message_size = m.length();

            std::vector<std::byte> sim_message(2 * sizeof(int) + sim_message_size + image_size);
            std::copy(reinterpret_cast<std::byte *>(&sim_message_size),
                      reinterpret_cast<std::byte *>(&sim_message_size + sizeof(int)),
                      &sim_message[0]);

            std::copy(reinterpret_cast<std::byte *>(&m[0]),
                      reinterpret_cast<std::byte *>(&m[0] + m.length()),
                      &sim_message[sizeof(int)]);

            std::copy(reinterpret_cast<const std::byte *>(&image_size),
                      reinterpret_cast<const std::byte *>(&image_size + sizeof(int)),
                      &sim_message[sizeof(int) + m.length()]);

            std::copy(reinterpret_cast<const std::byte *>(&image_data[0]),
                      reinterpret_cast<const std::byte *>(&image_data[0] + image_size),
                      &sim_message[2 * sizeof(int) + m.length()]);

            try {
                connection_->Publish(publisher_channel_, "Simulation", consumer_queue_name_, std::to_string(k),
                                     sim_message);
            }
            catch (const RabbitMqConnectionException &e) {
                connection_->ConnectSafe();
                publisher_channel_ = connection_->NextChannel();

                std::tuple<std::string, uint16_t> consumer_data = connection_->QueueDeclare();
                consumer_channel_ = std::get<1>(consumer_data);
                consumer_queue_name_ = std::get<0>(consumer_data);
                connection_->RegisterConsumer(consumer_queue_name_, consumer_channel_);
            }
        }

        for (int k = 0; k < input_vectors.size(); ++k) {
            try {
                std::tuple<std::string, std::string> res = connection_->Consume();
                std::string message = std::get<0>(res);
                float fitness = *reinterpret_cast<float *>(&message[0]);
                int correlation_id = std::stoi(std::get<1>(res));
                results.at(correlation_id) = fitness;
            }
            catch (const RabbitMqConnectionException &e) {
                connection_->ConnectSafe();
                publisher_channel_ = connection_->NextChannel();

                std::tuple<std::string, uint16_t> consumer_data = connection_->QueueDeclare();
                consumer_channel_ = std::get<1>(consumer_data);
                consumer_queue_name_ = std::get<0>(consumer_data);

                connection_->RegisterConsumer(consumer_queue_name_, consumer_channel_);
            }
        }

//
//        if (!fitness_bytes.empty()) {
//            MyType* bt = fitness_bytes.data<MyType>();
//            MyType final = *bt;
//            return final;
//        }
        return results;
    };

    std::vector<double> initial(2 * fit_job->BaseShapes().parameters.size());
    for (int i = 0; i < fit_job->BaseShapes().parameters.size(); ++i) {
        const auto &upper_x = fit_job->BaseShapes().upper_bounds[i].x;
        const auto &lower_x = fit_job->BaseShapes().lower_bounds[i].x;

        const auto &upper_y = fit_job->BaseShapes().upper_bounds[i].y;
        const auto &lower_y = fit_job->BaseShapes().lower_bounds[i].y;

        initial[i] = (upper_x + lower_x) / 2.0;
        initial[i + 1] = (upper_y + lower_y) / 2.0;
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

    CmaesOptimizer o(func, initial, lower, upper, 2.0, 300);
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


