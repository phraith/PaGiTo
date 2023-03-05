#ifndef MODEL_FITTER_CORE_MODEL_FITTER_H
#define MODEL_FITTER_CORE_MODEL_FITTER_H

#include "common/service.h"
#include "common/standard_defs.h"
#include "rabbitmq_connection.h"
#include <barrier>
#include <thread>

class ModelFitterV2 : public Service {
public:
    explicit ModelFitterV2(std::shared_ptr<RabbitMqConnection> connection);

    ~ModelFitterV2();

    [[nodiscard]] std::string ServiceName() const override;

    [[nodiscard]]RequestResult HandleRequest(const std::string &request) override;
    static std::vector<double> ConvertFlat(const std::vector<Vector2<MyType>> &input);
private:
    static double Fitness(const std::vector<double> &parameters);

    std::shared_ptr<RabbitMqConnection> connection_;
    uint16_t  consumer_channel_;
    uint16_t  publisher_channel_;
    std::string consumer_queue_name_;

};

#endif