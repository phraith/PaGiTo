#ifndef MODEL_FITTER_CORE_MODEL_FITTER_H
#define MODEL_FITTER_CORE_MODEL_FITTER_H

#include "common/service.h"
#include "common/standard_defs.h"
#include <barrier>
#include <thread>

class ModelFitterV2 : public Service {
public:
    explicit ModelFitterV2(const std::string &broker_address);

    ~ModelFitterV2();

    [[nodiscard]] std::string ServiceName() const override;

    [[nodiscard]]RequestResult HandleRequest(const std::string &request, std::vector<std::byte> image_data, const std::string &origin) override;
    static std::vector<double> ConvertFlat(const std::vector<Vector2<MyType>> &input);
private:
    static double Fitness(const std::vector<double> &parameters);

//    std::shared_ptr<majordomo::RabbitMqClient> client_;
    const std::string &broker_address_;


};

#endif