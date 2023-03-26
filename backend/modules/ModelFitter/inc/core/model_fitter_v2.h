#ifndef MODEL_FITTER_CORE_MODEL_FITTER_H
#define MODEL_FITTER_CORE_MODEL_FITTER_H

#include "common/service.h"
#include "common/standard_defs.h"
#include "fit_job_client.h"
#include <barrier>
#include <thread>
#include <amqpcpp.h>
#include <amqpcpp/linux_tcp.h>

class ModelFitterV2 : public Service {
public:
    explicit ModelFitterV2(const std::string &host, int port, const std::string &username, const std::string &password);

    ~ModelFitterV2();

    [[nodiscard]] std::string ServiceName() const override;

    [[nodiscard]]RequestResult HandleRequest(const std::string &request) override;

    static std::vector<double> ConvertFlat(const std::vector<Vector2<MyType>> &input);

private:
    static double Fitness(const std::vector<double> &parameters);

    FitJobClient client_;
};

#endif