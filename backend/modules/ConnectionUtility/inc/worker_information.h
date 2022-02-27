//
// Created by Phil on 23.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_WORKER_INFORMATION_H
#define GISAXSMODELINGFRAMEWORK_WORKER_INFORMATION_H

#include "majordomo_utility.h"

class ServiceInformation;

class WorkerInformation {
public:
    WorkerInformation(std::shared_ptr<ServiceInformation> service, majordomo::remote_id_t worker_id, majordomo::ms_time_t expiry);

    [[nodiscard]] const majordomo::remote_id_t &Identity() const;
    [[nodiscard]] majordomo::ms_time_t Expiry() const;
    void Expiry(majordomo::ms_time_t expiry);
    [[nodiscard]] bool HasService() const;
    [[nodiscard]] const std::shared_ptr<ServiceInformation> &AssignedService() const;
    void AssignService(std::shared_ptr<ServiceInformation> service);

private:
    std::shared_ptr<ServiceInformation> service_;
    majordomo::remote_id_t worker_id_;
    majordomo::ms_time_t expiry_;
};

#endif //GISAXSMODELINGFRAMEWORK_WORKER_INFORMATION_H
