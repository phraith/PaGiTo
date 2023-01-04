//
// Created by Phil on 23.01.2022.
//

#include "worker_information.h"
#include <utility>

WorkerInformation::WorkerInformation(std::shared_ptr<ServiceInformation> service, majordomo::remote_id_t worker_id, majordomo::ms_time_t expiry)
        :
        service_(std::move(service)),
        worker_id_(std::move(worker_id)),
        expiry_(expiry) {

}

const majordomo::remote_id_t &WorkerInformation::Identity() const {
    return worker_id_;
}

majordomo::ms_time_t WorkerInformation::Expiry() const {
    return expiry_;
}

bool WorkerInformation::HasService() const {
    return service_ != nullptr;
}

const std::shared_ptr<ServiceInformation> &WorkerInformation::AssignedService() const {
    return service_;
}

void WorkerInformation::Expiry(majordomo::ms_time_t expiry) {
    expiry_ = expiry;
}

void WorkerInformation::AssignService(std::shared_ptr<ServiceInformation> service) {
    service_ = std::move(service);
}
