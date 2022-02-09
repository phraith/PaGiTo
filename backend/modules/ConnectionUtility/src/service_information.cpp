//
// Created by Phil on 23.01.2022.
//

#include "service_information.h"
#include <utility>

ServiceInformation::ServiceInformation(std::string name)
        :
        name_(std::move(name)),
        worker_count_(0) {

}

void ServiceInformation::DeleteWorker(const std::shared_ptr<WorkerInformation> &worker) {
    for (auto it = waiting_workers_.begin(); it != waiting_workers_.end();) {
        if (worker.get() == (*it).get()) {
            it = waiting_workers_.erase(it);
            --worker_count_;
        } else {
            ++it;
        }
    }
}

size_t ServiceInformation::WorkerCount() const {
    return worker_count_;
}

std::vector<std::shared_ptr<WorkerInformation>> &ServiceInformation::WaitingWorkers() {
    return waiting_workers_;
}

std::deque<zmq::multipart_t> &ServiceInformation::Requests() {
    return requests_;
}

const std::string &ServiceInformation::Name() const {
    return name_;
}


