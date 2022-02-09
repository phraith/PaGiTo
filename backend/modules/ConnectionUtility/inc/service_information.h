//
// Created by Phil on 23.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_SERVICE_INFORMATION_H
#define GISAXSMODELINGFRAMEWORK_SERVICE_INFORMATION_H


#include <string>
#include <deque>
#include <vector>
#include <zmq_addon.hpp>
#include "worker_information.h"

class ServiceInformation {
public:
    explicit ServiceInformation(std::string name);

private:
    std::string name_;
    std::deque<zmq::multipart_t> requests_;
    std::vector<std::shared_ptr<WorkerInformation>> waiting_workers_;
public:
    [[nodiscard]] const std::string &Name() const;
    void DeleteWorker(const std::shared_ptr<WorkerInformation> &worker);
    [[nodiscard]] size_t WorkerCount() const;
    [[nodiscard]] std::vector<std::shared_ptr<WorkerInformation>> &WaitingWorkers();
    [[nodiscard]] std::deque<zmq::multipart_t> &Requests();

private:
    size_t worker_count_;
};


#endif //GISAXSMODELINGFRAMEWORK_SERVICE_INFORMATION_H
