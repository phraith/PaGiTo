//
// Created by Phil on 16.01.2022.
//

#include "broker.h"

#include <utility>
#include <iostream>
#include <spdlog/spdlog.h>

void majordomo::Broker::Start() {
    //ms_time_t now = ms_now();
    //ms_time_t heartbeat_at = now + heartbeat_interval_;

    zmq::poller_t poller;
    poller.add(socket_, zmq::event_flags::pollin);
}

majordomo::Broker::Broker(const std::string &ip)
        :
        ip_(ip),
        socket_(context_, zmq::socket_type::router),
        heartbeat_expiry_(HEARTBEAT_EXPIRY),
        heartbeat_interval_(HEARTBEAT_INTERVAL)
        {
    socket_.bind(ip);
}

void majordomo::Broker::ProcessInputSocket() {
    zmq::multipart_t multipart_msg;
    remote_id_t origin;
    ReceiveFromFromRouter(socket_, multipart_msg, origin);
    std::string header = multipart_msg.popstr();
    if (header == majordomo::client::ident) {
        ProcessClient(origin, multipart_msg);
    } else if (header == majordomo::worker::ident) {
        ProcessRequestFromWorker(origin, multipart_msg);
    }
}

void majordomo::Broker::ProcessHeartbeat(majordomo::ms_time_t heartbeat_at) {
    ms_time_t now = ms_now();
    if (now < heartbeat_at) { return; }
    RemoveDeadWorker();
    for (auto &worker: waiting_workers_) {
        zmq::multipart_t multipart_msg;
        multipart_msg.pushstr(majordomo::worker::heartbeat);
        multipart_msg.pushstr(majordomo::worker::ident);
        SendToRouter(socket_, multipart_msg, worker->Identity());
    }
}

void majordomo::Broker::RemoveDeadWorker() {
    ms_time_t now = ms_now();
    std::vector<std::shared_ptr<WorkerInformation>> dead;
    for (auto &worker: waiting_workers_) {
        std::cout << "updating worker expiry from " << worker->Expiry().count() << " to " << now.count() << " | diff: " << worker->Expiry().count() - now.count() <<std::endl;
        if (worker->Expiry() <= now) {
            std::cout << "Kill worker" << std::endl;
            dead.push_back(worker);
        }
    }
    for (auto &worker: dead) {
        DeleteWorker(worker, false);
    }
}

std::shared_ptr<ServiceInformation> majordomo::Broker::GetService(const std::string &name) {
    std::shared_ptr<ServiceInformation> service = services_[name];
    if (service == nullptr) {
        service = std::make_shared<ServiceInformation>(name);
        services_[name] = service;
    }
    return service;
}

void majordomo::Broker::ServiceDispatch(const std::shared_ptr<ServiceInformation> &service) {
    //RemoveDeadWorker();
    while (!service->WaitingWorkers().empty() and !service->Requests().empty()) {
        auto worker_it = service->WaitingWorkers().begin();
        auto next = worker_it;
        for (++next; next != service->WaitingWorkers().end(); ++next) {
            if ((*next)->Expiry() > (*worker_it)->Expiry()) {
                worker_it = next;
            }
        }

        zmq::multipart_t &multipart_msg = service->Requests().front();
        SendToRouter(socket_, multipart_msg, (*worker_it)->Identity());
        service->Requests().pop_front();
        std::shared_ptr<WorkerInformation> w = *worker_it;
        waiting_workers_.erase(*worker_it);
        service->WaitingWorkers().erase(worker_it);
    }
}

void majordomo::Broker::ServiceInternal(const majordomo::remote_id_t &rid, const std::string &name,
                                        zmq::multipart_t &multipart_msg) {
    zmq::multipart_t response;
    if (name == "mmi.Service") {
        std::string sn = multipart_msg.popstr();
        std::shared_ptr<ServiceInformation> service = services_[sn];
        if (service != nullptr and service->WorkerCount() > 0) {
            response.pushstr("200");
        } else {
            response.pushstr("404");
        }
    } else {
        response.pushstr("501");
    }
    SendToRouter(socket_, response, rid);
}

void majordomo::Broker::DeleteWorker(const std::shared_ptr<WorkerInformation> &worker, bool disconnect) {
    if (disconnect) {
        zmq::multipart_t multipart_msg;
        multipart_msg.pushstr(majordomo::worker::disconnect);
        multipart_msg.pushstr(majordomo::worker::ident);
        SendToRouter(socket_, multipart_msg, worker->Identity());
    }

    if (worker->HasService()) {
        worker->AssignedService()->DeleteWorker(worker);
    }
    waiting_workers_.erase(worker);
    workers_.erase(worker->Identity());
}

void
majordomo::Broker::ProcessRequestFromWorker(const majordomo::remote_id_t &origin, zmq::multipart_t &multipart_msg) {

//    if (multipart_msg.size() == 1)
//    {
//        const std::string command = multipart_msg.popstr();
//        assert(multipart_msg.size() > 1);
//    }

    const std::string command = multipart_msg.popstr();
    bool worker_ready = (workers_.find(origin) != workers_.end());
    std::shared_ptr<WorkerInformation> worker = WorkerRequire(origin);

    if (majordomo::worker::info == command) {
        remote_id_t client_id = multipart_msg.popstr();
        multipart_msg.pop();
        multipart_msg.pushstr("fitting");
        multipart_msg.pushstr(majordomo::client::ident);

         zmq::send_result_t res = SendToRouter(socket_, multipart_msg, client_id);
    }

    if (majordomo::worker::ready == command) {
        if (worker_ready) {
            DeleteWorker(worker, true);
            return;
        }

        std::string service_name = multipart_msg.popstr();

        worker->AssignService(GetService(service_name));
        MakeAvailable(worker);
        return;
    }

    if (majordomo::worker::reply == command) {
        if (!worker_ready) {
            DeleteWorker(worker, false);
            return;
        }

        remote_id_t client_id = multipart_msg.popstr();
        multipart_msg.pop();
        multipart_msg.pushstr(worker->AssignedService()->Name());
        multipart_msg.pushstr(majordomo::client::ident);

        zmq::send_result_t res = SendToRouter(socket_, multipart_msg, client_id);
        MakeAvailable(worker);
    }

    if (majordomo::worker::heartbeat == command) {
        if (!worker_ready) {
            DeleteWorker(worker, false);
            return;
        }
        std::cout << "Heartbeat: " << majordomo::ms_now().count() << " " << heartbeat_expiry_.count() << std::endl;
        worker->Expiry(majordomo::ms_now() + heartbeat_expiry_);
        return;
    }

    if (majordomo::worker::disconnect == command) {
        DeleteWorker(worker, true);
        return;
    }
}

void majordomo::Broker::MakeAvailable(const std::shared_ptr<WorkerInformation> &worker) {
    waiting_workers_.insert(worker);
    worker->AssignedService()->WaitingWorkers().push_back(worker);
    spdlog::info("Worker {} with identity {} is now available!", worker->AssignedService()->Name(), worker->Identity());
    worker->Expiry(majordomo::ms_now() + heartbeat_expiry_);
    ServiceDispatch(worker->AssignedService());
}

void majordomo::Broker::ProcessClient(const majordomo::remote_id_t &client_id, zmq::multipart_t &multipart_msg) {
    std::string service_name = multipart_msg.popstr();
    std::shared_ptr<ServiceInformation> service = GetService(service_name);
    if (service_name.size() >= 4 and service_name.find_first_of("mmi.") == 0) {
        ServiceInternal(client_id, service_name, multipart_msg);
        return;
    }

    multipart_msg.pushmem(nullptr, 0);
    multipart_msg.pushstr(client_id);
    multipart_msg.pushstr(majordomo::worker::request);
    multipart_msg.pushstr(majordomo::worker::ident);
    service->Requests().emplace_back(std::move(multipart_msg));
    ServiceDispatch(service);
}

std::shared_ptr<WorkerInformation>
majordomo::Broker::WorkerRequire(const majordomo::remote_id_t &rid) {
    std::shared_ptr<WorkerInformation> worker = workers_[rid];
    if (worker == nullptr) {
        worker = std::make_shared<WorkerInformation>(nullptr, rid, majordomo::ms_time_t::zero());
        workers_[rid] = worker;
    }
    return worker;
}

zmq::socket_t &majordomo::Broker::Socket() {
    return socket_;
}

majordomo::Broker::~Broker() = default;