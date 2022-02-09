#ifndef GISAXSMODELINGFRAMEWORK_BROKER_H
#define GISAXSMODELINGFRAMEWORK_BROKER_H

#include <zmq.hpp>
#include <deque>
#include <zmq_addon.hpp>
#include <set>
#include "service_information.h"

namespace majordomo {
    class Broker {
    public:
        explicit Broker(const std::string &ip);

        ~Broker();

        void Start();

        void ProcessInputSocket();

        void ProcessHeartbeat(ms_time_t heartbeat_at);

        zmq::socket_t &Socket();

    private:
        void RemoveDeadWorker();

        std::shared_ptr<ServiceInformation> GetService(const std::string& name);

        void ServiceDispatch(const std::shared_ptr<ServiceInformation>& service);

        void ServiceInternal(const remote_id_t& rid, const std::string& name, zmq::multipart_t &multipart_msg);

        std::shared_ptr<WorkerInformation> WorkerRequire(const remote_id_t& rid);

        void DeleteWorker(const std::shared_ptr<WorkerInformation>& worker, bool disconnect);

        void ProcessRequestFromWorker(const remote_id_t& origin, zmq::multipart_t &multipart_msg);

        void MakeAvailable(const std::shared_ptr<WorkerInformation>& worker);

        void ProcessClient(const remote_id_t& client_id, zmq::multipart_t &multipart_msg);

    private:
        zmq::context_t context_;
        zmq::socket_t socket_;
        const std::string &ip_;
        ms_time_t heartbeat_interval_;
        ms_time_t  heartbeat_expiry_;

        std::unordered_map<remote_id_t, std::shared_ptr<ServiceInformation>> services_;
        std::unordered_map<remote_id_t, std::shared_ptr<WorkerInformation>> workers_;
        std::set<std::shared_ptr<WorkerInformation>> waiting_workers_;
    };
}
#endif