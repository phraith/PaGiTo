//
// Created by Phil on 05.02.2022.
//

#include "worker.h"
#include "worker_request_handler.h"

using majordomo::ms_time_t;
using majordomo::ms_now;
using majordomo::HEARTBEAT_INTERVAL;
using majordomo::HEARTBEAT_INTERVAL;


void Worker::Start() {
    WorkerRequestHandler worker(worker_address_, broker_address_, service_->ServiceName());
    ms_time_t now = ms_now();
    ms_time_t heartbeat_interval = HEARTBEAT_INTERVAL;
    ms_time_t heartbeat_at = now + heartbeat_interval;

    zmq::poller_t poller;
    poller.add(worker.Socket(), zmq::event_flags::pollin);

    bool interrupted = false;
    zmq::multipart_t reply;
    while (!interrupted) {
        std::vector<zmq::poller_event<>> events(2);
        int events_count = poller.wait_all(events, ms_time_t{500});
        for (int i = 0; i < events_count; ++i) {
            if (events[i].socket == worker.Socket()) {
                zmq::multipart_t request;
                worker.Receive(request);
                if (request.empty()) {
                    break;
                }

                auto payload = request.popstr();

                std::vector<std::byte> image_data;

                if (request.size() == 1) {
                    auto image_bytes = request.pop();

                    if (!image_bytes.empty()) {
                        image_data = std::vector<std::byte>(image_bytes.size());
                        std::copy(reinterpret_cast<const std::byte *>(image_bytes.data()),
                                  reinterpret_cast<const std::byte *>(image_bytes.data()) + image_bytes.size(),
                                  &image_data[0]);
                    }
                }

                auto result = service_->HandleRequest(payload, image_data, worker.ReplyTo());
                request.pushmem(&result[0], result.size());
                reply = std::move(request);

                worker.Send(reply);
            }
        }
    }
}

Worker::Worker(std::unique_ptr<Service> &service, const std::string &worker_address, const std::string &broker_address)
        :
        service_(std::move(service)),
        worker_address_(worker_address),
        broker_address_(broker_address) {

}