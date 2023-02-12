//
// Created by phili on 29.01.2023.
//

#include <sys/time.h>
#include "rabbitmq_client.h"
#include "amqp.h"
#include "amqp_tcp_socket.h"
#include "amqp_framing.h"
#include <iostream>
#include <spdlog/spdlog.h>

#define SUMMARY_EVERY_US 1000000

uint64_t now_microseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t) tv.tv_sec * 1000000 + (uint64_t) tv.tv_usec;
}

void RabbitMqClient::Run() {
    uint64_t start_time = now_microseconds();
    int received = 0;
    int previous_received = 0;
    uint64_t previous_report_time = start_time;
    uint64_t next_summary_time = start_time + SUMMARY_EVERY_US;

    amqp_frame_t frame;

    uint64_t now;

    for (;;) {
        amqp_rpc_reply_t ret;
        amqp_envelope_t envelope;

        now = now_microseconds();
        if (now > next_summary_time) {
            int countOverInterval = received - previous_received;
            double intervalRate =
                    countOverInterval / ((now - previous_report_time) / 1000000.0);

            int current_ms = (int) (now - start_time) / 1000;
            spdlog::info("{} ms: Received {} - {} since last report ({} Hz)", current_ms, received, countOverInterval,
                         (int) intervalRate);

            previous_received = received;
            previous_report_time = now;
            next_summary_time += SUMMARY_EVERY_US;
        }

        amqp_maybe_release_buffers(connection_);
        ret = amqp_consume_message(connection_, &envelope, nullptr, 0);

        bool success = EvaluateRpcOperation(ret);
        if (!success) {
            SetUpConnection();
            continue;
        }

        received++;

        std::string message(static_cast<const char *>(envelope.message.body.bytes), envelope.message.body.len);
        auto result = service_->HandleRequest(message, {}, "");

        std::string message2(static_cast<const char *>(envelope.message.properties.reply_to.bytes),
                             envelope.message.properties.reply_to.len);


        amqp_basic_properties_t props;
        props._flags = AMQP_BASIC_CONTENT_TYPE_FLAG |
                        AMQP_BASIC_HEADERS_FLAG |
                       AMQP_BASIC_DELIVERY_MODE_FLAG |
                       AMQP_BASIC_CORRELATION_ID_FLAG;

        props.correlation_id = envelope.message.properties.correlation_id;
        props.content_type = amqp_cstring_bytes("text/plain");
        props.delivery_mode = 2; /* persistent delivery mode */
        props.headers = envelope.message.properties.headers;
        amqp_bytes_t result_bytes{result.data.size(), &result.data[0]};
        int publish_result = amqp_basic_publish(connection_, 2, amqp_cstring_bytes(""),
                                                envelope.message.properties.reply_to, 1, 0, &props,
                                                result_bytes);

        amqp_destroy_envelope(&envelope);

        if (publish_result < 0) {
            spdlog::error("Publishing failed! Resetting connection...");
            SetUpConnection();
        }
    }
}


RabbitMqClient::RabbitMqClient(const std::string &host_name, int port, const std::string &queue_name,
                               std::unique_ptr<Service> service)
        :
        host_name_(host_name),
        queue_name_(queue_name),
        port_(port),
        service_(std::move(service)),
        connection_(nullptr) {

    SetUpConnection();
    Run();
}

void RabbitMqClient::SetUpConnection() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (connection_ != nullptr) {
            int ret_connection = amqp_destroy_connection(connection_);
            if (ret_connection < 0) {
                spdlog::error("Closing connection did not work...");
                continue;
            }
        }

        connection_ = amqp_new_connection();
        auto socket = amqp_tcp_socket_new(connection_);
        if (!socket) {
            spdlog::error("Creating TCP socket failed...");
            continue;
        }

        auto status = amqp_socket_open(socket, host_name_.c_str(), port_);
        if (status) {
            spdlog::error("Opening TCP socket failed...");
            continue;
        }

        auto user = std::getenv("RABBITMQ_USER");
        auto password = std::getenv("RABBITMQ_PASSWORD");

        if (user == nullptr || password == nullptr) {
            spdlog::error("RABBITMQ_USER or RABBITMQ_PASSWORD are not set...");
            continue;
        }

        auto ret = amqp_login(connection_, "/", 0, 131072, 0, AMQP_SASL_METHOD_PLAIN, user, password,
                              std::getenv("RABBITMQ_PASSWORD"));
        bool success = EvaluateRpcOperation(ret);
        if (!success) {
            spdlog::error("Login failed...");
            continue;
        }

        amqp_channel_open(connection_, 1);
        amqp_channel_open(connection_, 2);

        amqp_get_rpc_reply(connection_);

        amqp_bytes_t amq_queue_name{queue_name_.size(), (void *) queue_name_.c_str()};
        amqp_basic_consume(connection_, 1, amq_queue_name, amqp_empty_bytes, 0, 1, 0,
                           amqp_empty_table);

        return;
    }
}

bool RabbitMqClient::EvaluateRpcOperation(amqp_rpc_reply_t reply) {
    if (AMQP_RESPONSE_NORMAL == reply.reply_type) {
        return true;
    }

    if (AMQP_RESPONSE_LIBRARY_EXCEPTION == reply.reply_type &&
        AMQP_STATUS_CONNECTION_CLOSED == reply.library_error) {
        return false;
    }

    if (AMQP_RESPONSE_LIBRARY_EXCEPTION == reply.reply_type) {

        switch (reply.library_error) {
            case AMQP_STATUS_CONNECTION_CLOSED:
                break;
            case AMQP_STATUS_UNEXPECTED_STATE:
                amqp_frame_t frame;
                if (AMQP_STATUS_OK != amqp_simple_wait_frame(connection_, &frame)) {

                }

                if (AMQP_FRAME_METHOD == frame.frame_type) {
                    switch (frame.payload.method.id) {
                        case AMQP_BASIC_ACK_METHOD:
                            /* if we've turned publisher confirms on, and we've published a
                             * message here is a message being confirmed.
                             */
                            break;
                        case AMQP_BASIC_RETURN_METHOD:
                            /* if a published message couldn't be routed and the mandatory
                             * flag was set this is what would be returned. The message then
                             * needs to be read.
                             */
                        {
                            amqp_message_t message;
                            reply = amqp_read_message(connection_, frame.channel, &message, 0);
                            if (AMQP_RESPONSE_NORMAL != reply.reply_type) {
                                break;
                            }

                            amqp_destroy_message(&message);
                        }
                            break;

                        case AMQP_CHANNEL_CLOSE_METHOD:
                            /* a channel.close method happens when a channel exception occurs,
                             * this can happen by publishing to an exchange that doesn't exist
                             * for example.
                             *
                             * In this case you would need to open another channel redeclare
                             * any queues that were declared auto-delete, and restart any
                             * consumers that were attached to the previous channel.
                             */
                            break;

                        case AMQP_CONNECTION_CLOSE_METHOD:
                            /* a connection.close method happens when a connection exception
                             * occurs, this can happen by trying to use a channel that isn't
                             * open for example.
                             *
                             * In this case the whole connection must be restarted.
                             */
                            break;

                        default:
                            spdlog::error("An unexpected method was received {}", frame.payload.method.id);
                            break;
                    }
                }
                break;
        }
    }
    return false;
}