//
// Created by phili on 19.02.2023.
//

#include <thread>
#include <spdlog/spdlog.h>
#include <amqp_tcp_socket.h>
#include "rabbitmq_connection.h"
#include <thread>

using namespace std::chrono_literals;

RabbitMqConnection::RabbitMqConnection(std::string host_name, int port, std::string user_name, std::string password)
        :
        host_name_(host_name),
        port_(port),
        user_name_(user_name),
        password_(password),
        connection_(nullptr),
        next_channel_id_(1) {
    ConnectSafe();
}

void RabbitMqConnection::Establish() {
    if (connection_ != nullptr) {
        for (int i = 1; i < next_channel_id_; ++i) {
            if (!EvaluateRpcOperation(amqp_channel_close(connection_, i, AMQP_REPLY_SUCCESS))) {
                spdlog::error("Closing already open channel did not work...");
//                throw RabbitMqConnectionException();
            }
        }
        next_channel_id_ = 1;
        int ret_connection = amqp_destroy_connection(connection_);
        if (ret_connection < 0) {
            spdlog::error("Closing connection did not work...");
            throw RabbitMqConnectionException();
        }
    }

    connection_ = amqp_new_connection();
    auto socket = amqp_tcp_socket_new(connection_);
    if (!socket) {
        spdlog::error("Creating TCP socket failed...");
        throw RabbitMqConnectionException();
    }

    auto status = amqp_socket_open(socket, host_name_.c_str(), port_);
    if (status) {
        spdlog::error("Opening TCP socket failed...");
        throw RabbitMqConnectionException();
    }

    auto user = std::getenv("RABBITMQ_USER");
    auto password = std::getenv("RABBITMQ_PASSWORD");

    if (user == nullptr || password == nullptr) {
        spdlog::error("RABBITMQ_USER or RABBITMQ_PASSWORD are not set...");
        throw RabbitMqConnectionException();
    }

    auto ret = amqp_login(connection_, "/", 0, 131072, 0, AMQP_SASL_METHOD_PLAIN, user, password,
                          std::getenv("RABBITMQ_PASSWORD"));
    bool success = EvaluateRpcOperation(ret);
    if (!success) {
        spdlog::error("Login failed...");
        throw RabbitMqConnectionException();
    }
}

bool RabbitMqConnection::EvaluateRpcOperation(amqp_rpc_reply_t reply) {
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

uint16_t RabbitMqConnection::QueueDeclare(std::string queueName) {

    uint16_t channel_id = NextChannel();
    amqp_bytes_t amq_queue_name{queueName.size(), (void *) queueName.c_str()};
    auto ok = amqp_queue_declare(connection_, channel_id, amq_queue_name, false, true, false, false, amqp_empty_table);

    if (!EvaluateRpcOperation(amqp_get_rpc_reply(connection_))) {
        spdlog::error("Declaring queue failed!");
        throw RabbitMqConnectionException();
    }

    return channel_id;
}

void RabbitMqConnection::RegisterConsumer(std::string queue_name) {

    uint16_t channel_id = QueueDeclare(queue_name);
    amqp_bytes_t amq_queue_name{queue_name.size(), (void *) queue_name.c_str()};
    auto ok = amqp_basic_consume(connection_, channel_id, amq_queue_name, amqp_empty_bytes, 0, 1, 0,
                                 amqp_empty_table);

    if (!EvaluateRpcOperation(amqp_get_rpc_reply(connection_))) {
        spdlog::error("Registering consumer failed!");
        throw RabbitMqConnectionException();
    }
}

void RabbitMqConnection::RegisterConsumer(std::string queue_name, uint16_t channel_id) {

    amqp_bytes_t amq_queue_name{queue_name.size(), (void *) queue_name.c_str()};
    auto ok = amqp_basic_consume(connection_, channel_id, amq_queue_name, amqp_empty_bytes, 0, 1, 0,
                                 amqp_empty_table);

    if (!EvaluateRpcOperation(amqp_get_rpc_reply(connection_))) {
        spdlog::error("Registering consumer failed!");
        throw RabbitMqConnectionException();
    }
}

std::tuple<std::string, std::string> RabbitMqConnection::Consume() {
    amqp_envelope_t envelope;
    amqp_maybe_release_buffers(connection_);
    if (!EvaluateRpcOperation(amqp_consume_message(connection_, &envelope, nullptr, 0))) {
        spdlog::error("Consuming message failed!");
        throw RabbitMqConnectionException();
    }

    std::string message(static_cast<const char *>(envelope.message.body.bytes), envelope.message.body.len);
    std::string correlation_id(static_cast<const char *>(envelope.message.properties.correlation_id.bytes),
                               envelope.message.properties.correlation_id.len);

    amqp_destroy_envelope(&envelope);
    return {message, correlation_id};
}

void RabbitMqConnection::ConsumeAndPublish(int publisher_channel,
                                           std::function<std::vector<std::byte>(std::string message)> handler) {
    amqp_envelope_t envelope;
    amqp_maybe_release_buffers(connection_);
    if (!EvaluateRpcOperation(amqp_consume_message(connection_, &envelope, nullptr, 0))) {
        spdlog::error("Consuming message failed!");
        throw RabbitMqConnectionException();
    }

    std::string message(static_cast<const char *>(envelope.message.body.bytes), envelope.message.body.len);
    auto result = handler(message);

    amqp_basic_properties_t props;
    props._flags = AMQP_BASIC_CONTENT_TYPE_FLAG |
                   AMQP_BASIC_HEADERS_FLAG |
                   AMQP_BASIC_DELIVERY_MODE_FLAG |
                   AMQP_BASIC_CORRELATION_ID_FLAG;

    props.correlation_id = envelope.message.properties.correlation_id;
    props.content_type = amqp_cstring_bytes("text/plain");
    props.delivery_mode = 2; /* persistent delivery mode */
    props.headers = envelope.message.properties.headers;
    amqp_bytes_t result_bytes{result.size(), &result[0]};

    std::string reply_to(static_cast<const char *>(envelope.message.properties.reply_to.bytes),
                         envelope.message.properties.reply_to.len);
    spdlog::info("Replying to {}", reply_to);
    int publish_result = amqp_basic_publish(connection_, publisher_channel, amqp_cstring_bytes(""),
                                            envelope.message.properties.reply_to, 1, 0, &props, result_bytes);
    amqp_destroy_envelope(&envelope);

    if (publish_result < 0) {
        spdlog::error("Publishing failed!");
        throw RabbitMqConnectionException();
    }
}

std::tuple<std::string, uint16_t> RabbitMqConnection::QueueDeclare() {
    uint16_t channel_id = NextChannel();
    auto ok = amqp_queue_declare(connection_, channel_id, amqp_empty_bytes, false, true, false, false,
                                 amqp_empty_table);
    if (!EvaluateRpcOperation(amqp_get_rpc_reply(connection_))) {
        spdlog::error("Declaring queue failed!");
        throw RabbitMqConnectionException();
    }
    std::string queue_name(static_cast<const char *>(ok->queue.bytes), ok->queue.len);
    return {queue_name, channel_id};
}

uint16_t RabbitMqConnection::NextChannel() {
    uint16_t channel_id = next_channel_id_;

    auto channel_ok = amqp_channel_open(connection_, channel_id);

    if (!EvaluateRpcOperation(amqp_get_rpc_reply(connection_))) {
        spdlog::error("Channel creation failed!");
        throw RabbitMqConnectionException();
    }
    ++next_channel_id_;
    return channel_id;
}

void RabbitMqConnection::ConnectSafe() {
    while (true) {
        std::this_thread::sleep_for(500ms);
        try {
            Establish();
            return;
        }
        catch (RabbitMqConnectionException &e) {
            spdlog::error(e.what());
        }
    }
}

void RabbitMqConnection::Publish(int publisher_channel, std::string target_queue_name, std::string reply_to,
                                 std::string correlation_id, std::vector<std::byte> &bytes) {
    amqp_basic_properties_t props;
    props._flags = AMQP_BASIC_CONTENT_TYPE_FLAG |
                   AMQP_BASIC_REPLY_TO_FLAG |
                   AMQP_BASIC_CORRELATION_ID_FLAG |
                   AMQP_BASIC_DELIVERY_MODE_FLAG;
    props.content_type = amqp_cstring_bytes("text/plain");
    props.delivery_mode = 2; /* persistent delivery mode */
    amqp_bytes_t correlation_id_bytes{correlation_id.size(), &correlation_id[0]};
    props.correlation_id = correlation_id_bytes;
    amqp_bytes_t reply_to_bytes{reply_to.size(), &reply_to[0]};

    props.reply_to = reply_to_bytes;
    amqp_bytes_t result_bytes{bytes.size(), &bytes[0]};
    amqp_bytes_t routing_key_bytes{target_queue_name.size(), &target_queue_name[0]};

    int publish_result = amqp_basic_publish(connection_, publisher_channel, amqp_cstring_bytes(""), routing_key_bytes,
                                            1, 0, &props,
                                            result_bytes);

    if (publish_result < 0) {
        spdlog::error("Publishing failed!");
        throw RabbitMqConnectionException();
    }
}
