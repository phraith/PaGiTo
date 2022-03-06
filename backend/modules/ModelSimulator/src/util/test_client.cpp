#include "../inc/util/test_client.h"

#include <iostream>
#include <zmq.hpp>


TestClient::TestClient(std::string ip)
	:
	ip_(ip)
{
}

TestClient::~TestClient()
{
}

void TestClient::Send(std::string message)
{
	zmq::context_t context;
	zmq::socket_t test_socket(context, zmq::socket_type::req);
	test_socket.connect("tcp://" + ip_);
	test_socket.send(zmq::buffer(message));

	zmq::message_t result;
	test_socket.recv(result);

	std::cout << result << std::endl;

	test_socket.close();
}
