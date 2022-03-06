#ifndef MODEL_SIMULATOR_TESTING_TESTCLIENT_H
#define MODEL_SIMULATOR_TESTING_TESTCLIENT_H

#include <string>

class TestClient {
public:
	TestClient(std::string ip);
	~TestClient();

	void Send(std::string message);

private:
	std::string ip_;
	
};

#endif