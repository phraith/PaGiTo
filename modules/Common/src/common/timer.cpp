#include "common/timer.h"

Timer::Timer()
{
}

void Timer::Start()
{
	start_ = std::chrono::steady_clock::now();
}

void Timer::End()
{
	end_ = std::chrono::steady_clock::now();
}

double Timer::Duration()
{
	std::chrono::duration<double> t = end_ - start_;
	return t.count();
}