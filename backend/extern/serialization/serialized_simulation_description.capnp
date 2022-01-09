@0xf8f69e4dd75f3ed5;

struct SerializedSimulationDescription
{
	timestamp @0 :UInt32;
	clientId @1 :Text;
	configData @2 :Text;
	instrumentationData @3 :Text;

	isLast @4 :Bool;
}

struct SerializedSimResult
{
	timestamp @0 :UInt32;
	clientId @1 :Text;
	simulatedIntensities @2 :List(Float32);
	simulatedQx @3 :List(Float32);
	simulatedQy @4 :List(Float32);
	simulatedQz @5 :List(Float32);

	xDim @6 :UInt32;
	yDim @7 :UInt32;

	deviceTimingData @8 :List(TimingData);

	struct TimingData
	{
		deviceName @0 :Text;

		kernelTime @1 :Float32;
		simulationTime @2 :Float32;
	}
}