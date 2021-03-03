@0xf8f69e4dd75f3ed3;

struct SerializedFittingDescription
{
	timestamp @0 :UInt32;
	clientId @1 :Text;
	configData @2 :Text;
	instrumentationData @3 :Text;

	intensities @4 :List(Float32);
	offsets @5 :List(UInt32);

	isLast @6 :Bool;
}

struct SerializedFittingResult
{
	timestamp @0 :UInt32;
	clientId @1 :Text;
	fittedShapes@2 :List(FittedShape);
	simulatedIntensities @3 :List(Float32);
	simulatedQx @4 :List(Float32);
	simulatedQy @5 :List(Float32);
	simulatedQz @6 :List(Float32);
	fitness @7 :Float32;
	scale @8 :Float32;

	deviceTimingData @9 :List(TimingData);

	struct FittedParameter
	{
		type @0 :Text;
		value @1 :Float32;
		stddev @2 :Float32;
	}
	
	struct FittedShape
	{
		type @0 :Text;
		parameters @1 :List(FittedParameter);
	}
	
	struct TimingData
	{
		deviceName @0 :Text;

		simRuns @1 :UInt32;
		kernelTime @2 :Float32;
		averageKernelTime @3 :Float32;

		simulationTime @4 :Float32;
		averageSimulationTime @5 :Float32;

		fittingTime @6 :Float32;
	}
}