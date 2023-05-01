using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Utility.ImageTransformations;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;

public class RequestResultDeserializer : IRequestResultDeserializer
{
    public IRequestData Deserialize(byte[] bytes, JobType jobType, string? colormap)
    {
        return jobType switch
        {
            JobType.Fitting => DeserializeFitResult(bytes),
            JobType.Simulation => DeserializeSimulationResult(bytes, colormap),
            _ => throw new ArgumentOutOfRangeException(nameof(jobType), jobType, null)
        };
    }

    private FitRequestData DeserializeFitResult(byte[] bytes)
    {
        int currentBytePosition = 0;
        int parameterCount = BitConverter.ToInt32(bytes);
        currentBytePosition += sizeof(int);

        double[] parameters = new double[parameterCount];
        for (int i = 0; i < parameterCount; i++)
        {
            double parameter = BitConverter.ToDouble(bytes, currentBytePosition);
            currentBytePosition += sizeof(double);
            parameters[i] = parameter;
        }

        int fitnessCount = BitConverter.ToInt32(bytes, currentBytePosition);
        currentBytePosition += sizeof(int);
        double[] fitnessValues = new double[fitnessCount];
        for (int i = 0; i < fitnessCount; i++)
        {
            double fitness = BitConverter.ToDouble(bytes, currentBytePosition);
            currentBytePosition += sizeof(double);
            fitnessValues[i] = fitness;
        }

        return new FitRequestData(fitnessValues, parameters);
    }

    private SimulationRequestData DeserializeSimulationResult(byte[] bytes, string? colormap)
    {
        List<JpegResult>? jpegResults = new();
        List<NumericResult>? numericResults = new();

        int currentBytePosition = 0;
        while (currentBytePosition < bytes.Length)
        {
            int startX = BitConverter.ToInt32(bytes);
            currentBytePosition += sizeof(int);

            int startY = BitConverter.ToInt32(bytes, currentBytePosition);
            currentBytePosition += sizeof(int);

            int endX = BitConverter.ToInt32(bytes, currentBytePosition);
            currentBytePosition += sizeof(int);

            int endY = BitConverter.ToInt32(bytes, currentBytePosition);
            currentBytePosition += sizeof(int);

            int width = BitConverter.ToInt32(bytes, currentBytePosition);
            currentBytePosition += sizeof(int);

            int height = BitConverter.ToInt32(bytes, currentBytePosition);
            currentBytePosition += sizeof(int);

            int valueByteCount = BitConverter.ToInt32(bytes, currentBytePosition);
            currentBytePosition += sizeof(int);

            int expectedByteCount = width * height * valueByteCount;

            byte[] data = bytes[currentBytePosition..(currentBytePosition + expectedByteCount)];
            switch (valueByteCount)
            {
                case sizeof(byte) when colormap == null:
                    throw new InvalidOperationException("Colormap is not set!");
                case sizeof(byte):
                {
                    string modifiedData = AppearanceModifier.ApplyColorMap(data, width, height, true, colormap);
                    JpegResult jpegResult = new JpegResult(modifiedData, width, height);
                    jpegResults.Add(jpegResult);
                    break;
                }
                case sizeof(double):
                {
                    double[] samples = new double[width * height];
                    Buffer.BlockCopy(data, 0, samples, 0, data.Length);
                    NumericResult numericResult =
                        new NumericResult(samples.Select(x => Math.Log(x + 1)).ToArray(), width, height);
                    numericResults.Add(numericResult);
                    break;
                }
                default:
                    throw new InvalidOperationException("Data format is not supported!");
            }

            currentBytePosition += data.Length;
        }

        return new SimulationRequestData(jpegResults, numericResults);
    }
}