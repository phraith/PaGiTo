using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Utility.ImageTransformations;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;

public class RequestResultDeserializer : IRequestResultDeserializer
{
    public RequestData Deserialize(byte[] bytes, string? colormap)
    {
        if (bytes.Length == sizeof(float))
        {
            float fitness = BitConverter.ToSingle(bytes.ToArray());
            return new RequestData(Array.Empty<JpegResult>(), Array.Empty<NumericResult>(), fitness);
        }

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
            if (valueByteCount == sizeof(byte))
            {
                if (colormap == null)
                {
                    throw new InvalidOperationException("Colormap is not set!");
                }

                string modifiedData = AppearanceModifier.ApplyColorMap(data, width, height, true, colormap);
                JpegResult jpegResult = new JpegResult(modifiedData, width, height);
                jpegResults.Add(jpegResult);
            }
            else if (valueByteCount == sizeof(double))
            {
                double[] samples = new double[width * height];
                Buffer.BlockCopy(data, 0, samples, 0, data.Length);
                NumericResult numericResult = new NumericResult(samples.Select(x => Math.Log(x + 1)).ToArray(), width, height);
                numericResults.Add(numericResult);
            }
            else
            {
                throw new InvalidOperationException("Data format is not supported!");
            }

            currentBytePosition += data.Length;
        }

        return new RequestData(jpegResults, numericResults, null);
    }
}