using System.Text.Json;
using ParallelGisaxsToolkit.Gisaxs.Configuration;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;

public interface IRequestResultDeserializer
{
    IRequestData Deserialize(byte[] bytes, JobType jobType, string? colormap);
}

public interface IRequestData
{
    string Serialize();
}

public record SimulationRequestData
    (IReadOnlyList<JpegResult> JpegResults, IReadOnlyList<NumericResult> NumericResults) : IRequestData
{
    public string Serialize()
    {
        return JsonSerializer.Serialize(
            this, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
    }
}

public record FitRequestData(IReadOnlyList<double> Fitness, IReadOnlyList<double> Parameters) : IRequestData
{
    public string Serialize()
    {
        return JsonSerializer.Serialize(
            this, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
    }
}