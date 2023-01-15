using ParallelGisaxsToolkit.Gisaxs.Configuration;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;

public interface IRequestResultDeserializer
{
    RequestData Deserialize(byte[] bytes, string? colormap);
}

public record RequestData(IReadOnlyList<JpegResult> JpegResults, IReadOnlyList<NumericResult> NumericResults,
    double? Fitness);