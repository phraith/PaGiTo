using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;

namespace ParallelGisaxsToolkit.GisaxsClient;

public interface IProducer
{
    void Produce(Request request);
}