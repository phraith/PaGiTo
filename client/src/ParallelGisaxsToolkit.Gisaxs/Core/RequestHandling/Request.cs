using ParallelGisaxsToolkit.Gisaxs.Configuration;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling
{
    public record Request(string ClientId, string JobHash, string JobId,
        RequestInformation RequestInformation, string RawRequest,
        byte[] ImageDataForFitting);
}