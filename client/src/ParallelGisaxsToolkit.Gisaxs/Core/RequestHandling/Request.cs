using ParallelGisaxsToolkit.Gisaxs.Configuration;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling
{
    public record Request(RequestInformation RequestInformation, string DataAccessor, string JobHash, string RawRequest)
    {
        public string InfoHash => $"{JobHash}:info";
    }
}