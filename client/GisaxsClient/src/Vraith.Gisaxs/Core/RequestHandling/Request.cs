using Vraith.Gisaxs.Configuration;

namespace Vraith.Gisaxs.Core.RequestHandling
{
    public record Request(RequestInformation RequestInformation, string DataAccessor, string JobHash, string RawRequest,
        byte[] ImageDataForFitting)
    {
        public string InfoHash => $"{JobHash}:info";
    }
}