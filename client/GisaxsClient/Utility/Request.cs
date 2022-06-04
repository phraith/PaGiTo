using GisaxsClient.Controllers;

namespace GisaxsClient.Utility
{
    public record Request
    {
        public MetaInformation JobInformation { get; init; }
        public string JobHash { get; init; }
        public string InfoHash => $"{JobHash}:info";
        public string Body { get; init; }
    }
}