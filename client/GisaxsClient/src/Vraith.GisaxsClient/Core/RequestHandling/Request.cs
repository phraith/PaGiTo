using Vraith.GisaxsClient.Controllers;

namespace Vraith.GisaxsClient.Core.RequestHandling
{
    public record Request
    {
        public MetaInformation JobInformation { get; }
        public string JobHash { get; }
        public string Body { get; }
        public string InfoHash => $"{JobHash}:info";
        public Request(MetaInformation jobInformation, string jobHash, string body)
        {
            JobInformation = jobInformation;
            JobHash = jobHash;
            Body = body;
        }
    }
}