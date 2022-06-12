namespace GisaxsClient.Controllers
{
    public record MetaInformation
    {
        public long ClientId { get; init; }
        public string JobType { get; init; }
        public long JobId { get; init; }
        public string ColormapName { get; init; }
    }
}
