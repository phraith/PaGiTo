namespace GisaxsClient.Controllers
{
    public record MetaInformation
    {
        public long ClientId { get; }
        public string JobType { get; }
        public long JobId { get; }
        public string ColormapName { get; }
        public MetaInformation(long clientId, string jobType, long jobId, string colormapName)
        {
            ClientId = clientId;
            JobType = jobType;
            JobId = jobId;
            ColormapName = colormapName;
        }
    }
}
