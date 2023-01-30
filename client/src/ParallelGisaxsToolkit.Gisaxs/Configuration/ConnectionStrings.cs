namespace ParallelGisaxsToolkit.Gisaxs.Configuration
{
    public class ConnectionStrings
    {
        public string Default { get; set; } = string.Empty;
        public string Redis { get; set; } = string.Empty;
        public string GisaxsBackend { get; set; } = string.Empty;
        public string RabbitMq { get; set; } = string.Empty;
    }
}
