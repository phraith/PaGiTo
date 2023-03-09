namespace ParallelGisaxsToolkit.Gisaxs.Configuration
{
    public class ConnectionStrings
    {
        public string Default { get; set; } = string.Empty;
        public string Redis { get; set; } = string.Empty;
        public string GisaxsBackend { get; set; } = string.Empty;
        public string RabbitMqHost { get; set; } = string.Empty;
        public string RabbitMqUser { get; set; } = string.Empty;
        public string RabbitMqPassword { get; set; } = string.Empty;
        public string RabbitMqConsumerQueueName { get; set; } = string.Empty;
    }
}
