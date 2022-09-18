namespace GisaxsClient.Configuration
{
    public enum LaunchMode { Kubernetes, Docker }
    public static class LaunchConfig
    {
        public static LaunchMode LaunchMode  { get; }
        static LaunchConfig()
        {
            string? environmentContent = Environment.GetEnvironmentVariable("DEPLOY_MODE");
            LaunchMode = environmentContent == null ? LaunchMode.Docker : Enum.Parse<LaunchMode>(environmentContent);
        }
    }
}
