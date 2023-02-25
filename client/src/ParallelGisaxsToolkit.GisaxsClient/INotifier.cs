namespace ParallelGisaxsToolkit.GisaxsClient;

public interface INotifier
{
    Task Notify(string target, string notificationType, string notification);
}