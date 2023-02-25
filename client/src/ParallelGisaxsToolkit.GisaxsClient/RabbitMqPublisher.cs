using System.Text;
using System.Text.Json;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.Gisaxs.Core.ResultStore;
using RabbitMQ.Client;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient;

public interface IRabbitMqPublisher
{
    Task Publish(Request request);
}

public class RabbitMqPublisher : IRabbitMqPublisher
{
    private readonly INotifier _notifier;
    private readonly IDatabase _redisClient;
    private readonly IResultStore _resultStore;
    private readonly IRequestResultDeserializer _requestResultDeserializer;
    private readonly IRabbitMqService _rabbitMqService;

    public RabbitMqPublisher(INotifier notifier, IDatabase redisClient,
        IResultStore resultStore, IRequestResultDeserializer requestResultDeserializer, IRabbitMqService rabbitMqService)
    {
        _notifier = notifier;
        _redisClient = redisClient;
        _resultStore = resultStore;
        _requestResultDeserializer = requestResultDeserializer;
        _rabbitMqService = rabbitMqService;
    }

    public async Task Publish(Request request)
    {
        byte[]? cachedResponse = _redisClient.StringGet(request.JobHash);
        string? colormap = request.RequestInformation.MetaInformation.Colormap;
        if (cachedResponse is { Length: > 0 })
        {
            RequestData resultData = _requestResultDeserializer.Deserialize(cachedResponse, colormap);
            string serializedResult = JsonSerializer.Serialize(
                resultData, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });

            await _resultStore.Insert(new Result(serializedResult, request.JobId, request.ClientId));
            await _notifier.Notify(request!.ClientId, request.RequestInformation.MetaInformation.Notification,
                request.JobId);
            return;
        }

        IBasicProperties properties = _rabbitMqService.PublisherChannel.CreateBasicProperties();
        properties.ReplyTo = _rabbitMqService.ConsumerQueueName;
        properties.CorrelationId = request.JobId;
        properties.Headers = new Dictionary<string, object>();
        properties.Headers["notification"] = request.RequestInformation.MetaInformation.Notification;
        properties.Headers["clientId"] = request.ClientId;
        properties.Headers["jobHash"] = request.JobHash;

        if (colormap != null)
        {
            properties.Headers["colormap"] = colormap;
        }

        byte[] message = Encoding.UTF8.GetBytes(request.RawRequest);

        _rabbitMqService.PublisherChannel.BasicPublish(exchange: "",
            routingKey: $"{request.RequestInformation.MetaInformation.Type}",
            basicProperties: properties,
            body: message);
    }


}