using System.Text;
using System.Text.Json;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
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
    private readonly IRabbitMqFactory _rabbitMqFactory;

    public RabbitMqPublisher(INotifier notifier, IDatabase redisClient,
        IResultStore resultStore, IRequestResultDeserializer requestResultDeserializer,
        IRabbitMqFactory rabbitMqFactory)
    {
        _notifier = notifier;
        _redisClient = redisClient;
        _resultStore = resultStore;
        _requestResultDeserializer = requestResultDeserializer;
        _rabbitMqFactory = rabbitMqFactory;
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

        var publisherQueueName = $"{request.RequestInformation.MetaInformation.Type}";

        using IModel publisherChannel = _rabbitMqFactory.CreatePublisherModel(publisherQueueName);
        IBasicProperties properties = publisherChannel.CreateBasicProperties();
        properties.ReplyTo = _rabbitMqFactory.ConsumerQueueName;
        properties.CorrelationId = request.JobId;
        properties.Headers = new Dictionary<string, object>();
        properties.Headers["notification"] = request.RequestInformation.MetaInformation.Notification;
        properties.Headers["clientId"] = request.ClientId;
        properties.Headers["jobHash"] = request.JobHash;

        if (colormap != null)
        {
            properties.Headers["colormap"] = colormap;
        }
        
        byte[] message = RequestToBytes(request);

        publisherChannel.BasicPublish(exchange: "",
            routingKey: $"{request.RequestInformation.MetaInformation.Type}",
            basicProperties: properties,
            body: message);
    }

    private static byte[] RequestToBytes(Request request)
    {
        byte[] message = Encoding.UTF8.GetBytes(request.RawRequest);
        byte[] messageSize = BitConverter.GetBytes(message.Length);
        byte[] imageSize = BitConverter.GetBytes(request.ImageDataForFitting.Length);
        return messageSize.Concat(message).Concat(imageSize).Concat(request.ImageDataForFitting).ToArray();
    }
}