using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.SignalR;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.Hubs;
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
    private readonly IHubContext<MessageHub> _notificationHub;
    private readonly IDatabase _redisClient;
    private readonly IResultStore _resultStore;
    private readonly IRequestResultDeserializer _requestResultDeserializer;
    private readonly IModel _producerChannel;

    public RabbitMqPublisher(IConnection connection, IHubContext<MessageHub> notificationHub, IDatabase redisClient,
        IResultStore resultStore, IRequestResultDeserializer requestResultDeserializer)
    {
        _notificationHub = notificationHub;
        _redisClient = redisClient;
        _resultStore = resultStore;
        _requestResultDeserializer = requestResultDeserializer;
        _producerChannel = CreateProducerModel(connection);
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
            await _notificationHub.Clients.Group(request!.ClientId)
                .SendAsync(request.RequestInformation.MetaInformation.Notification, request.JobId);
            return;
        }

        IBasicProperties properties = _producerChannel.CreateBasicProperties();
        properties.ReplyTo = RabbitMqConsumer.ConsumerQueueName.ToString();
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

        _producerChannel.BasicPublish(exchange: "",
            routingKey: $"{request.RequestInformation.MetaInformation.Type}",
            basicProperties: properties,
            body: message);
    }

    private static IModel CreateProducerModel(IConnection connection)
    {
        IModel producerChannel = connection.CreateModel();
        producerChannel.QueueDeclare(queue: $"{JobType.Simulation}",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);

        producerChannel.QueueDeclare(queue: $"{JobType.Fitting}",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);

        return producerChannel;
    }
}