using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Options;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.Hubs;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient;

public class RabbitMqService : IProducer
{
    private readonly IHubContext<MessageHub> _notificationHub;
    private readonly IDatabase _redisClient;
    private readonly IModel _producerChannel;
    private readonly Guid _guid;
    private readonly ConcurrentDictionary<string, Request?> _trackedJobs;
    private readonly IRequestResultDeserializer _requestResultDeserializer;
    private readonly IModel _consumerChannel;
    private readonly EventingBasicConsumer _consumer;

    public RabbitMqService(IOptionsMonitor<ConnectionStrings> connectionStrings,
        IHubContext<MessageHub> notificationHub, IDatabase redisClient)
    {
        _trackedJobs = new ConcurrentDictionary<string, Request?>();
        _requestResultDeserializer = new RequestResultDeserializer();
        ConnectionFactory factory = new ConnectionFactory() { HostName = connectionStrings.CurrentValue.RabbitMq };
        IConnection connection = factory.CreateConnection();
        _producerChannel = connection.CreateModel();
        _consumerChannel = connection.CreateModel();
        _guid = Guid.NewGuid();
        
        _notificationHub = notificationHub;
        _redisClient = redisClient;
        
        _producerChannel.QueueDeclare(queue: $"{JobType.Simulation}",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);

        _producerChannel.QueueDeclare(queue: $"{JobType.Fitting}",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);

        _consumerChannel.QueueDeclare(queue: $"{_guid}",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);


        _consumer = new EventingBasicConsumer(_consumerChannel);

        _consumer.Received += (model, ea) =>
        {
            if (!_trackedJobs.TryRemove(ea.BasicProperties.CorrelationId, out Request? request))
            {
                throw new InvalidOperationException(
                    $"Unexpected message with correlation id {ea.BasicProperties.CorrelationId} received!");
            }

            byte[] body = ea.Body.ToArray();

            RequestResult? result = HandleResult(body, request!);
            if (result != null)
            {
                _notificationHub.Clients.Group(request!.ClientId).SendAsync(result.Notification, result.JobId).GetAwaiter().GetResult();
            }
        };

        _consumerChannel.BasicConsume(
            queue: $"{_guid}",
            autoAck: true,
            consumer: _consumer
        );
    }

    public void Produce(Request request)
    {
        IBasicProperties properties = _producerChannel.CreateBasicProperties();
        properties.ReplyTo = _guid.ToString();
        properties.CorrelationId = request.JobId;

        byte[] message = Encoding.UTF8.GetBytes(request.RawRequest);

        if (!_trackedJobs.TryAdd(request.JobId, request))
        {
            throw new InvalidOperationException(
                $"Could not add job with id {request.JobId}!");
        }

        _producerChannel.BasicPublish(exchange: "",
            routingKey: $"{request.RequestInformation.MetaInformation.Type}",
            basicProperties: properties,
            body: message);
    }

    private RequestResult? HandleResult(byte[] contentFrameData, Request request)
    {
        if (contentFrameData.Length <= 0)
        {
            return null;
        }

        string? colormap = request.RequestInformation.MetaInformation.Colormap;

        var resultData = _requestResultDeserializer.Deserialize(contentFrameData, colormap);
        var serialized = JsonSerializer.Serialize(
            resultData, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

        _redisClient.StringSet(request.JobId, serialized);
        return new RequestResult(request.JobId, request.RequestInformation.MetaInformation.Notification);
    }
}

public interface IProducer
{
    void Produce(Request request);
}