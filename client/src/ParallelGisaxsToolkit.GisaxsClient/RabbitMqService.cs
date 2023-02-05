using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.SignalR;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.Hubs;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient;

public class RabbitMqService : IGisaxsService
{
    private readonly IHubContext<MessageHub> _notificationHub;
    private readonly IDatabase _redisClient;
    private readonly IModel _producerChannel;
    private readonly Guid _guid;
    private readonly ConcurrentDictionary<string, Request?> _trackedJobs;
    private readonly IRequestResultDeserializer _requestResultDeserializer;

    public RabbitMqService(IConnection connection,
        IHubContext<MessageHub> notificationHub, IDatabase redisClient)
    {
        _trackedJobs = new ConcurrentDictionary<string, Request?>();
        _requestResultDeserializer = new RequestResultDeserializer();
        _guid = Guid.NewGuid();
        _notificationHub = notificationHub;
        _redisClient = redisClient;

        _producerChannel = CreateProducerModel(connection);
        IModel consumerChannel = CreateConsumerModel(connection, _guid.ToString());

        EventingBasicConsumer consumer = new EventingBasicConsumer(consumerChannel);
        consumer.Received += EventConsume;

        consumerChannel.BasicConsume(
            queue: _guid.ToString(),
            autoAck: true,
            consumer: consumer
        );
    }

    public void Issue(Request request)
    {
        byte[]? cachedResponse = _redisClient.StringGet(request.JobHash);
        if (cachedResponse != null)
        {
            Consume(request, cachedResponse);
            return;
        }

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

    private void Consume(Request request, byte[] response)
    {
        if (response.Length <= 0)
        {
            throw new InvalidDataException("Response was empty!");
        }

        _redisClient.StringSet(request.JobHash, response);

        RequestResult result = HandleResult(response, request!);
        _notificationHub.Clients.Group(request!.ClientId).SendAsync(result.Notification, result.JobId).GetAwaiter()
            .GetResult();
    }

    private void EventConsume(object? model, BasicDeliverEventArgs deliverEventArgs)
    {
        string jobId = deliverEventArgs.BasicProperties.CorrelationId;
        if (!_trackedJobs.TryRemove(jobId, out Request? request) || request == null)
        {
            throw new InvalidOperationException(
                $"Unexpected message with correlation id {jobId} received!");
        }

        Consume(request, deliverEventArgs.Body.ToArray());
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

    private static IModel CreateConsumerModel(IConnection connection, string queueName)
    {
        IModel consumerChannel = connection.CreateModel();
        consumerChannel.QueueDeclare(queue: queueName,
            durable: true,
            exclusive: false,
            autoDelete: true,
            arguments: null);

        return consumerChannel;
    }

    private RequestResult HandleResult(byte[] contentFrameData, Request request)
    {
        string? colormap = request.RequestInformation.MetaInformation.Colormap;

        RequestData resultData = _requestResultDeserializer.Deserialize(contentFrameData, colormap);
        string serialized = JsonSerializer.Serialize(
            resultData, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

        _redisClient.StringSet(request.JobId, serialized);

        return new RequestResult(request.JobId, request.RequestInformation.MetaInformation.Notification);
    }
}