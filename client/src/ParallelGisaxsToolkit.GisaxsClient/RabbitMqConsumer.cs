using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.SignalR;
using ParallelGisaxsToolkit.Gisaxs.Core.Hubs;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.Gisaxs.Core.ResultStore;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient;

public interface INotifier
{
    void Notify(string target, string notificationType, string notification);
}

public class MessageHubNotifier : INotifier
{
    private readonly IHubContext<MessageHub> _hubContext;

    public MessageHubNotifier(IHubContext<MessageHub> hubContext)
    {
        _hubContext = hubContext;
    }

    public async void Notify(string target, string notificationType, string notification)
    {
        await _hubContext.Clients.Group(target).SendAsync(notificationType, notification);
    }
}

public class RabbitMqConsumer : BackgroundService
{
    private readonly IHubContext<MessageHub> _notificationHub;
    private readonly IDatabase _redisClient;
    private readonly IResultStore _resultStore;
    private readonly ConcurrentDictionary<string, Request?> _trackedJobs;
    private readonly IRequestResultDeserializer _requestResultDeserializer;
    private readonly IModel _consumerChannel;

    public static Guid ConsumerQueueName { get; } = Guid.NewGuid();

    public RabbitMqConsumer(IServiceScopeFactory serviceScopeFactory, IHubContext<MessageHub> notificationHub)
    {
        _trackedJobs = new ConcurrentDictionary<string, Request?>();
        _requestResultDeserializer = new RequestResultDeserializer();
        _notificationHub = notificationHub;

        IServiceScope scope = serviceScopeFactory.CreateScope();
        IDatabase? redisClient = scope.ServiceProvider.GetService<IDatabase>();
        IResultStore? resultStore = scope.ServiceProvider.GetService<IResultStore>();
        IConnection? connection = scope.ServiceProvider.GetService<IConnection>();

        if (connection == null)
        {
            throw new ArgumentNullException();
        }

        _redisClient = redisClient ?? throw new ArgumentNullException();
        _resultStore = resultStore ?? throw new ArgumentException();
        _consumerChannel = CreateConsumerModel(connection, ConsumerQueueName.ToString());
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        stoppingToken.ThrowIfCancellationRequested();

        AsyncEventingBasicConsumer consumer = new AsyncEventingBasicConsumer(_consumerChannel);
        consumer.Received += EventConsume;
        _consumerChannel.BasicConsume(
            queue: ConsumerQueueName.ToString(),
            autoAck: true,
            consumer: consumer
        );

        await Task.CompletedTask;
    }

    private async Task EventConsume(object? model, BasicDeliverEventArgs deliverEventArgs)
    {
        var basicProperties = deliverEventArgs.BasicProperties;

        IDictionary<string, object>? headers = basicProperties.Headers;
        
        string jobId = basicProperties.CorrelationId;



        string? jobHash = TryGetHeaderValue("jobHash");
        string? clientId = TryGetHeaderValue("clientId");
        string? notification = TryGetHeaderValue("notification");
        string? colormap = TryGetHeaderValue("colormap");


        byte[] response = deliverEventArgs.Body.ToArray();

        if (response.Length <= 0 || jobHash == null || clientId == null || notification == null)
        {
            throw new InvalidDataException("Response was empty!");
        }

        await _redisClient.StringSetAsync(jobHash, response);

        RequestData resultData = _requestResultDeserializer.Deserialize(response, colormap);
        string serializedResult = JsonSerializer.Serialize(
            resultData, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

        await _resultStore.Insert(new Result(serializedResult, jobId, clientId));
        await _notificationHub.Clients.Group(clientId)
            .SendAsync(notification, jobId);

        string? TryGetHeaderValue(string key)
        {
            if (headers == null)
            {
                return null;}
            
            if (!headers.TryGetValue(key, out object? value))
            {
                return null;
            }

            if (value is not byte[] byteValue)
            {
                return null;
            }

            string? encoded = Encoding.UTF8.GetString(byteValue);
            return encoded;
        }
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
}