using System.Text;
using System.Text.Json;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.Gisaxs.Core.ResultStore;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient;

public class RabbitMqConsumer : BackgroundService
{
    private readonly IServiceScopeFactory _serviceScopeFactory;
    private readonly IRabbitMqService _rabbitMqService;

    public RabbitMqConsumer(IServiceScopeFactory serviceScopeFactory, IRabbitMqService rabbitMqService)
    {
        _serviceScopeFactory = serviceScopeFactory;
        _rabbitMqService = rabbitMqService;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        stoppingToken.ThrowIfCancellationRequested();

        AsyncEventingBasicConsumer consumer = new AsyncEventingBasicConsumer(_rabbitMqService.ConsumerChannel);
        consumer.Received += EventConsume;
        _rabbitMqService.ConsumerChannel.BasicConsume(
            queue: string.Empty,
            autoAck: true,
            consumer: consumer
        );

        await Task.CompletedTask;
    }

    private async Task EventConsume(object? model, BasicDeliverEventArgs deliverEventArgs)
    {
        using IServiceScope scope = _serviceScopeFactory.CreateScope();
        IDatabase redisClient = scope.ServiceProvider.GetRequiredService<IDatabase>();
        IResultStore resultStore = scope.ServiceProvider.GetRequiredService<IResultStore>();
        INotifier notifier = scope.ServiceProvider.GetRequiredService<INotifier>();
        IRequestResultDeserializer requestResultDeserializer = scope.ServiceProvider.GetRequiredService<IRequestResultDeserializer>();

        IBasicProperties basicProperties = deliverEventArgs.BasicProperties;

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

        await redisClient.StringSetAsync(jobHash, response);

        RequestData resultData = requestResultDeserializer.Deserialize(response, colormap);
        string serializedResult = JsonSerializer.Serialize(
            resultData, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

        await resultStore.Insert(new Result(serializedResult, jobId, clientId));
        await notifier.Notify(clientId, notification, jobId);

        string? TryGetHeaderValue(string key)
        {
            if (headers == null)
            {
                return null;
            }

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
}