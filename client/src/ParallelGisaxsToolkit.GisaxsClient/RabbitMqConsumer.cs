using System.Text;
using System.Text.Json;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient;

public class RabbitMqConsumer : BackgroundService
{
    private readonly IServiceScopeFactory _serviceScopeFactory;
    private readonly IRabbitMqFactory _rabbitMqFactory;
    private IModel _consumerChannel;

    public RabbitMqConsumer(IServiceScopeFactory serviceScopeFactory, IRabbitMqFactory rabbitMqFactory)
    {
        _serviceScopeFactory = serviceScopeFactory;
        _rabbitMqFactory = rabbitMqFactory;
        _consumerChannel = rabbitMqFactory.CreateConsumerModel();
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        stoppingToken.ThrowIfCancellationRequested();
        AsyncEventingBasicConsumer consumer = new AsyncEventingBasicConsumer(_consumerChannel);
        consumer.Received += EventConsume;
        consumer.Shutdown += Reconnect;
        _consumerChannel.BasicConsume(
            queue: string.Empty,
            autoAck: true,
            consumer: consumer
        );

        await Task.CompletedTask;
    }

    private Task Reconnect(object? model, ShutdownEventArgs? args)
    {
        while (!_consumerChannel.IsOpen)
        {
            _consumerChannel = _rabbitMqFactory.CreateConsumerModel();
        }

        return Task.CompletedTask;
    }

    private async Task EventConsume(object? model, BasicDeliverEventArgs deliverEventArgs)
    {
        using IServiceScope scope = _serviceScopeFactory.CreateScope();
        IDatabase redisClient = scope.ServiceProvider.GetRequiredService<IDatabase>();
        IJobStore jobStore = scope.ServiceProvider.GetRequiredService<IJobStore>();
        INotifier notifier = scope.ServiceProvider.GetRequiredService<INotifier>();
        IRequestResultDeserializer requestResultDeserializer =
            scope.ServiceProvider.GetRequiredService<IRequestResultDeserializer>();

        IBasicProperties basicProperties = deliverEventArgs.BasicProperties;

        IDictionary<string, object>? headers = basicProperties.Headers;

        string jobId = basicProperties.CorrelationId;

        string? jobTypeAsString = TryGetHeaderValue("jobType");
        string? jobHash = TryGetHeaderValue("jobHash");
        string? clientId = TryGetHeaderValue("clientId");
        string? notification = TryGetHeaderValue("notification");
        string? colormap = TryGetHeaderValue("colormap");

        byte[] response = deliverEventArgs.Body.ToArray();

        if (response.Length <= 0 || jobHash == null || clientId == null || notification == null ||
            jobTypeAsString == null)
        {
            throw new InvalidDataException("Response was empty!");
        }

        await redisClient.StringSetAsync(jobHash, response);

        JobType jobType = Enum.Parse<JobType>(jobTypeAsString);
        IRequestData resultData = requestResultDeserializer.Deserialize(response, jobType, colormap);
        string serializedResult = resultData.Serialize();

        Job? job = await jobStore.Get(jobId, clientId, false, false);
        if (job == null)
        {
            throw new ArgumentNullException(nameof(Job), "Job to update does not exist");
        }

        await jobStore.Update(job with { FinishDate = DateTime.Now, Result = serializedResult });
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