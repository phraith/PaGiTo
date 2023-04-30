using System.Text;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
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
    private readonly IJobStore _jobStore;
    private readonly IRequestResultDeserializer _requestResultDeserializer;
    private readonly IRabbitMqFactory _rabbitMqFactory;

    public RabbitMqPublisher(INotifier notifier, IDatabase redisClient,
        IJobStore jobStore,
        IRequestResultDeserializer requestResultDeserializer,
        IRabbitMqFactory rabbitMqFactory)
    {
        _notifier = notifier;
        _redisClient = redisClient;
        _jobStore = jobStore;
        _requestResultDeserializer = requestResultDeserializer;
        _rabbitMqFactory = rabbitMqFactory;
    }

    public Task Publish(Request request)
    {
        var publisherQueueName = $"{request.RequestInformation.MetaInformation.Type}";
        string? colormap = request.RequestInformation.MetaInformation.Colormap;
        using IModel publisherChannel = _rabbitMqFactory.CreatePublisherModel(publisherQueueName);
        IBasicProperties properties = publisherChannel.CreateBasicProperties();
        properties.ReplyTo = _rabbitMqFactory.ConsumerQueueName;
        properties.CorrelationId = request.JobId;
        properties.Headers = new Dictionary<string, object>();
        properties.Headers["notification"] = request.RequestInformation.MetaInformation.Notification;
        properties.Headers["clientId"] = request.ClientId;
        properties.Headers["jobHash"] = request.JobHash;
        properties.Headers["jobType"] = request.RequestInformation.MetaInformation.Type.ToString();

        if (colormap != null)
        {
            properties.Headers["colormap"] = colormap;
        }

        byte[] message = RequestToBytes(request);

        publisherChannel.BasicPublish(exchange: "",
            routingKey: $"{request.RequestInformation.MetaInformation.Type}",
            basicProperties: properties,
            body: message);
        return Task.CompletedTask;
    }

    private static byte[] RequestToBytes(Request request)
    {
        byte[] message = Encoding.UTF8.GetBytes(request.RawRequest);
        byte[] messageSize = BitConverter.GetBytes(message.Length);
        byte[] imageSize = BitConverter.GetBytes(request.ImageDataForFitting.Length);
        return messageSize.Concat(message).Concat(imageSize).Concat(request.ImageDataForFitting).ToArray();
    }
}