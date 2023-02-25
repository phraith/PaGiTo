using ParallelGisaxsToolkit.Gisaxs.Configuration;
using RabbitMQ.Client;

namespace ParallelGisaxsToolkit.GisaxsClient;

public class RabbitMqService : IRabbitMqService
{
    public string ConsumerQueueName { get; }
    public IModel ConsumerChannel { get; }
    public IModel PublisherChannel { get; }

    public RabbitMqService(IConnection rabbitMqConnection)
    {
        var (consumerChannel, queueName) = CreateConsumerModel(rabbitMqConnection);
        var publisherChannel = CreateProducerModel(rabbitMqConnection);
        
        ConsumerChannel = consumerChannel;
        PublisherChannel = publisherChannel;
        ConsumerQueueName = queueName;
    }
    
    private static (IModel, string) CreateConsumerModel(IConnection connection)
    {
        IModel consumerChannel = connection.CreateModel();
        QueueDeclareOk result = consumerChannel.QueueDeclare(queue: string.Empty,
            durable: true,
            exclusive: false,
            autoDelete: true,
            arguments: null);


        return (consumerChannel, result.QueueName);
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