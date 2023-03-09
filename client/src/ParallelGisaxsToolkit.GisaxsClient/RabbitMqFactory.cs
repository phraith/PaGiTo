using RabbitMQ.Client;

namespace ParallelGisaxsToolkit.GisaxsClient;

public class RabbitMqFactory : IRabbitMqFactory
{
    public string ConsumerQueueName { get; }
    private readonly IConnection _consumerConnection;
    private readonly IConnection _publisherConnection;

    public RabbitMqFactory(string userName, string password, string hostName, string queueName)
    {
        ConsumerQueueName = queueName;
        ConnectionFactory factory = new ConnectionFactory()
        {
            HostName = hostName,
            UserName = userName,
            Password = password,
            DispatchConsumersAsync = true,
            AutomaticRecoveryEnabled = true
        };

        _consumerConnection = factory.CreateConnection();
        _publisherConnection = factory.CreateConnection();
    }

    public IModel CreateConsumerModel()
    {
        (IModel model, _) = CreateModel(ConsumerQueueName, _consumerConnection);
        return model;
    }

    public IModel CreatePublisherModel(string queueName)
    {
        (IModel model, _) = CreateModel(queueName, _publisherConnection);
        return model;
    }

    private static (IModel model, string queueName) CreateModel(string queueName, IConnection connection)
    {
        IModel channel = connection.CreateModel();
        QueueDeclareOk result = channel.QueueDeclare(queue: queueName,
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);
        return (channel, result.QueueName);
    }
}