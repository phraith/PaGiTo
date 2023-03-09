using RabbitMQ.Client;

namespace ParallelGisaxsToolkit.GisaxsClient;

public interface IRabbitMqFactory
{
    string ConsumerQueueName { get;  }
    IModel CreateConsumerModel();
    IModel CreatePublisherModel(string queueName);
}