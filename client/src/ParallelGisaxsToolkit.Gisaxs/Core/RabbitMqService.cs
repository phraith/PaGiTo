using System.Collections.Concurrent;
using System.Text;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using RabbitMQ.Client.Events;

namespace ParallelGisaxsToolkit.Gisaxs.Core;

using RabbitMQ.Client;

public class RabbitMqService : IProducer
{
    private readonly IModel _channel;
    private readonly Guid _guid;
    private readonly EventingBasicConsumer _consumer;
    private readonly ConcurrentDictionary<string, Request> _trackedJobs;

    public RabbitMqService(string hostName)
    {
        _guid = Guid.NewGuid();
        _trackedJobs = new ConcurrentDictionary<string, Request>();

        ConnectionFactory factory = new ConnectionFactory() { HostName = hostName };
        IConnection connection = factory.CreateConnection();
        _channel = connection.CreateModel();

        _channel.QueueDeclare(queue: $"{JobType.Simulation}",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);

        _channel.QueueDeclare(queue: $"{JobType.Fitting}",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);

        _channel.QueueDeclare(queue: $"{_guid}",
            durable: true,
            exclusive: false,
            autoDelete: false,
            arguments: null);


        _consumer = new EventingBasicConsumer(_channel);

        _consumer.Received += (model, ea) =>
        {
            if (!_trackedJobs.TryRemove(ea.BasicProperties.CorrelationId, out _))
            {
                throw new InvalidOperationException(
                    $"Unexpected message with correlation id {ea.BasicProperties.CorrelationId} received!");
            }

            byte[] body = ea.Body.ToArray();
            string response = Encoding.UTF8.GetString(body);
        };
    }

    public void Produce(Request request)
    {
        IBasicProperties properties = _channel.CreateBasicProperties();
        properties.ReplyTo = _guid.ToString();
        properties.CorrelationId = request.JobHash;

        byte[] message = Encoding.UTF8.GetBytes(request.RawRequest);

        if (!_trackedJobs.TryAdd(request.JobHash, request))
        {
            throw new InvalidOperationException(
                $"Could not job with id {request.JobHash}!");
        }

        _channel.BasicPublish(exchange: "",
            routingKey: $"{request.RequestInformation.MetaInformation.Type}",
            basicProperties: properties,
            body: message);
    }
}

public interface IProducer
{
    void Produce(Request request);
}