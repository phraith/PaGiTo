#nullable enable

using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.Options;
using NetMQ;
using Polly;
using Polly.Retry;
using StackExchange.Redis;
using Vraith.Gisaxs.Configuration;
using Vraith.Gisaxs.Core.Connection;

namespace Vraith.Gisaxs.Core.RequestHandling
{
    internal class MajordomoRequestHandler : IRequestHandler
    {
        private readonly IDatabase db;
        private readonly RetryPolicy retryPolicy;
        private readonly IOptionsMonitor<ConnectionStrings> connectionStrings;

        public MajordomoRequestHandler(IOptionsMonitor<ConnectionStrings> connectionStrings)
        {
            this.db = ConnectionMultiplexer.Connect(connectionStrings.CurrentValue.Redis).GetDatabase();

            retryPolicy = Policy.Handle<TransientException>()
                .WaitAndRetry(retryCount: 3, sleepDurationProvider: i => TimeSpan.FromSeconds(50000));
            this.connectionStrings = connectionStrings;
        }

        public RequestResult? HandleRequest(Request request)
        {
            if (db.KeyExists(request.JobHash))
            {
                return new RequestResult
                {
                    SignalREndpoint = request.DataAccessor,
                    DataAccessor = request.JobHash
                };
            }

            RequestResult? response = null;

            using var client = new MajordomoClient(connectionStrings.CurrentValue.GisaxsBackend);
            var attempt = 0;
            retryPolicy.Execute(() =>
            {
                NetMQMessage msg = new();
                msg.Append(request.RawRequest);
                msg.Append(CreateTestImageData());
                Console.WriteLine($"Attempt {++attempt}");
                client.Send(request.RequestInformation.ClientInformation.JobType, msg);

                NetMQFrame frame = NetMQFrame.Empty;
                JobIntermediateInfo intermediateInfo = new() { Values = new List<double>() };

                while ((frame = NextPayload()) != NetMQFrame.Empty)
                {
                    string frameContent = frame.ConvertToString();
                    if (!frameContent.StartsWith("info:"))
                    {
                        break;
                    }

                    string[] infoSplit = frameContent.Split("info:", StringSplitOptions.RemoveEmptyEntries);

                    if (infoSplit.Length == 1)
                    {
                        var parsedValue = double.Parse(infoSplit[0], CultureInfo.InvariantCulture);
                        intermediateInfo.Values.Add(parsedValue);
                    }

                    db.StringSet(request.InfoHash,
                        JsonSerializer.Serialize(new JobIntermediateInfo()
                            { Values = intermediateInfo.Values.TakeLast(100).ToList() }));
                    Console.WriteLine($"Info: {frameContent}");
                }

                if (frame == NetMQFrame.Empty)
                {
                    throw new TransientException();
                }

                response = request.RequestInformation.ClientInformation.JobType switch
                {
                    "simulation" => HandleSimulationResult(frame.ToByteArray(), request.JobHash, request.DataAccessor),
                    "fitting" => null,
                    _ => throw new TransientException()
                };
            });

            NetMQFrame NextPayload()
            {
                NetMQMessage? currentMessage = client.Receive(request.RequestInformation.ClientInformation.JobType);
                if (currentMessage == null || currentMessage.IsEmpty)
                {
                    throw new TransientException();
                }

                return currentMessage.First;
            }

            return response;
        }

        private RequestResult? HandleSimulationResult(byte[] contentFrameData, string hash, string dataAccessor)
        {

            if (contentFrameData.Length > 0)
            {
                db.StringSet($"{hash}", contentFrameData);
                return new RequestResult
                {
                    SignalREndpoint = dataAccessor,
                    DataAccessor = hash
                };
            }

            return null;
        }

        private static byte[] CreateTestImageData()
        {
            var lp0 = new double[] { 1.0, 2.0, 1.0, 2.0, 1.0 }.SelectMany(value => BitConverter.GetBytes(value))
                .ToArray();
            var offsetLp0 = new int[] { 0, 1, 2, 3, 4 }.SelectMany(value => BitConverter.GetBytes(value)).ToArray();
            var lp0PxCount = BitConverter.GetBytes(5);
            var lpCount = BitConverter.GetBytes(1);
            return lpCount.Concat(lp0PxCount).Concat(lp0).Concat(offsetLp0).ToArray();
        }
    }
}