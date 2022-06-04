using GisaxsClient;
using NetMQ;
using Polly;
using Polly.Retry;
using StackExchange.Redis;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;

#nullable enable

namespace GisaxsClient.Utility
{
    public class MajordomoRequestHandler : IRequestHandler
    {
        private readonly IDatabase db;
        private readonly RetryPolicy retryPolicy;
        public MajordomoRequestHandler()
        {
            db = RedisConnectorHelper.Connection.GetDatabase();
            retryPolicy = Policy.Handle<TransientException>()
                .WaitAndRetry(retryCount: 3, sleepDurationProvider: i => TimeSpan.FromSeconds(5));
        }

        public RequestResult? HandleRequest(Request request)
        {
            if (db.KeyExists(request.JobHash))
            {
                return new RequestResult
                {
                    Command = "ReceiveJobId",
                    Body = $"hash={request.JobHash}&colorMapName={request.JobInformation.ColormapName}"
                };
            }

            RequestResult? response = null;

            using (var client = new MajordomoClient("tcp://127.0.0.1:5555"))
            {
                var attempt = 0;
                retryPolicy.Execute(() =>
                {
                    NetMQMessage msg = new();
                    msg.Append(request.Body);
                    msg.Append(CreateTestImageData());
                    Console.WriteLine($"Attempt {++attempt}");
                    client.Send(request.JobInformation.JobType, msg);

                    NetMQFrame frame = NetMQFrame.Empty;
                    JobIntermediateInfo intermediateInfo = new() { Values = new List<double>() };

                    while ((frame = NextPayload()) != NetMQFrame.Empty)
                    {
                        string frameContent = frame.ConvertToString();
                        if (!frameContent.StartsWith("info:")) { break; }

                        string[] infoSplit = frameContent.Split("info:", StringSplitOptions.RemoveEmptyEntries);

                        if (infoSplit.Length == 1)
                        {
                            var parsedValue = double.Parse(infoSplit[0], CultureInfo.InvariantCulture);
                            intermediateInfo.Values.Add(parsedValue);
                        }

                        db.StringSet(request.InfoHash, JsonSerializer.Serialize(new JobIntermediateInfo() { Values = intermediateInfo.Values.TakeLast(100).ToList() }));
                        Console.WriteLine($"Info: {frameContent}");
                    }

                    if (frame == NetMQFrame.Empty) { throw new TransientException(); }

                    response = request.JobInformation.JobType switch
                    {
                        "sim" => HandleSimulationResult(frame, request.JobHash, request.JobInformation.ColormapName),
                        "fit" => null,
                        _ => throw new TransientException()
                    };
                });

                NetMQFrame NextPayload()
                {
                    NetMQMessage? currentMessage = client.Receive(request.JobInformation.JobType);
                    if (currentMessage == null || currentMessage.IsEmpty) { throw new TransientException(); }
                    return currentMessage.First;
                }
            };

            return response;
        }

        private RequestResult HandleSimulationResult(NetMQFrame contentFrame, string dbKey, string colormapName)
        {
            byte[] data = contentFrame.ToByteArray();
            db.StringSet(dbKey, data);

            return new RequestResult
            {
                Command = "ReceiveJobId",
                Body = $"hash={dbKey}&colorMapName={colormapName}"
            };
        }

        private static byte[] CreateTestImageData()
        {
            var lp0 = new double[] { 1.0, 2.0, 1.0, 2.0, 1.0 }.SelectMany(value => BitConverter.GetBytes(value)).ToArray();
            var offsetLp0 = new int[] { 0, 1, 2, 3, 4 }.SelectMany(value => BitConverter.GetBytes(value)).ToArray();
            var lp0PxCount = BitConverter.GetBytes(5);
            var lpCount = BitConverter.GetBytes(1);
            return lpCount.Concat(lp0PxCount).Concat(lp0).Concat(offsetLp0).ToArray();
        }
    }
}
