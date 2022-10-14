#nullable enable

using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.Options;
using NetMQ;
using Polly;
using Polly.Retry;
using StackExchange.Redis;
using Vraith.GisaxsClient.Controllers;
using Vraith.GisaxsClient.Core.Connection;

namespace Vraith.GisaxsClient.Core.RequestHandling
{
    public class MajordomoRequestHandler : IRequestHandler
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
            if (db.KeyExists($"{request.JobHash}-simple"))
            {
                return new RequestResult
                {
                    Command = "ReceiveJobId",
                    Body = $"colorMapName={request.JobInformation.ColormapName}&hash={request.JobHash}"
                };
            }

            RequestResult? response = null;

            using (var client = new MajordomoClient(connectionStrings.CurrentValue.GisaxsBackend))
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
            int x = BitConverter.ToInt32(data, 0);
            int y = BitConverter.ToInt32(data, sizeof(int));

            int start = 2 * sizeof(int);
            int end = 2 * sizeof(int) + x * y;
            db.StringSet($"{dbKey}-simple", data[start..end]);
            db.StringSet($"{dbKey}-width", x);
            db.StringSet($"{dbKey}-height", y);

            byte[] exactData = data[end..];
            for (int i = 0; i < x; i++)
            {
                byte[] verticalLineprofile = new byte[y * sizeof(double)];
                for (int j = 0; j < y; j++)
                {
                    Array.Copy(exactData, (j * x + i) * sizeof(double), verticalLineprofile, j * sizeof(double), sizeof(double));
                }
                db.StringSet($"{dbKey}-v-{i}", verticalLineprofile);
            }

            for (int i = 0; i < y; i++)
            {
                int lineProfileStart = i * x * sizeof(double);
                int lineProfileEnd = lineProfileStart + x * sizeof(double);
                byte[] horizontalLineprofile = exactData[lineProfileStart..lineProfileEnd];
                db.StringSet($"{dbKey}-h-{i}", horizontalLineprofile);
            }


            return new RequestResult
            {
                Command = "ReceiveJobId",
                Body = $"colorMapName={colormapName}&hash={dbKey}"
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
