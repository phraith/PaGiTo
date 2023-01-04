#nullable enable

using System.Text.Json;
using System.Text.Json.Nodes;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Utility.HashComputer;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling
{
    public class RequestFactory : IRequestFactory
    {
        private readonly IHashComputer _hashComputer;
        private readonly IImageStore _imageStore;

        public RequestFactory(IHashComputer hashComputer, IImageStore imageStore)
        {
            _hashComputer = hashComputer;
            _imageStore = imageStore;
        }

        public Request? CreateRequest(string request, string dataAccessor)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            JsonNode? jsonNode = JsonNode.Parse(request);
            if (jsonNode == null)
            {
                return null;
            }


            JsonNode? clientInfoNode = jsonNode["clientInfo"];
            JsonNode? jobInfoNode = jsonNode["jobInfo"];

            if (jsonNode is not JsonObject payload || !payload.Remove("clientInfo"))
            {
                throw new ArgumentException("Couldn't remove client info!");
            }

            if (clientInfoNode == null || jobInfoNode == null)
            {
                return null;
            }

            var payloadString = payload.ToJsonString();
            string hash = _hashComputer.Hash(payloadString);

            ClientInformation? clientInformation =
                JsonSerializer.Deserialize<ClientInformation>(clientInfoNode.ToJsonString(), options);
            JobInformation? jobInformation =
                JsonSerializer.Deserialize<JobInformation>(jobInfoNode.ToJsonString(), options);

            if (clientInformation == null || jobInformation == null)
            {
                return null;
            }


            List<byte> bytes = new List<byte>();
            if (clientInformation.JobType == "fitting" && jobInformation.ImageId != null)
            {
                var simulationTargets = jobInformation.SimulationTargets;
                var binarySimulationTargetCount = BitConverter.GetBytes(simulationTargets.Count);
                bytes.AddRange(binarySimulationTargetCount);

                foreach (SimulationTarget simulationTarget in simulationTargets)
                {
                    bytes.AddRange(simulationTarget.ToBytes());
                    byte[] data = ProfileFromStore(simulationTarget, jobInformation.ImageId.Value).GetAwaiter()
                        .GetResult();
                    bytes.AddRange(data);
                }
            }

            return new Request(new RequestInformation(jobInformation, clientInformation), dataAccessor, hash, request,
                bytes.ToArray());
        }

        private async Task<byte[]> ProfileFromStore(SimulationTarget simulationTarget, long id)
        {
            var start = simulationTarget.Start;
            var end = simulationTarget.End;

            if (start.X == 0 && start.Y == end.Y)
            {
                double[] horizontalProfile = await _imageStore.GetHorizonalProfile((int)id, start.X, end.X, start.Y);
                byte[] horizontalProfileCount = BitConverter.GetBytes(horizontalProfile.Length);
                return horizontalProfileCount.Concat(horizontalProfile.Reverse().SelectMany(BitConverter.GetBytes))
                    .ToArray();
            }

            double[] verticalProfile = await _imageStore.GetVerticalProfile((int)id, start.Y, end.Y, start.X);
            byte[] verticalProfileCount = BitConverter.GetBytes(verticalProfile.Length);

            return verticalProfileCount.Concat(verticalProfile.Reverse().SelectMany(BitConverter.GetBytes)).ToArray();
        }
    }
}