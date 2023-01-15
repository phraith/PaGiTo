using System.Diagnostics.CodeAnalysis;
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

        public Request? CreateRequest(string request, string clientId)
        {
            JsonNode? jsonNode = JsonNode.Parse(request);
            if (jsonNode is not JsonObject jsonObject)
            {
                return null;
            }

            if (!TryExtract<MetaInformation>(jsonObject, "meta", out var metaInformation))
            {
                return null;
            }

            if (!TryExtract<JobProperties>(jsonObject, "properties", out var jobProperties))
            {
                return null;
            }

            string jobPropertiesAsString = jsonObject["properties"]!.ToJsonString();
            string configAsString = jsonObject["config"]!.ToJsonString();
            
            string jobHash = _hashComputer.Hash(jobPropertiesAsString, configAsString);

            byte[] imageData = RetrieveImageData(metaInformation, jobProperties).ToArray();

            jsonObject.Add("clientId", clientId);
            jsonObject.Add("jobId", jobHash);
            var requestInformation = new RequestInformation(jobProperties, metaInformation);
            var updatedRequest = jsonObject.ToJsonString();
            return new Request(clientId, jobHash, Guid.NewGuid().ToString(), requestInformation, updatedRequest, imageData);
        }

        private IReadOnlyList<byte> RetrieveImageData(MetaInformation metaInformation, JobProperties jobProperties)
        {
            if (metaInformation.Type != JobType.Fitting || jobProperties.ImageId != null)
            {
                return Array.Empty<byte>();
            }

            List<byte> bytes = new List<byte>();
            var simulationTargets = jobProperties.SimulationTargets;
            var binarySimulationTargetCount = BitConverter.GetBytes(simulationTargets.Count);
            bytes.AddRange(binarySimulationTargetCount);

            foreach (SimulationTarget simulationTarget in simulationTargets)
            {
                bytes.AddRange(simulationTarget.ToBytes());
                byte[] data = ProfileFromStore(simulationTarget, jobProperties.ImageId!.Value).GetAwaiter()
                    .GetResult();
                bytes.AddRange(data);
            }

            return bytes.ToArray();
        }

        private static bool TryExtract<T>(JsonObject jsonObject, string jsonFieldName,
            [NotNullWhen(true)] out T? extracted) where T : class
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            extracted = null;
            if (!jsonObject.ContainsKey(jsonFieldName))
            {
                return false;
            }

            JsonNode? fieldToExtract = jsonObject[jsonFieldName];
            if (fieldToExtract == null)
            {
                return false;
            }

            T? extractedField =
                JsonSerializer.Deserialize<T>(fieldToExtract.ToJsonString(), options);

            if (extractedField == null)
            {
                return false;
            }

            extracted = extractedField;
            return true;
        }

        private async Task<byte[]> ProfileFromStore(SimulationTarget simulationTarget, long id)
        {
            var start = simulationTarget.Start;
            var end = simulationTarget.End;

            if (start.X == 0 && start.Y == end.Y)
            {
                double[] horizontalProfile = await _imageStore.GetHorizontalProfile((int)id, start.X, end.X, start.Y);
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