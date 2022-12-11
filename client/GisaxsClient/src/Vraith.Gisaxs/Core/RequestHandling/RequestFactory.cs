#nullable enable

using System.Text.Json;
using System.Text.Json.Nodes;
using Vraith.Gisaxs.Configuration;
using Vraith.Gisaxs.Utility.HashComputer;

namespace Vraith.Gisaxs.Core.RequestHandling
{
    public class RequestFactory : IRequestFactory
    {
        private readonly IHashComputer _hashComputer;

        public RequestFactory(IHashComputer hashComputer)
        {
            _hashComputer = hashComputer;
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

            return new Request(new RequestInformation(jobInformation, clientInformation), dataAccessor, hash, request);
        }
    }
}