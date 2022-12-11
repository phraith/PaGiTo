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
            if (jsonNode == null) { return null; }


            var clientInfo = jsonNode["clientInfo"];
            var jobInfo = jsonNode["jobInfo"];
            
            var payload = jsonNode as JsonObject;
            if (payload == null || !payload.Remove("clientInfo"))
            {
                throw new ArgumentException("Couldnt remove client info!");
            }

            if (payload == null || clientInfo == null) { return null; }

            var payloadString = payload.ToJsonString();
            string hash = _hashComputer.Hash(payloadString);

            var c = JsonSerializer.Deserialize<ClientInformation>(clientInfo.ToJsonString(), options);
            var j = JsonSerializer.Deserialize<JobInformation>(jobInfo.ToJsonString(), options);
            
            return new Request(new RequestInformation(j, c), dataAccessor, hash, request);
        }
    }
}