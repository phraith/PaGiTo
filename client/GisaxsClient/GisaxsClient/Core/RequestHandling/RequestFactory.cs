using GisaxsClient.Controllers;
using GisaxsClient.Utility.HashComputer;
using System.Text.Json;
using System.Text.Json.Nodes;

#nullable enable

namespace GisaxsClient.Core.RequestHandling
{
    internal class RequestFactory : IRequestFactory
    {
        private readonly IHashComputer hashComputer;
        public RequestFactory(IHashComputer hashComputer)
        {
            this.hashComputer = hashComputer;
        }

        public Request? CreateRequest(string request)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            JsonNode? jsonNode = JsonNode.Parse(request);
            if (jsonNode == null) { return null; }

            var info = jsonNode["info"];
            var config = jsonNode["config"];
            if (info == null || config == null) { return null; }

            string configJson = config.ToString();
            string hash = hashComputer.Hash(configJson);

            MetaInformation? metaInformation = JsonSerializer.Deserialize<MetaInformation>(info.ToString(), options);
            return new Request(metaInformation!, hash, configJson);
        }
    }
}