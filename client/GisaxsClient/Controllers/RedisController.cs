using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using StackExchange.Redis;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace RedisTest.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class RedisController : ControllerBase
    {
        private readonly ILogger<RedisController> logger;
        private readonly IConnectionMultiplexer redis;

        public RedisController(ILogger<RedisController> logger, IConnectionMultiplexer redis)
        {
            this.logger = logger;
            this.redis = redis;
        }
        
        public class DataRequest
        {
            public string Hash { get; set; }
        }

        public class DatabaseRequest
        {
            public string id { get; set; }
            public char[] intensities { get; set; }
        }

        //[HttpGet("{id}")]
        [HttpGet]
        public async Task<IActionResult> GetData(string hash, string colormapName)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            IDatabase db = redis.GetDatabase();
            if (!db.KeyExists(hash)) { return NotFound();}
            RedisValue data = await db.StringGetAsync(hash);
            var dataString = data.ToString();
            var dataEntry = JsonSerializer.Deserialize<DataEntry>(dataString, options);
            string modifiedData = AppearenceModifier.ApplyColorMap(dataEntry.Data, dataEntry.Width, dataEntry.Height, colormapName);

            return Ok(JsonSerializer.Serialize(new FinalResult() { data = modifiedData }));

            //return Ok($"{{data : \"{modifiedData}\"}}");
        }
    }

    public class FinalResult
    {
        public string data { get; set; }
    }

    public class DataEntry
    {
        public byte[] Data { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }
    }

    public class GisaxsConfigWithMetaInformation
    {
        public FormDataGisaxsConfig Config { get; set; }

        public MetaInformation Info { get; set; }
    }

    public class MetaInformation
    {
        public long ClientId { get; set; }
        public long JobId { get; set; }
        public string  ColormapName { get; set; }
    }

    public class GisaxsConfig
    {
        public string name { get; set; }
        public IReadOnlyCollection<ShapeConfig> shapes { get; set; }
        public UnitcellConfig unitcell { get; set; }
        public SubstrateConfig substrate { get; set; }
    }

    public class SubstrateConfig
    {
        public int order { get; set; }
        public RefractionIndex refindex { get; set; }
    }

    public class UnitcellConfig
    {
        public IReadOnlyCollection<ComponentConfig> components { get; set; }
        public IReadOnlyCollection<int> repetitions { get; set; }
        public IReadOnlyCollection<double> distances { get; set; }
    }

    public class ComponentConfig
    {
        public string shape { get; set; }
        public IReadOnlyCollection<IReadOnlyCollection<int>> locations { get; set; }
    }
}
