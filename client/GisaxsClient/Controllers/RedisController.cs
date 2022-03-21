using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using StackExchange.Redis;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;

namespace RedisTest.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize]
    public class RedisController : ControllerBase
    {
        private readonly ILogger<RedisController> logger;
        private readonly IConnectionMultiplexer redis;

        public RedisController(ILogger<RedisController> logger, IConnectionMultiplexer redis)
        {
            this.logger = logger;
            this.redis = redis;
        }

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
            string modifiedData = AppearenceModifier.ApplyColorMap(dataEntry.Intensities.ToArray(), dataEntry.Width, dataEntry.Height, colormapName);
            return Ok(JsonSerializer.Serialize(new FinalResult() { data = modifiedData }));
        }
    }

    public class FinalResult
    {
        public string data { get; set; }
    }

    public class DataEntry
    {
        public List<byte> Intensities { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }
        public string Id { get; set; }
    }

    public class MetaInformation
    {
        public long ClientId { get; set; }
        public long JobId { get; set; }
        public string  ColormapName { get; set; }
    }
}
