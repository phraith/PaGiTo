using GisaxsClient.Utility;
using Microsoft.AspNetCore.Mvc;
using StackExchange.Redis;
using System.Diagnostics;
using System.Text.Json;

namespace GisaxsClient.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    //[Authorize]
    public class RedisController : ControllerBase
    {
        private readonly ILogger<RedisController> logger;

        public RedisController(ILogger<RedisController> logger)
        {
            this.logger = logger;

            ThreadPool.SetMinThreads(16, 16);
        }

        [HttpGet("info")]
        public async Task<IActionResult> GetInfo(string hash)
        {
            IDatabase db = RedisConnectorHelper.Connection.GetDatabase();
            if (!db.KeyExists(hash))
            {
                return NotFound();
            }
            string data = await db.StringGetAsync(hash);
            return Ok(data);
        }

        [HttpGet("data")]
        public async Task<IActionResult> GetData(string hash, string colormapName)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            var keyData = $"{hash}-simple";
            var keyWidth = $"{hash}-width";
            var keyHeight = $"{hash}-height";

            IDatabase db = RedisConnectorHelper.Connection.GetDatabase();
            if (!db.KeyExists(keyData) || !db.KeyExists(keyWidth) || !db.KeyExists(keyHeight)) { return NotFound(); }
            byte[] data = await db.StringGetAsync(keyData);
            string heightAsString = await db.StringGetAsync(keyHeight);
            string widthAsString = await db.StringGetAsync(keyWidth);

            int height = int.Parse(heightAsString);
            int width = int.Parse(widthAsString);

            //int x = BitConverter.ToInt32(data, 0);
            //int y = BitConverter.ToInt32(data, sizeof(int));

            //int start = 2 * sizeof(int);
            //int end = 2 * sizeof(int) + x * y;

            //Stopwatch w = new();
            //w.Start();
            //byte[] intensities = data[start..end];
            //w.Stop();
            //Console.WriteLine($"{w.ElapsedMilliseconds} ms");

            string modifiedData = AppearenceModifier.ApplyColorMap(data, width, height, true, colormapName);
            return Ok(JsonSerializer.Serialize(new FinalResult() { data = modifiedData, width = width, height = height }));
        }

        [HttpGet("lineprofiles")]
        public async Task<IActionResult> GetLineprofiles(string hash)
        {
            IDatabase db = RedisConnectorHelper.Connection.GetDatabase();
            if (!db.KeyExists(hash)) { return NotFound(); }

            Stopwatch w = new();
            w.Start();
            byte[] data = db.StringGet(hash);
            w.Stop();
            Console.WriteLine($"{w.ElapsedMilliseconds} ms");
            int x = BitConverter.ToInt32(data, 0);
            int y = BitConverter.ToInt32(data, sizeof(int));

            int start = 2 * sizeof(int);
            int end = 2 * sizeof(int) + x * y;


            byte[] intensities = data[end..];


            return Ok();
        }
    }

    public class FinalResult
    {
        public string data { get; set; }
        public int width { get; set; }
        public int height { get; set; }
    }

    public class DataEntry
    {
        public List<byte> Intensities { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }
        public string Id { get; set; }
    }

    public record MetaInformation
    {
        public long ClientId { get; init; }
        public string JobType { get; init; }
        public long JobId { get; init; }
        public string ColormapName { get; init; }
    }
}
