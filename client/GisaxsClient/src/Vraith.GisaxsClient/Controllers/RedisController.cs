using System.Text.Json;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using StackExchange.Redis;
using Vraith.Gisaxs.Configuration;
using Vraith.Gisaxs.Core.ImageStore;
using Vraith.Gisaxs.Utility.ImageTransformations;

namespace Vraith.GisaxsClient.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize]
    public class RedisController : ControllerBase
    {
        private readonly ILogger<RedisController> logger;
        private readonly ConnectionMultiplexer connection;

        public RedisController(IOptionsMonitor<ConnectionStrings> connectionStrings, ILogger<RedisController> logger)
        {
            this.logger = logger;
            this.connection = ConnectionMultiplexer.Connect(connectionStrings.CurrentValue.Redis);
            ThreadPool.SetMinThreads(16, 16);
        }

        [HttpGet("info")]
        public async Task<IActionResult> GetInfo(string hash)
        {
            IDatabase db = connection.GetDatabase();
            if (!db.KeyExists(hash))
            {
                return NotFound();
            }

            string? data = await db.StringGetAsync(hash);
            if (data == null)
            {
                logger.LogError("$data is null", data);
                return NoContent();
            }

            return Ok(data);
        }

        [HttpGet("data")]
        public async Task<IActionResult> GetData(string hash)
        {
            IDatabase db = connection.GetDatabase();
            if (!db.KeyExists(hash))
            {
                return NotFound();
            }

            byte[]? dim = await db.StringGetRangeAsync(hash, 0, 2 * sizeof(int));

            int width = BitConverter.ToInt32(dim, 0);
            int height = BitConverter.ToInt32(dim, sizeof(int));

            int start = 2 * sizeof(int);
            int end = 2 * sizeof(int) + width * height * sizeof(double);
            byte[]? data = await db.StringGetRangeAsync(hash, start, end);

            double[] modifiedData = new double[data.Length / 8];
            Buffer.BlockCopy(data, 0, modifiedData, 0, modifiedData.Length * 8);
            var logData = modifiedData.Select(x => Math.Log(x + 1)).Reverse().ToArray();
            return Ok(JsonSerializer.Serialize(
                new NumericResult(logData, width, height), new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                }));
        }
        
        [HttpGet("image")]
        public async Task<IActionResult> GetImage(string hash, string colormapName)
        {
            IDatabase db = connection.GetDatabase();
            if (!db.KeyExists(hash))
            {
                return NotFound();
            }

            byte[]? dim = await db.StringGetRangeAsync(hash, 0, 2 * sizeof(int));

            int width = BitConverter.ToInt32(dim, 0);
            int height = BitConverter.ToInt32(dim, sizeof(int));

            int start = 2 * sizeof(int);
            int end = 2 * sizeof(int) + width * height;
            byte[]? data = await db.StringGetRangeAsync(hash, start, end);

            string modifiedData = AppearanceModifier.ApplyColorMap(data, width, height, true, colormapName);
            return Ok(JsonSerializer.Serialize(
                new JpegResult(modifiedData, width, height), new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                }));
        }
    }

    public record NumericResult(double[] ModifiedData, int Width, int Height);
}