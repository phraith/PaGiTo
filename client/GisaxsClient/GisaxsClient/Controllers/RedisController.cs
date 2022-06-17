﻿using GisaxsClient.Utility.ImageTransformations;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using StackExchange.Redis;
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

            string modifiedData = AppearanceModifier.ApplyColorMap(data, width, height, true, colormapName);
            return Ok(JsonSerializer.Serialize(new FinalResult() { data = modifiedData, width = width, height = height }));
        }
    }
}