using System.Diagnostics;
using System.Text.Json;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Utility.Images;
using ParallelGisaxsToolkit.Gisaxs.Utility.ImageTransformations;

namespace ParallelGisaxsToolkit.GisaxsClient.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ScatterStoreController : ControllerBase
    {
        private readonly ILogger<ScatterStoreController> _logger;
        private readonly IImageStore _imageStore;

        public ScatterStoreController(ILogger<ScatterStoreController> logger, IImageStore imageStore)
        {
            _logger = logger;
            _imageStore = imageStore;
        }

        [Authorize]
        [HttpGet("info")]
        public async Task<IEnumerable<ImageInfoDto>> Get()
        {
            return await _imageStore.Get();
        }


        [Authorize]
        [HttpPost("profile")]
        public async Task<IActionResult> Profile(SimulationTargetWithId targetWithId)
        {
            var id = targetWithId.Id;
            var target = targetWithId.Target;

            var start = target.Start;
            var end = target.End;

            if (start.X == 0 && start.Y == end.Y)
            {
                double[] horizontalProfile = await _imageStore.GetHorizonalProfile(id, start.X, end.X, start.Y);
                var horizontalLogData = horizontalProfile.Select(x => Math.Log(x + 1)).Reverse().ToArray();
                // var horizontalLogData = horizontalProfile.Reverse().ToArray();
                return Ok(JsonSerializer.Serialize(
                    new NumericResult(horizontalLogData, horizontalLogData.Length, 1), new JsonSerializerOptions
                    {
                        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                    }));
            }

            double[] verticalProfile = await _imageStore.GetVerticalProfile(id, start.Y, end.Y, start.X);
            var verticalLogData = verticalProfile.Select(x => Math.Log(x + 1)).Reverse().ToArray();
            // var verticalLogData = verticalProfile.Reverse().ToArray();

            return Ok(JsonSerializer.Serialize(
                new NumericResult(verticalLogData, 1, verticalLogData.Length), new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                }));
        }

        [Authorize]
        [HttpGet("get")]
        public async Task<string> Get(int id, string colormap)
        {

            SimpleImage image = await _imageStore.Get(id);
            if (image == null)
            {
                return string.Empty;
            }

            var s = new Stopwatch();
            s.Start();
            var base64 = AppearanceModifier.ApplyColorMap(image.GreyscaleData.ToArray(), image.Info.Width,
                image.Info.Height, false,
                colormap);
            s.Stop();
            
            Console.WriteLine($"Postgres took {s.ElapsedMilliseconds} ms");
            
            return base64;
        }

        [HttpPost("push")]
        [RequestSizeLimit(100_000_000)]
        public void Push(Image image)
        {
            _imageStore.Insert(image);
        }
    }
}