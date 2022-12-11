using System.Text.Json;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Vraith.Gisaxs.Configuration;
using Vraith.Gisaxs.Core.ImageStore;
using Vraith.Gisaxs.Utility.Images;
using Vraith.Gisaxs.Utility.ImageTransformations;

namespace Vraith.GisaxsClient.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ScatterStoreController : ControllerBase
    {
        private readonly ILogger<ScatterStoreController> _logger;
        private readonly ImageStore _imageStore;

        public ScatterStoreController(ILogger<ScatterStoreController> logger, IConfiguration configuration)
        {
            this._logger = logger;
            _imageStore = new ImageStore(configuration);
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
            var image = await _imageStore.Get(id);
            if (image == null)
            {
                return string.Empty;
            }

            var maxIntensity = image.RowWiseData.Max();
            Console.WriteLine($"BornAgain {maxIntensity}");
            byte[] normalizedImage = image.RowWiseData.Select(x => Normalize(x, maxIntensity)).ToArray();
            var base64 = AppearanceModifier.ApplyColorMap(normalizedImage, image.Info.Width, image.Info.Height, false,
                colormap);
            return base64;
        }

        [HttpPost("push")]
        [RequestSizeLimit(100_000_000)]
        public void Push(Image image)
        {
            _imageStore.Insert(image);
        }

        private static byte Normalize(double intensity, double max)
        {
            double logmax = Math.Log(max);
            double logmin = Math.Log(Math.Max(2, 1e-10 * max));

            double logval = Math.Log(intensity);
            logval /= logmax - logmin;
            return (byte)(logval * 255.0);
        }
    }
}