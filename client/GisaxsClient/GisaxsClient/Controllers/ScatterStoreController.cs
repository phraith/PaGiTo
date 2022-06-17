using GisaxsClient.Core.ImageStore;
using GisaxsClient.Utility.ImageTransformations;
using Microsoft.AspNetCore.Mvc;

namespace GisaxsClient.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ScatterStoreController : ControllerBase
    {
        private readonly ILogger<ScatterStoreController> logger;
        private readonly ImageStore imageStore;

        public ScatterStoreController(ILogger<ScatterStoreController> logger, IConfiguration configuration)
        {
            this.logger = logger;
            imageStore = new ImageStore(configuration);
        }

        [HttpGet("info")]
        public async Task<IEnumerable<ImageInfo>> Get()
        {
            return await imageStore.Get();
        }

        [HttpGet("get")]
        public async Task<string> Get(int id)
        {
            var image = await imageStore.Get(id);

            if (image == null) { return string.Empty; }

            var maxIntensity = image.Data.Max();
            Console.WriteLine($"BornAgain {maxIntensity}");
            byte[] normalizedImage = image.Data.Select(x => Normalize(x, maxIntensity)).ToArray();
            var base64 = AppearanceModifier.ApplyColorMap(normalizedImage, image.Width, image.Height, false, "twilight");
            return base64;
        }

        [HttpPost("push")]
        [RequestSizeLimit(100_000_000)]
        public void Push(Image image)
        {
            imageStore.Insert(image);
        }

        private byte Normalize(double intensity, double max)
        {
            double logmax = Math.Log(max);
            double logmin = Math.Log(Math.Max(2, 1e-10 * max));

            double logval = Math.Log(intensity);
            logval /= logmax - logmin;
            return (byte)(logval * 255.0);
        }
    }
}