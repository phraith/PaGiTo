using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ScatterStore.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ScatterStoreController : ControllerBase
    {
        private readonly ILogger<ScatterStoreController> logger;
        private readonly ImageStore imageStore;

        public ScatterStoreController(ILogger<ScatterStoreController> logger, IConfiguration configuration)
        {
            this.logger = logger;
            imageStore = new ImageStore(configuration);
        }

        [HttpGet]
        public async Task<IEnumerable<ImageInfo>> Get()
        {
            return await imageStore.Get();        
        }

        [HttpPost]
        [RequestSizeLimit(100_000_000)]
        public void Push(Image image)
        {
            imageStore.Insert(image);   
        }

        //[HttpPost]
        //public void Remove(IReadOnlyCollection<ImageInfo> infos)
        //{
        //}
    }
}