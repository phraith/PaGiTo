using System.Text.Json;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Vraith.Gisaxs.Core.JobStore;

namespace Vraith.GisaxsClient.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class JobStoreController : ControllerBase
    {
        private readonly ILogger<JobStoreController> logger;
        private readonly JobStore jobStore;

        public JobStoreController(ILogger<JobStoreController> logger, IConfiguration configuration)
        {
            this.logger = logger;
            jobStore = new JobStore(configuration);
        }

        //[Authorize]
        [HttpGet("info")]
        public async Task<IEnumerable<JobInfoDto>> Get()
        {
            return await jobStore.Get();
        }

        [Authorize]
        [HttpGet("get")]
        public async Task<string> Get(int id)
        {
            var job = await jobStore.Get(id);
            if (job == null) { return string.Empty; }
            return JsonSerializer.Serialize(job);
        }

        [HttpPost("push")]
        public void Push(Job job)
        {
            jobStore.Insert(job);
        }
    }
}
