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
        private readonly ILogger<JobStoreController> _logger;
        private readonly JobStore _jobStore;

        public JobStoreController(ILogger<JobStoreController> logger, IConfiguration configuration)
        {
            this._logger = logger;
            _jobStore = new JobStore(configuration);
        }

        //[Authorize]
        [HttpGet("info")]
        public async Task<IEnumerable<JobInfoDto>> Get()
        {
            var res = await _jobStore.Get();
            return res;
        }

        [Authorize]
        [HttpGet("get")]
        public async Task<string> Get(int id)
        {
            var job = await _jobStore.Get(id);
            if (job == null) { return string.Empty; }
            return JsonSerializer.Serialize(job);
        }

        [HttpPost("push")]
        public void Push(Job job)
        {
            _jobStore.Insert(job);
        }
    }
}
