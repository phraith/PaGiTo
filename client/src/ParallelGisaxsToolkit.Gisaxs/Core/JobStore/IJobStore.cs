namespace ParallelGisaxsToolkit.Gisaxs.Core.JobStore;

public interface IJobStore
{
    Task<IEnumerable<Job>> Get();
    Task Update(Job job);
    Task<IEnumerable<long>> Count();
    Task<Job?> Get(string jobId, string userId, bool includeConfig, bool includeResult);
    Task Delete(long jobId);
    Task Insert(Job job);
}