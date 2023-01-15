namespace ParallelGisaxsToolkit.Gisaxs.Core.JobStore;

public interface IJobStore
{
    Task<IEnumerable<JobInfoWithId>> Get();
    Task<Job?> Get(long id);
    Task Delete(long id);
    Task Insert(Job job);
    Task Insert(IReadOnlyCollection<Job> jobs);
}