using ParallelGisaxsToolkit.Gisaxs.Core.UserStore;

namespace ParallelGisaxsToolkit.Gisaxs.Core.ResultStore;

public interface IResultStore
{
    Task<IEnumerable<Result>> Get();
    Task<Result?> Get(string jobId);
    Task Delete(long id);
    Task Insert(Result result);
    Task Insert(IReadOnlyList<Result> results);
}