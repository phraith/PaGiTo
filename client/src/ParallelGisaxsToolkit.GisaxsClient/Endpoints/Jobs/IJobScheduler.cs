using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

public interface IJobScheduler
{
    Task ScheduleJob(Request request, CancellationToken cancellationToken);
}