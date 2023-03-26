﻿namespace ParallelGisaxsToolkit.Gisaxs.Core.JobStore;

public interface IJobStore
{
    Task<IEnumerable<Job>> Get();
    Task<IEnumerable<long>> Count();
    Task<Job?> Get(long jobId);
    Task Delete(long jobId);
    Task Insert(Job job);
    Task Insert(IReadOnlyCollection<Job> jobs);
}