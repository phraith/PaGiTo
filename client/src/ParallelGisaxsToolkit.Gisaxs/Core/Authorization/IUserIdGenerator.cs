namespace ParallelGisaxsToolkit.Gisaxs.Core.Authorization;

public interface IUserIdGenerator
{
    long Generate(string username);
}