using Microsoft.Extensions.Options;
using ParallelGisaxsToolkit.Gisaxs.Configuration;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;

public static class RequestHandlerFactory 
{
    public static IRequestHandler CreateMajordomoRequestHandler(IOptionsMonitor<ConnectionStrings> connectionStrings)
    {
        return new MajordomoRequestHandler(connectionStrings);
    }
}