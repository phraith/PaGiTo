using Microsoft.Extensions.Options;
using Vraith.Gisaxs.Configuration;
using Vraith.Gisaxs.Core.ImageStore;

namespace Vraith.Gisaxs.Core.RequestHandling;

public static class RequestHandlerFactory 
{
    public static IRequestHandler CreateMajordomoRequestHandler(IOptionsMonitor<ConnectionStrings> connectionStrings)
    {
        return new MajordomoRequestHandler(connectionStrings);
    }
}