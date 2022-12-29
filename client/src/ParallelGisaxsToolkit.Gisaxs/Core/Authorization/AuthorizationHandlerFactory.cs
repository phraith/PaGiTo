using Microsoft.Extensions.Options;
using ParallelGisaxsToolkit.Gisaxs.Configuration;

namespace ParallelGisaxsToolkit.Gisaxs.Core.Authorization;

public static class AuthorizationHandlerFactory {
    public static IAuthorizationHandler CreateDefaultAuthorizationHandler(IOptionsMonitor<AuthConfig> authOptions)
    {
        return new AuthorizationHandler(authOptions);
    }
}