using Microsoft.Extensions.Options;
using Vraith.Gisaxs.Configuration;

namespace Vraith.Gisaxs.Core.Authorization;

public static class AuthorizationHandlerFactory {
    public static IAuthorizationHandler CreateDefaultAuthorizationHandler(IOptionsMonitor<AuthConfig> authOptions)
    {
        return new AuthorizationHandler(authOptions);
    }
}