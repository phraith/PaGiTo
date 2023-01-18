using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.UserStore;
using IAuthorizationHandler = ParallelGisaxsToolkit.Gisaxs.Core.Authorization.IAuthorizationHandler;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Authorization;

[AllowAnonymous]
[HttpPost("/api/auth/register")]
public class RegisterEndpoint : Endpoint<RegisterRequest, RegisterResponse>
{
    private readonly IUserStore _userStore;
    private readonly IAuthorizationHandler _authorizationHandler;

    public RegisterEndpoint(IAuthorizationHandler authorizationHandler, IUserStore userStore)
    {
        _authorizationHandler = authorizationHandler;
        _userStore = userStore;
    }

    public override async Task HandleAsync(RegisterRequest request, CancellationToken ct)
    {
        (long userId, byte[] passwordHash, byte[] passwordSalt) =
            _authorizationHandler.CreatePasswordHash(request.Password, request.Username);

        IEnumerable<User> users = await _userStore.Get();
        if (users.Any(u => u.UserId == userId))
        {
            throw new InvalidOperationException("User already exists!");
        }

        User user = new(userId, passwordSalt, passwordHash);
        await _userStore.Insert(user);
        await SendAsync(new RegisterResponse(user.UserId, passwordHash, passwordSalt), cancellation: ct);
    }
}

public record RegisterRequest(string Username, string Password)
{
    public RegisterRequest() : this(string.Empty, string.Empty)
    {
    }
}

public record RegisterResponse(long UserId, byte[] PasswordSalt, byte[] PasswordHash);