using System.ComponentModel.DataAnnotations;
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
        User user = _authorizationHandler.CreateUser(request.Username, request.Password);

        IEnumerable<User> users = await _userStore.Get();
        if (users.Any(u => u.UserId == user.UserId))
        {
            throw new InvalidOperationException("User already exists!");
        }

        await _userStore.Insert(user);
        RegisterResponse response = new RegisterResponse(user.UserId, user.PasswordHash, user.PasswordSalt);
        await SendAsync(response, cancellation: ct);
    }
}

public sealed record RegisterRequest
{
    [Required] public string Username { get; init; } = string.Empty;
    [Required] public string Password { get; init; } = string.Empty;
}

public sealed record RegisterResponse(long UserId, byte[] PasswordSalt, byte[] PasswordHash);