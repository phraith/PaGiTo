using System.ComponentModel.DataAnnotations;
using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.UserStore;
using IAuthorizationHandler = ParallelGisaxsToolkit.Gisaxs.Core.Authorization.IAuthorizationHandler;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Authorization;

[AllowAnonymous]
[HttpPost("/api/auth/login")]
public class LoginEndpoint : Endpoint<LoginRequest, LoginResponse>
{
    private readonly IUserStore _userStore;
    private readonly IAuthorizationHandler _authorizationHandler;

    public LoginEndpoint(IUserStore userStore, IAuthorizationHandler authorizationHandler)
    {
        _userStore = userStore;
        _authorizationHandler = authorizationHandler;
    }

    public override async Task HandleAsync(LoginRequest request, CancellationToken ct)
    {
        IEnumerable<User> users = await _userStore.Get();
        User[] matchingUsers =
            users.Where(u => u.UserId == _authorizationHandler.CreateUserId(request.Username)).ToArray();
        if (matchingUsers.Length != 1)
        {
            throw new InvalidOperationException("A matching user cant be determined!");
        }

        User matchingUser = matchingUsers[0];
        if (!_authorizationHandler.VerifyPassword(matchingUser, request.Password))
        {
            throw new InvalidOperationException("Password is incorrect!");
        }

        string token = _authorizationHandler.CreateJwtToken(matchingUser);
        await SendAsync(new LoginResponse(token), cancellation: ct);
    }
}

public sealed record LoginRequest
{
    [Required] public string Username { get; init; } = string.Empty;
    [Required] public string Password { get; init; } = string.Empty;
}

public sealed record LoginResponse(string Token);