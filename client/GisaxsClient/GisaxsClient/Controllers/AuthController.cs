using GisaxsClient.Core.Authorization;
using GisaxsClient.Core.UserStore;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;

namespace GisaxsClient.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly UserStore userStore;
        private readonly IAuthorizationHandler authorizationHandler;

        public AuthController(IOptionsMonitor<ConnectionStrings> connectionStrings, IOptionsMonitor<AuthConfig> authOptions)
        {
            userStore = new UserStore(connectionStrings.CurrentValue.Default);
            authorizationHandler = new AuthorizationHandler(authOptions);
        }

        [HttpPost("register")]
        public async Task<ActionResult<User>> Register(UserDto request)
        {
            (long userId, byte[] passwordHash, byte[] passwordSalt) = authorizationHandler.CreatePasswordHash(request.Password, request.Username);

            IEnumerable<User> users = await userStore.Get();
            if (users.Any(u => u.UserId == userId))
            {
                return BadRequest();
            }

            User user = new() { UserId = userId, PasswordHash = passwordHash, PasswordSalt = passwordSalt };
            userStore.Insert(user);
            return Ok();
        }

        [HttpPost("login")]
        public async Task<ActionResult<string>> Login(UserDto request)
        {
            IEnumerable<User> users = await userStore.Get();
            IEnumerable<User> matchingUsers = users.Where(u => u.UserId == authorizationHandler.CreateUserId(request.Username));
            if (matchingUsers.Count() != 1)
            {
                return BadRequest();
            }

            User matchingUser = matchingUsers.First();
            if (!authorizationHandler.VerifyPasswordHash(matchingUser, request.Password))
            {
                return BadRequest();
            }

            AuthInfo authInfo= authorizationHandler.CreateJwtToken(matchingUser);
            return Ok(authInfo);
        }
    }
}
