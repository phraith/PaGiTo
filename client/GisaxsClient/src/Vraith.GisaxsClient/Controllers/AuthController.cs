using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using Vraith.Gisaxs.Configuration;
using Vraith.Gisaxs.Core.Authorization;
using Vraith.Gisaxs.Core.UserStore;

namespace Vraith.GisaxsClient.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly UserStore _userStore;
        private readonly IAuthorizationHandler _authorizationHandler;

        public AuthController(IOptionsMonitor<ConnectionStrings> connectionStrings,
            IOptionsMonitor<AuthConfig> authOptions)
        {
            _userStore = new UserStore(connectionStrings.CurrentValue.Default);
            _authorizationHandler = AuthorizationHandlerFactory.CreateDefaultAuthorizationHandler(authOptions);
        }

        [HttpPost("register")]
        public async Task<ActionResult<User>> Register(UserDto request)
        {
            (long userId, byte[] passwordHash, byte[] passwordSalt) =
                _authorizationHandler.CreatePasswordHash(request.Password, request.Username);

            IEnumerable<User> users = await _userStore.Get();
            if (users.Any(u => u.UserId == userId))
            {
                return BadRequest();
            }

            User user = new(userId, passwordHash, passwordSalt);
            _userStore.Insert(user);
            return Ok();
        }

        [HttpPost("login")]
        public async Task<ActionResult<string>> Login(UserDto request)
        {
            IEnumerable<User> users = await _userStore.Get();
            User[] matchingUsers =
                users.Where(u => u.UserId == _authorizationHandler.CreateUserId(request.Username)).ToArray();
            if (matchingUsers.Length != 1)
            {
                return BadRequest();
            }

            User matchingUser = matchingUsers[0];
            if (!_authorizationHandler.VerifyPasswordHash(matchingUser, request.Password))
            {
                return BadRequest();
            }

            AuthInfo authInfo = _authorizationHandler.CreateJwtToken(matchingUser);
            return Ok(authInfo);
        }
    }
}