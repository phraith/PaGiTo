using GisaxsClient.Core.UserStore;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;

namespace GisaxsClient.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AuthController : ControllerBase
    {
        private readonly IConfiguration configuration;
        private readonly HMACSHA512 userIdGenerator;
        private readonly UserStore userStore;

        public AuthController(IConfiguration configuration)
        {
            this.configuration = configuration;
            userIdGenerator = new HMACSHA512(Encoding.UTF8.GetBytes("MySecretKey"));
            userStore = new UserStore(configuration);
        }

        //[Authorize]
        [HttpPost("register")]
        public async Task<ActionResult<User>> Register(UserDto request)
        {
            (long userId, byte[] passwordHash, byte[] passwordSalt) = CreatePasswordHash(request.Password, request.Username);

            var users = await userStore.Get();
            if (users.Any(u => u.UserId == userId))
            {
                return BadRequest();
            }

            var user = new User { UserId = userId, PasswordHash = passwordHash, PasswordSalt = passwordSalt };
            userStore.Insert(user);
            return Ok();
        }

        [HttpPost("login")]
        public async Task<ActionResult<string>> Login(UserDto request)
        {
            var users = await userStore.Get();
            var matchingUsers = users.Where(u => u.UserId == CreateUserId(request.Username));
            if (matchingUsers.Count() != 1)
            {
                return BadRequest();
            }

            var matchingUser = matchingUsers.First();
            if (!VerifyPasswordHash(matchingUser, request.Password))
            {
                return BadRequest();
            }

            string token = CreateToken(matchingUser);
            return Ok(token);
        }

        private string CreateToken(User matchingUser)
        {
            List<Claim> claims = new()
            {
                new Claim(ClaimTypes.NameIdentifier, $"{matchingUser.UserId}")
            };

            var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(configuration.GetSection("AppSettings:Token").Value));
            var cred = new SigningCredentials(key, SecurityAlgorithms.HmacSha512Signature);

            var token = new JwtSecurityToken
            (
                claims: claims,
                signingCredentials: cred,
                expires: DateTime.Now.AddDays(1)
            );

            var jwt = new JwtSecurityTokenHandler().WriteToken(token);
            return jwt;
        }

        private static bool VerifyPasswordHash(User user, string password)
        {
            using var hmac = new HMACSHA512(user.PasswordSalt);
            var passwordHash = hmac.ComputeHash(Encoding.UTF8.GetBytes(password));
            return passwordHash.SequenceEqual(user.PasswordHash);
        }

        private long CreateUserId(string username)
        {
            var hash = userIdGenerator.ComputeHash(Encoding.UTF8.GetBytes(username));
            return BitConverter.ToInt64(hash);
        }

        private (long userId, byte[] passwordHash, byte[] passwordSalt) CreatePasswordHash(string password, string username)
        {
            using var hmac = new HMACSHA512();
            return (CreateUserId(username), hmac.ComputeHash(Encoding.UTF8.GetBytes(password)), hmac.Key);
        }
    }
}
