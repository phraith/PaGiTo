using GisaxsClient.Security;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;
using System;
using System.Collections.Generic;
using System.IdentityModel.Tokens.Jwt;
using System.Linq;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using UserDataProvider;

namespace GisaxsClient.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AuthController : ControllerBase
    {
        private readonly UserDataContext context;
        private readonly IConfiguration configuration;
        private readonly HMACSHA512 userIdGenerator;

        public AuthController(UserDataContext context, IConfiguration configuration)
        {
            this.context = context;
            this.configuration = configuration;
            this.userIdGenerator = new HMACSHA512(Encoding.UTF8.GetBytes("MySecretKey"));
        }

        [HttpPost("register")]
        public async Task<ActionResult<User>> Register(UserDto request)
        {
            (long userId, byte[] passwordHash, byte[] passwordSalt) = CreatePasswordHash(request.Password, request.Username);

            if (context.Users.Any(u => u.Id == userId))
            {
                return BadRequest();
            }

            var user = new User { Id = userId, PasswordHash = passwordHash, PasswordSalt = passwordSalt };
            await context.AddAsync(user);
            await context.SaveChangesAsync();
            return Ok();
        }
        
        [HttpPost("login")]
        public async Task<ActionResult<string>> Login(UserDto request)
        {
            var matchingUsers = context.Users.Where(u => u.Id == CreateUserId(request.Username));
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
                new Claim(ClaimTypes.NameIdentifier, $"{matchingUser.Id}")
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
