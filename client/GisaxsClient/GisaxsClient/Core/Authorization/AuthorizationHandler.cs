using GisaxsClient.Core.UserStore;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;

namespace GisaxsClient.Core.Authorization
{
    public class AuthorizationHandler : IAuthorizationHandler
    {
        private readonly HMACSHA512 userIdGenerator;
        private readonly IOptionsMonitor<AuthConfig> authOptions;

        public AuthorizationHandler(IOptionsMonitor<AuthConfig> authOptions)
        {
            userIdGenerator = new HMACSHA512(Encoding.UTF8.GetBytes("MySecretKey"));
            this.authOptions = authOptions;
        }

        public string CreateJwtToken(User user)
        {
            List<Claim> claims = new()
            {
                new Claim(ClaimTypes.NameIdentifier, $"{user.UserId}")
            };

            var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(authOptions.CurrentValue.Token));
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

        public bool VerifyPasswordHash(User user, string password)
        {
            using var hmac = new HMACSHA512(user.PasswordSalt);
            var passwordHash = hmac.ComputeHash(Encoding.UTF8.GetBytes(password));
            return passwordHash.SequenceEqual(user.PasswordHash);
        }

        public (long userId, byte[] passwordHash, byte[] passwordSalt) CreatePasswordHash(string password, string username)
        {
            using var hmac = new HMACSHA512();
            return (CreateUserId(username), hmac.ComputeHash(Encoding.UTF8.GetBytes(password)), hmac.Key);
        }

        public long CreateUserId(string username)
        {
            var hash = userIdGenerator.ComputeHash(Encoding.UTF8.GetBytes(username));
            return BitConverter.ToInt64(hash);
        }
    }
}
