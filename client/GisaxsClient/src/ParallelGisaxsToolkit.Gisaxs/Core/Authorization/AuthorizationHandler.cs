using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Tokens;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.UserStore;

namespace ParallelGisaxsToolkit.Gisaxs.Core.Authorization
{
    internal class AuthorizationHandler : IAuthorizationHandler
    {
        private readonly HMACSHA512 _userIdGenerator;
        private readonly IOptionsMonitor<AuthConfig> _authOptions;

        public AuthorizationHandler(IOptionsMonitor<AuthConfig> authOptions)
        {
            _userIdGenerator = new HMACSHA512("MySecretKey"u8.ToArray());
            this._authOptions = authOptions;
        }

        public AuthInfo CreateJwtToken(User user)
        {
            List<Claim> claims = new()
            {
                new Claim(ClaimTypes.NameIdentifier, $"{user.UserId}")
            };

            SymmetricSecurityKey key = new(Encoding.UTF8.GetBytes(_authOptions.CurrentValue.Token));
            SigningCredentials cred = new(key, SecurityAlgorithms.HmacSha512Signature);

            JwtSecurityToken token = new            (
                claims: claims,
                signingCredentials: cred,
                expires: DateTime.Now.AddDays(1)
            );

            string jwt = new JwtSecurityTokenHandler().WriteToken(token);
            return new AuthInfo(jwt);
        }

        public bool VerifyPasswordHash(User user, string password)
        {
            using HMACSHA512 hmac = new HMACSHA512(user.PasswordSalt.ToArray());
            byte[] passwordHash = hmac.ComputeHash(Encoding.UTF8.GetBytes(password));
            return passwordHash.SequenceEqual(user.PasswordHash);
        }

        public (long userId, byte[] passwordHash, byte[] passwordSalt) CreatePasswordHash(string password, string username)
        {
            using HMACSHA512 hmac = new HMACSHA512();
            return (CreateUserId(username), hmac.ComputeHash(Encoding.UTF8.GetBytes(password)), hmac.Key);
        }

        public long CreateUserId(string username)
        {
            var hash = _userIdGenerator.ComputeHash(Encoding.UTF8.GetBytes(username));
            return BitConverter.ToInt64(hash);
        }
    }
}
