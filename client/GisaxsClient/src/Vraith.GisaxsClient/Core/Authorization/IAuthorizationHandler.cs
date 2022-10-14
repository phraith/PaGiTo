using Vraith.GisaxsClient.Core.UserStore;

namespace Vraith.GisaxsClient.Core.Authorization
{
    public interface IAuthorizationHandler
    {
        (long userId, byte[] passwordHash, byte[] passwordSalt) CreatePasswordHash(string password, string username);
        AuthInfo CreateJwtToken(User user);
        bool VerifyPasswordHash(User user, string password);
        long CreateUserId(string username);
    }
}
