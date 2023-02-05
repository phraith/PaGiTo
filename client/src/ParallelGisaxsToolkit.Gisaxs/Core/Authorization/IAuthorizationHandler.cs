using ParallelGisaxsToolkit.Gisaxs.Core.UserStore;

namespace ParallelGisaxsToolkit.Gisaxs.Core.Authorization
{
    public interface IAuthorizationHandler
    {
        User CreateUser(string username, string password);
        string CreateJwtToken(User user);
        bool VerifyPassword(User user, string password);
        long CreateUserId(string username);
    }
}
