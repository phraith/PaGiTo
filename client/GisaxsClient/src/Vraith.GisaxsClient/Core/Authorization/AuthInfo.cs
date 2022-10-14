namespace Vraith.GisaxsClient.Core.Authorization
{
    public class AuthInfo
    {
        public string Token { get; }
        public AuthInfo(string token)
        {
            Token = token;
        }
    }
}
