using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using ParallelGisaxsToolkit.Gisaxs.Core.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.UserStore;

namespace ParallelGisaxsToolkit.Gisaxs.Tests.Authorization;

[TestClass]
public class AuthorizationTests
{
    private static IAuthorizationHandler _handler = null!;

    [ClassInitialize]
    public static void Initialize(TestContext testContext)
    {
        _handler = new AuthorizationHandler("tokentokentokentoken", new HmacSha512UserIdGenerator());
    }
    
    [TestMethod]
    [DataTestMethod]
    [DataRow("test", "test")]
    [DataRow("", "")]
    [DataRow("$", "$")]
    public void CanAuthorize(string username, string password)
    {
        User user = _handler.CreateUser(username, password);
        bool isValid = _handler.VerifyPassword(user, password);
        Assert.IsTrue(isValid);
    }
    
    [TestMethod]
    [DataTestMethod]
    [DataRow("test", "test", "testt")]
    [DataRow("", "", "a")]
    [DataRow("$", "$", "$$")]
    public void CantAuthorize(string username, string password, string wrongPassword)
    {
        User user = _handler.CreateUser(username, password);
        bool isValid = _handler.VerifyPassword(user, wrongPassword);
        Assert.IsFalse(isValid);
    }

    [TestMethod]
    [DataTestMethod]
    [DataRow("test", "test")]
    [DataRow("", "")]
    [DataRow("$", "$")]
    public void CanCreateUserId(string username, string password)
    {
        User user = _handler.CreateUser(username, password);
        long userId = _handler.CreateUserId(username);
        Assert.AreEqual(user.UserId, userId);
    }

    [TestMethod]
    public void CanCreateJwt()
    {
        User user = _handler.CreateUser("test", "test");
        string jsonToken = _handler.CreateJwtToken(user);
        JwtSecurityTokenHandler handler = new JwtSecurityTokenHandler();
        JwtSecurityToken? token = handler.ReadToken(jsonToken) as JwtSecurityToken;
        string identifier = token!.Claims.First(c => c.Type == ClaimTypes.NameIdentifier).Value;
        Assert.AreEqual(user.UserId, long.Parse(identifier));
        
    }
}