using Microsoft.AspNetCore.Mvc.Testing;

namespace Vraith.GisaxsClient.Tests;

[TestClass]
public class UnitTest1
{
    private static HttpClient _client;

    [ClassInitialize]
    public static void Initialize(TestContext testContext)
    {
        var application = new WebApplicationFactory<Program>();
            // .WithWebHostBuilder(builder =>
            // {
            //     builder.ConfigureServices(services =>
            //     {
            //         services.AddControllersWithViews();
            //     });
            // });
        _client = application.CreateClient();
    }
        
    [TestMethod]
    public void TestMethod1()
    {
        var res = _client.GetAsync("/api/jobstore/info").GetAwaiter().GetResult();
        var f = 5;
    }
}
