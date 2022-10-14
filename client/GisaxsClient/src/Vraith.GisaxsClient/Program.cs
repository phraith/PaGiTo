using Dapper;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authorization;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using Vraith.GisaxsClient.Configuration;
using Vraith.GisaxsClient.Controllers;
using Vraith.GisaxsClient.Hubs;


var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
if (LaunchConfig.LaunchMode == LaunchMode.Kubernetes)
{
    builder.WebHost.ConfigureAppConfiguration(webBuilder => 
    {
        webBuilder.SetBasePath("/vault/secrets/").AddJsonFile("appsettings.json", false, true);
    });
}

builder.Services.AddControllersWithViews();

builder.Services.AddSignalR();

builder.Services.Configure<ConnectionStrings>(builder.Configuration.GetSection("ConnectionStrings"));
builder.Services.Configure<AuthConfig>(builder.Configuration.GetSection("AuthOptions"));

builder.Services.AddAuthorization(auth =>
{
    auth.AddPolicy("Bearer", new AuthorizationPolicyBuilder()
        .AddAuthenticationSchemes(JwtBearerDefaults.AuthenticationScheme)
        .RequireAuthenticatedUser().Build());
});

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme).AddJwtBearer(options =>
{
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuerSigningKey = true,
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(builder.Configuration.GetSection("AuthOptions:Token").Value)),
        ValidateIssuer = false,
        ValidateAudience = false
    };

    options.Events = new JwtBearerEvents
    {
        OnMessageReceived = context =>
        {

            string accessToken = context.Request.Query["access_token"];
            // If the request is for our hub...
            var path = context.HttpContext.Request.Path;
            if (!string.IsNullOrEmpty(accessToken) &&
                path.StartsWithSegments("/message"))
            {
                // Read the token out of the query string
                context.Token = accessToken;
            }
            return Task.CompletedTask;
        }
    };
});

builder.Services.AddSpaStaticFiles(configuration =>
{
    configuration.RootPath = "gisaxs-client-app/dist";
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseWebSockets();
//app.UseHttpsRedirection();
//app.UseSpa(configuration => {  });
app.UseRouting();
app.UseAuthentication();
app.UseAuthorization();

app.UseSpaStaticFiles();
app.UseSpa(spa =>
{
    spa.Options.SourcePath = "gisaxs-client-app";
});

app.UseEndpoints(endpoints =>
{
    endpoints.MapControllerRoute(
        name: "default",
        pattern: "{controller}/{action=Index}/{id?}");
});
app.MapHub<MessageHub>("/message");
app.Run();
