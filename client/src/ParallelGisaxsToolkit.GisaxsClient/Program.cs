using System.Data;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authorization;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using FastEndpoints;
using FastEndpoints.Swagger;
using Microsoft.AspNetCore.Http.Json;
using Microsoft.Extensions.Options;
using Npgsql;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.Gisaxs.Utility.HashComputer;
using ParallelGisaxsToolkit.GisaxsClient.Configuration;
using ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;
using ParallelGisaxsToolkit.GisaxsClient.Hubs;
using StackExchange.Redis;


var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
if (LaunchConfig.LaunchMode == LaunchMode.Kubernetes)
{
    builder.Configuration.SetBasePath("/vault/secrets/").AddJsonFile("appsettings.json", false, true);
}

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerDoc();
builder.Services.AddFastEndpoints();
builder.Services.AddSignalR();

builder.Services.Configure<ConnectionStrings>(builder.Configuration.GetSection("ConnectionStrings"));
builder.Services.Configure<AuthConfig>(builder.Configuration.GetSection("AuthOptions"));

builder.Services.AddScoped<IImageStore, ImageStore>();
builder.Services.AddScoped<IJobStore, JobStore>();
builder.Services.AddScoped<IRequestFactory, RequestFactory>();

builder.Services.AddSingleton<IHashComputer, Sha256HashComputer>();
builder.Services.AddSingleton<IRequestHandler, MajordomoRequestHandler>();
builder.Services.AddSingleton<IJobScheduler, JobScheduler>();
builder.Services.AddSingleton<IDatabase>(provider =>
{
    var connectionStrings = provider.GetService<IOptionsMonitor<ConnectionStrings>>();
    if (connectionStrings == null)
    {
        throw new InvalidOperationException("ConnectionStrings do not exist!");
    }
    IConnectionMultiplexer multiplexer = ConnectionMultiplexer.Connect(connectionStrings.CurrentValue.Redis);
    return multiplexer.GetDatabase();
});

builder.Services.AddScoped<IDbConnection>(provider =>
{
    var connectionStrings = provider.GetService<IOptionsMonitor<ConnectionStrings>>();
    if (connectionStrings == null)
    {
        throw new InvalidOperationException("ConnectionStrings do not exist!");
    }    
    IDbConnection connection = new NpgsqlConnection(connectionStrings.CurrentValue.Default);
    return connection;
});

builder.Services.AddAuthorization(auth =>
{
    auth.AddPolicy("Bearer", new AuthorizationPolicyBuilder()
        .AddAuthenticationSchemes(JwtBearerDefaults.AuthenticationScheme)
        .RequireAuthenticatedUser().Build());
});

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme).AddJwtBearer(options =>
{
    string? authOptionsToken = builder.Configuration.GetSection("AuthOptions:Token").Value;
    if (authOptionsToken == null)
    {
        throw new ArgumentNullException(nameof(authOptionsToken));
    }

    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuerSigningKey = true,
        IssuerSigningKey =
            new SymmetricSecurityKey(
                Encoding.UTF8.GetBytes(authOptionsToken)),
        ValidateIssuer = false,
        ValidateAudience = false
    };

    options.Events = new JwtBearerEvents
    {
        OnMessageReceived = context =>
        {
            string? accessToken = context.Request.Query["access_token"];
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

// builder.Services.AddSpaStaticFiles(configuration => { configuration.RootPath = "gisaxs-client-app/dist"; });
builder.Services.Configure<JsonOptions>(options => { options.SerializerOptions.PropertyNameCaseInsensitive = true; });


var app = builder.Build();
app.UseDefaultExceptionHandler();
// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseWebSockets();
//app.UseHttpsRedirection();
app.UseRouting();
app.UseAuthentication();
app.UseAuthorization();
app.UseFastEndpoints();
app.UseOpenApi();

if (app.Environment.IsDevelopment())
{
    app.UseSwaggerGen();
}

app.MapHub<MessageHub>("/message");
app.Run();