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
using ParallelGisaxsToolkit.Gisaxs.Core.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.Hubs;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.Gisaxs.Core.UserStore;
using ParallelGisaxsToolkit.Gisaxs.Utility.HashComputer;
using ParallelGisaxsToolkit.GisaxsClient;
using ParallelGisaxsToolkit.GisaxsClient.Configuration;
using Serilog;
using Serilog.Events;
using StackExchange.Redis;
using Steeltoe.Extensions.Configuration.Placeholder;
using Steeltoe.Extensions.Configuration.RandomValue;

Log.Logger = new LoggerConfiguration()
    .WriteTo.Console()
    .CreateBootstrapLogger();

try
{
    Log.Information("Starting web application");
    WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

    builder.Host.UseSerilog((context, services, configuration) => configuration
        .ReadFrom.Configuration(context.Configuration)
        .ReadFrom.Services(services)
        .Enrich.FromLogContext()
        .WriteTo.Console());

// Add services to the container.
    if (LaunchConfig.LaunchMode == LaunchMode.Kubernetes)
    {
        builder.Configuration.SetBasePath("/vault/secrets/").AddJsonFile("appsettings.json", false, true);
    }

    builder.Configuration.AddEnvironmentVariables("ConnectionStrings");
    builder.Configuration.AddRandomValueSource();
    builder.Configuration.AddPlaceholderResolver();
    builder.Services.AddEndpointsApiExplorer();
    builder.Services.AddSwaggerDoc();
    builder.Services.AddFastEndpoints();
    builder.Services.AddSignalR();

    builder.Services.Configure<ConnectionStrings>(builder.Configuration.GetSection("ConnectionStrings"));
    builder.Services.Configure<AuthConfig>(builder.Configuration.GetSection("AuthOptions"));

    builder.Services.AddScoped<IImageStore, ImageStore>();
    builder.Services.AddScoped<IJobStore, JobStore>();
    builder.Services.AddScoped<IRequestFactory, RequestFactory>();
    builder.Services.AddScoped<IUserStore, UserStore>();
    builder.Services.AddScoped<INotifier, MessageHubNotifier>();
    builder.Services.AddSingleton<IUserIdGenerator, HmacSha512UserIdGenerator>();
    builder.Services.AddSingleton<IHashComputer, Sha256HashComputer>();
    builder.Services.AddSingleton<IRequestResultDeserializer, RequestResultDeserializer>();

    builder.Services.AddHostedService<RabbitMqConsumer>();
    builder.Services.AddScoped<IRabbitMqPublisher, RabbitMqPublisher>();

    builder.Services.AddLogging(x =>
    {
        x.ClearProviders();
        x.AddSerilog(dispose: true);
    });

    builder.Services.AddSingleton<ParallelGisaxsToolkit.Gisaxs.Core.Authorization.IAuthorizationHandler>(provider =>
    {
        IOptionsMonitor<AuthConfig> authConfig = provider.GetRequiredService<IOptionsMonitor<AuthConfig>>();
        IUserIdGenerator userIdGenerator = provider.GetRequiredService<IUserIdGenerator>();
        return new AuthorizationHandler(authConfig.CurrentValue.Token, userIdGenerator);
    });

    builder.Services.AddSingleton<IRabbitMqFactory>(provider =>
    {
        IOptionsMonitor<ConnectionStrings>? connectionStrings =
            provider.GetRequiredService<IOptionsMonitor<ConnectionStrings>>();

        string rabbitMqConsumerQueueName = connectionStrings.CurrentValue.RabbitMqConsumerQueueName;
        Environment.SetEnvironmentVariable("RABBIT_MQ_CONSUMER_QUEUE_NAME", rabbitMqConsumerQueueName,
            EnvironmentVariableTarget.User);
        return new RabbitMqFactory(connectionStrings.CurrentValue.RabbitMqUser,
            connectionStrings.CurrentValue.RabbitMqPassword, connectionStrings.CurrentValue.RabbitMqHost,
            rabbitMqConsumerQueueName);
    });

    builder.Services.AddScoped<IDatabase>(provider =>
    {
        IOptionsMonitor<ConnectionStrings> connectionStrings =
            provider.GetRequiredService<IOptionsMonitor<ConnectionStrings>>();
        IConnectionMultiplexer multiplexer = ConnectionMultiplexer.Connect(connectionStrings.CurrentValue.Redis);
        return multiplexer.GetDatabase();
    });

    builder.Services.AddScoped<IDbConnection>(provider =>
    {
        IOptionsMonitor<ConnectionStrings>? connectionStrings =
            provider.GetRequiredService<IOptionsMonitor<ConnectionStrings>>();
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
        string authOptionsToken = builder.Configuration.GetRequiredSection("AuthOptions:Token").Value!;
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
                PathString path = context.HttpContext.Request.Path;
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

    builder.Services.Configure<JsonOptions>(
        options => { options.SerializerOptions.PropertyNameCaseInsensitive = true; });


    WebApplication app = builder.Build();

    app.UseSerilogRequestLogging(options =>
    {
        // Customize the message template
        options.MessageTemplate = "Handled {RequestPath}";

        // Emit debug-level events instead of the defaults
        options.GetLevel = (httpContext, elapsed, ex) => LogEventLevel.Debug;

        // Attach additional properties to the request completion event
        options.EnrichDiagnosticContext = (diagnosticContext, httpContext) =>
        {
            diagnosticContext.Set("RequestHost", httpContext.Request.Host.Value);
            diagnosticContext.Set("RequestScheme", httpContext.Request.Scheme);
        };
    });

    app.UseDefaultExceptionHandler();
// Configure the HTTP request pipeline.
    if (!app.Environment.IsDevelopment())
    {
        // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
        app.UseHsts();
    }

    app.UseWebSockets();
    app.UseStaticFiles();
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

    app.MapFallbackToFile("index.html");

    app.Run();
}
catch (Exception ex)
{
    Log.Fatal(ex, "Application terminated unexpectedly");
}
finally
{
    Log.CloseAndFlush();
}