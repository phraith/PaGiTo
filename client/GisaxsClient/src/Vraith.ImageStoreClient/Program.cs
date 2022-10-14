using Spectre.Cli;
using Vraith.ImageStoreClient.CommandLineInterface.Commands;

var app = new CommandApp();
app.Configure(config =>
{
    config.CaseSensitivity(CaseSensitivity.None);
    config.SetApplicationName("Vraith.ImageStoreClient");

    config.AddBranch("store", store =>
    {
        store.SetDescription("Store utility.");

        store.AddCommand<StorePushCommand>("push")
        .WithDescription("Pushes an image into a image store");
    });
});
return app.Run(args);