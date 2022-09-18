using ImageStoreClient.CommandLineInterface.Commands;
using Spectre.Cli;

var app = new CommandApp();
app.Configure(config =>
{
    config.CaseSensitivity(CaseSensitivity.None);
    config.SetApplicationName("ImageStoreClient");

    config.AddBranch("store", store =>
    {
        store.SetDescription("Store utility.");

        store.AddCommand<StorePushCommand>("push")
        .WithDescription("Pushes an image into a image store");
    });
});
return app.Run(args);