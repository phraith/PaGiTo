using Spectre.Cli;

using ParallelGisaxsToolkit.ImageStoreClient.CommandLineInterface.Commands;

CommandApp app = new CommandApp();
app.Configure(config =>
{
    config.CaseSensitivity(CaseSensitivity.None);
    config.SetApplicationName("ParallelGisaxsToolkit.ImageStoreClient");

    config.AddBranch("store", store =>
    {
        store.SetDescription("Store utility.");

        store.AddCommand<StorePushCommand>("push")
        .WithDescription("Pushes an image into a image store");
        
    });
});
return app.Run(args);