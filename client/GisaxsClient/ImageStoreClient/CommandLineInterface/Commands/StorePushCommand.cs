using ImageStoreClient.ImageUtility;
using ImageStoreClient.ImageUtility.ImageLoaders;
using Spectre.Cli;
using System.ComponentModel;
using System.Text;
using System.Text.Json;

namespace ImageStoreClient.CommandLineInterface.Commands
{
    internal sealed class StorePushCommand : Command<StorePushCommand.Settings>
    {
        public sealed class Settings : CommandSettings
        {
            [Description("Path to image data")]
            [CommandArgument(0, "<imageDataPath>")]
            public string? ImageDataPath { get; init; }

            [Description("Image store url")]
            [CommandArgument(0, "<imageStoreUrl>")]
            public string? ImageStoreUrl { get; init; }
        }

        public override int Execute(CommandContext context, Settings settings)
        {
            if (settings.ImageDataPath == null || settings.ImageStoreUrl == null) { return 1; }

            var extension = Path.GetExtension(settings.ImageDataPath);
            Image image = extension switch
            {
                ".txt" => new AsciiLoader().Load(settings.ImageDataPath),
                ".tif" => new TifLoader().Load(settings.ImageDataPath), 
                _ => throw new NotImplementedException(),
            };

            HttpClient client = new();
            var body = new StringContent(JsonSerializer.Serialize(image), Encoding.UTF8, "application/json");
            Console.WriteLine($"{settings.ImageStoreUrl}");
            var result = client.PostAsync(settings.ImageStoreUrl, body).Result;
            return 0;
        }
    }
}
