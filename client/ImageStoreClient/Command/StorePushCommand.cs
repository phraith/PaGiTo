using Spectre.Cli;
using System.ComponentModel;
using System.Text;
using System.Text.Json;

namespace ImageStoreClient.Command
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
                ".txt" => LoadImageFromAscii(settings.ImageDataPath),
                _ => throw new NotImplementedException(),
            };

            HttpClient client = new();
            var body = new StringContent(JsonSerializer.Serialize(image), Encoding.UTF8, "application/json");
            var result = client.PostAsync(settings.ImageStoreUrl, body).Result;
            return 0;
        }

        private static Image LoadImageFromAscii(string imageDataPath)
        {
            var content =  File.ReadAllText(imageDataPath);
            var split = content.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            var data = split[3..];
            var height = data.Length;
            var width = data[0].Split("    ", StringSplitOptions.RemoveEmptyEntries).Length - 1;
            var imageData = new List<double>(width * height);

            foreach (var row in data)
            {
                var rowSplit = row.Split("    ", StringSplitOptions.RemoveEmptyEntries);
                var doubles = rowSplit[..^1].Select(x => double.Parse(x)).ToArray();
                imageData.AddRange(doubles);
            }

            ImageInfo info = new(-1, Path.GetFileNameWithoutExtension(imageDataPath), imageData.Count * sizeof(double));
            Image image = new()
            {
                Data = imageData.ToArray(),
                Height = height,
                Width = width,
                Name = info.Name,
                Id = info.Id,
                Size = info.Size
            };
            return image;
        }
    }
}
