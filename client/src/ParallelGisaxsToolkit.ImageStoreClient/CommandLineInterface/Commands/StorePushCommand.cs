﻿using System.ComponentModel;
using System.Text;
using System.Text.Json;
using Spectre.Cli;
using Spectre.Console;
using ParallelGisaxsToolkit.Gisaxs.Utility.Images;
using ParallelGisaxsToolkit.Gisaxs.Utility.Images.ImageLoaders;

namespace ParallelGisaxsToolkit.ImageStoreClient.CommandLineInterface.Commands
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
            if (settings.ImageDataPath == null || settings.ImageStoreUrl == null)
            {
                return 1;
            }

            string extension = Path.GetExtension(settings.ImageDataPath);
            Image image = extension switch
            {
                ".txt" => new AsciiLoader().Load(settings.ImageDataPath),
                ".tif" => new TifLoader().Load(settings.ImageDataPath),
                _ => throw new NotImplementedException(),
            };

            HttpClient client = new();
            string data = @$"{{""image"": {JsonSerializer.Serialize(image)}}}";

            StringContent body = new(data, Encoding.UTF8, "application/json");
            Console.WriteLine($"{settings.ImageStoreUrl}");
            HttpResponseMessage result = client.PostAsync($"{settings.ImageStoreUrl}/api/image", body).Result;
            AnsiConsole.WriteLine(result.ToString());
            AnsiConsole.WriteLine(StatusMessage(result.IsSuccessStatusCode));
            return 0;
        }

        private static string StatusMessage(bool requestWasSuccessful)
        {
            if (requestWasSuccessful)
            {
                return $"Request was successfull!";
            }

            return $"Request failed!";
        }
    }
}