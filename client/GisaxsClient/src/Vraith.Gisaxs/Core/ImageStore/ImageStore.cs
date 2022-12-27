﻿#nullable enable

using System.Data;
using System.Text.Json;
using Dapper;
using Microsoft.Extensions.Configuration;
using Npgsql;
using Vraith.Gisaxs.Utility.Database;
using Vraith.Gisaxs.Utility.Images;

namespace Vraith.Gisaxs.Core.ImageStore
{
    public class ImageStore
    {
        private readonly string _connectionString;

        public ImageStore(IConfiguration configuration, string connectionId = "Default")
        {
            _connectionString = configuration.GetConnectionString(connectionId) ??
                                throw new ArgumentNullException(nameof(connectionId));
            using IDbConnection connection = new NpgsqlConnection(_connectionString);
            connection.Execute(
                @"CREATE TABLE IF NOT EXISTS images (
                    Id INT NOT NULL PRIMARY KEY GENERATED BY DEFAULT AS IDENTITY,
                    Info JSONB NOT NULL,
                    RowWiseData double precision[] NOT NULL,
                    ColumnWiseData double precision[] NOT NULL); "
            );
        }

        public async Task<IEnumerable<ImageInfoDto>> Get()
        {
            using IDbConnection connection = new NpgsqlConnection(_connectionString);
            return await connection.QueryAsync(@"SELECT info, id FROM images",
                (string info, int id) => new ImageInfoDto(id, JsonSerializer.Deserialize<ImageInfo>(info)!));
        }

        public async Task<Image?> Get(long id)
        {
            using IDbConnection connection = new NpgsqlConnection(_connectionString);
            IEnumerable<Image>? images = await connection.QueryAsync(
                @$"SELECT info, rowWiseData as rowWiseDataId, columnWiseData as columnsWiseDataId FROM images WHERE id = {id}",
                (string info, double[] rowWiseData, double[] columnWiseData) =>
                    new Image(JsonSerializer.Deserialize<ImageInfo>(info)!, rowWiseData, columnWiseData),
                splitOn: "rowWiseDataId, columnsWiseDataId");
            return images.FirstOrDefault();
        }

        public async void Delete(long id)
        {
            using IDbConnection connection = new NpgsqlConnection(_connectionString);
            await connection.ExecuteAsync($@"DELETE * FROM images WHERE id = {id}");
        }

        public async void Insert(Image image)
        {
            using IDbConnection connection = new NpgsqlConnection(_connectionString);
            await connection.ExecuteAsync($@"
                    INSERT INTO images (info, rowWiseData, columnWiseData)
                    VALUES (@info, @rowWiseData, @columnWiseData)",
                new
                {
                    info = new JsonParameter(JsonSerializer.Serialize(image.Info)), rowWiseData = image.RowWiseData,
                    columnWiseData = image.ColumnWiseData
                });
        }

        public async void Insert(IReadOnlyCollection<Image> images)
        {
            using IDbConnection connection = new NpgsqlConnection(_connectionString);
            using IDbTransaction transaction = connection.BeginTransaction();
            foreach (var image in images)
            {
                await connection.ExecuteAsync($@"
                        INSERT INTO images (info, rowWiseData, columnWiseData)
                    VALUES (@info, @rowWiseData, @columnWiseData)",
                    new
                    {
                        info = new JsonParameter(JsonSerializer.Serialize(image.Info)), rowWiseData = image.RowWiseData,
                        columnWiseData = image.ColumnWiseData
                    },
                    transaction);
            }

            transaction.Commit();
        }

        public async Task<double[]> GetHorizonalProfile(int id, int startX, int endX, int startY)
        {
            int width = endX - startX;
            int start = width * startY;
            int end = start + width - 1;
            using IDbConnection connection = new NpgsqlConnection(_connectionString);
            IEnumerable<double[]> dataSliceEnumerator = await connection.QueryAsync<double[]>(
                @$"SELECT rowWiseData[{start}:{end}] FROM images WHERE id = {id}");

            if (dataSliceEnumerator == null)
            {
                return Array.Empty<double>();
            }

            var dataSlices = dataSliceEnumerator.ToArray();
            if (dataSlices.Length != 1 || dataSlices[0].Length != width)
            {
                return Array.Empty<double>();
            }

            return dataSlices[0];
        }

        public async Task<double[]> GetVerticalProfile(int id, int startY, int endY, int startX)
        {
            int height = startY - endY;
            int start = height * startX;
            int end = start + height - 1;
            using IDbConnection connection = new NpgsqlConnection(_connectionString);
            IEnumerable<double[]>? dataSliceEnumerator = await connection.QueryAsync<double[]>(
                @$"SELECT columnWiseData[{start}:{end}] FROM images WHERE id = {id}");

            if (dataSliceEnumerator == null)
            {
                return Array.Empty<double>();
            }

            var dataSlices = dataSliceEnumerator.ToArray();
            if (dataSlices.Length != 1 || dataSlices[0].Length != height)
            {
                return Array.Empty<double>();
            }

            return dataSlices[0];
        }
    }
}