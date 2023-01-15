﻿using System.Data;
using System.Text.Json;
using Dapper;
using ParallelGisaxsToolkit.Gisaxs.Utility.Database;
using ParallelGisaxsToolkit.Gisaxs.Utility.Images;

namespace ParallelGisaxsToolkit.Gisaxs.Core.ImageStore
{
    public class ImageStore : IImageStore
    {
        private readonly IDbConnection _connection;

        public ImageStore(IDbConnection connection)
        {
            _connection = connection;
            connection.Execute(
                @"CREATE TABLE IF NOT EXISTS images (
                    Id INT NOT NULL PRIMARY KEY GENERATED BY DEFAULT AS IDENTITY,
                    Info JSONB NOT NULL,
                    RowWiseData double precision[] NOT NULL,
                    ColumnWiseData double precision[] NOT NULL,
                    GreyScaleData bytea NOT NULL ); "
            );
        }

        public async Task<IEnumerable<ImageInfoWithId>> Get()
        {
            return await _connection.QueryAsync(@"SELECT info, id FROM images",
                (string info, int id) => new ImageInfoWithId(id, JsonSerializer.Deserialize<Utility.Images.ImageInfo>(info)!));
        }

        public async Task<GreyScaleImage?> Get(long id)
        {
            IEnumerable<GreyScaleImage>? images = await _connection.QueryAsync(
                @$"SELECT info, greyScaleData as greyScaleDataId FROM images WHERE id = {id}",
                (string info, byte[] greyScaleData) =>
                    new GreyScaleImage(JsonSerializer.Deserialize<Utility.Images.ImageInfo>(info)!, greyScaleData),
                splitOn: "greyScaleDataId");
            return images.FirstOrDefault();
        }

        public async Task Delete(long id)
        {
            await _connection.ExecuteAsync($@"DELETE * FROM images WHERE id = {id}");
        }

        public async Task Insert(Image image)
        {
            await _connection.ExecuteAsync($@"
                    INSERT INTO images (info, rowWiseData, columnWiseData, greyScaleData)
                    VALUES (@info, @rowWiseData, @columnWiseData, @greyScaleData)",
                new
                {
                    info = new JsonParameter(JsonSerializer.Serialize(image.Info)), rowWiseData = image.RowWiseData,
                    columnWiseData = image.ColumnWiseData, greyScaleData = image.GreyScaleData
                });
        }

        public async Task Insert(IEnumerable<Image> images)
        {
            using IDbTransaction transaction = _connection.BeginTransaction();
            foreach (var image in images)
            {
                await _connection.ExecuteAsync($@"
                        INSERT INTO images (info, rowWiseData, columnWiseData, greyScaleData)
                    VALUES (@info, @rowWiseData, @columnWiseData, @greyScaleData)",
                    new
                    {
                        info = new JsonParameter(JsonSerializer.Serialize(image.Info)), rowWiseData = image.RowWiseData,
                        columnWiseData = image.ColumnWiseData
                    },
                    transaction);
            }

            transaction.Commit();
        }

        public async Task<double[]> GetHorizontalProfile(int id, int startX, int endX, int startY)
        {
            int width = endX - startX;
            int start = width * startY;
            int end = start + width - 1;
            IEnumerable<double[]> dataSliceEnumerator = await _connection.QueryAsync<double[]>(
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
            IEnumerable<double[]>? dataSliceEnumerator = await _connection.QueryAsync<double[]>(
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