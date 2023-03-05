using ParallelGisaxsToolkit.Gisaxs.Utility.Images;
using Image = ParallelGisaxsToolkit.Gisaxs.Utility.Images.Image;

namespace ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;

public interface IImageStore
{
    Task<double[]> GetVerticalProfile(int id, int startY, int endY, int startX);
    Task<double[]> GetHorizontalProfile(int id, int startX, int endX, int startY);
    Task Insert(Image image);
    Task Insert(IEnumerable<Image> images);
    Task<GreyScaleImage?> Get(long id);
    Task<IEnumerable<ImageInfoWithId>> Get();
    Task Delete(long id);
}