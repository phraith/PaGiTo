using ParallelGisaxsToolkit.Gisaxs.Utility.Images;
using ParallelGisaxsToolkit.Gisaxs.Utility.Images.ImageLoaders;

namespace ParallelGisaxsToolkit.ImageStoreClient.Tests
{
    [TestClass]
    public class ImageStoreClientTests
    {
        [TestMethod]
        public void CanLoadTifImage()
        {
            Image image = new TifLoader().Load(@"../../../../test-assets/ImageStoreClient/test.tif");
            Assert.AreEqual("test", image.Info.Name);
            Assert.AreEqual(100, image.Info.Width);
            Assert.AreEqual(100, image.Info.Height);
            Assert.AreEqual(100 * 100 * sizeof(double), image.Info.Size);
        }
    }
}