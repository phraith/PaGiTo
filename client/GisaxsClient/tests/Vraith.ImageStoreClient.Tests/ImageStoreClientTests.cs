using Vraith.Gisaxs.Utility.Images.ImageLoaders;

namespace Vraith.ImageStoreClient.Tests
{
    [TestClass]
    public class ImageStoreClientTests
    {
        [TestMethod]
        public void CanLoadTifImage()
        {
            var image = new TifLoader().Load(@"../../../../test-assets/ImageStoreClient/test.tif");
            Assert.AreEqual("test", image.Info.Name);
            Assert.AreEqual(100, image.Info.Width);
            Assert.AreEqual(100, image.Info.Height);
            Assert.AreEqual(100 * 100 * sizeof(double), image.Info.Size);
        }
    }
}