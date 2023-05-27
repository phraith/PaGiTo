using ParallelGisaxsToolkit.Optimization.FrechetDistance;

namespace ParallelGisaxsToolkit.Utilities.Tests;

[TestClass]
public class FrechetDistanceTests
{
    [TestMethod]
    public void EqualLinesHaveZeroDistance()
    {
        List<Point> lineA = new List<Point> { new(1, 1), new(2, 1), new(3, 1), new(4, 1)};
        double distance = FrechetDistance.Calculate(lineA, lineA);
        Assert.AreEqual(0.0, distance);
    }
    
    [TestMethod]
    public void ConstantOffsettedLinesHaveOffsetAsDistance()
    {
        List<Point> lineA = new List<Point> { new(1, 1), new(2, 1), new(3, 1), new(4, 1)};
        List<Point> lineB = new List<Point> { new(1, 2), new(2, 2), new(3, 2), new(4, 2)};
        double distance = FrechetDistance.Calculate(lineA, lineB);
        Assert.AreEqual(1.0, distance);
    }
    
    [TestMethod]
    public void ConstantOffsettedLinesHaveOffsetAsDistance2()
    {
        List<Point> lineA = new List<Point> { new(1, 1), new(2, 1), new(3, 1), new(4, 1)};
        List<Point> lineB = new List<Point> { new(1, 3), new(2, 3), new(3, 3), new(4, 3)};
        double distance = FrechetDistance.Calculate(lineA, lineB);
        Assert.AreEqual(2.0, distance);
    }
}