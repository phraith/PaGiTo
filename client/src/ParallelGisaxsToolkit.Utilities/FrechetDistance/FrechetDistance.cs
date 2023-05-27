namespace ParallelGisaxsToolkit.Optimization.FrechetDistance;

public record Point(double X, double Y);

public static class FrechetDistance
{
    public static double Calculate(IReadOnlyList<Point> lineA, IReadOnlyList<Point> lineB)
    {
        var freeSpaceDiagram = FreeSpaceDiagram(lineA, lineB);
        return freeSpaceDiagram[lineA.Count - 1, lineB.Count - 1];
    }

    private static double[,] FreeSpaceDiagram(IReadOnlyList<Point> lineA, IReadOnlyList<Point> lineB)
    {
        double[,] freeSpaceDiagram = new double[lineA.Count, lineB.Count];
        for (int i = 0; i < lineA.Count; ++i)
        {
            Point pointA = lineA[i];
            for (int j = 0; j < lineB.Count; ++j)
            {
                Point pointB = lineB[j];
                double distance = Math.Sqrt(Math.Pow(pointB.X - pointA.X, 2) + Math.Pow(pointB.Y - pointA.Y, 2));

                if (i > 0 && j > 0)
                {
                    freeSpaceDiagram[i, j] =
                        Math.Max(Math.Min(Math.Min(freeSpaceDiagram[i - 1, j], freeSpaceDiagram[i - 1, j - 1]),
                            freeSpaceDiagram[i, j - 1]), distance);
                }
                else if (i > 0 && j == 0)
                {
                    freeSpaceDiagram[i, j] = Math.Max(freeSpaceDiagram[i - 1, 0], distance);
                }
                else if (i == 0 && j > 0)
                {
                    freeSpaceDiagram[i, j] = Math.Max(freeSpaceDiagram[0, j - 1], distance);
                }
                else if (i == 0 && j == 0)
                {
                    freeSpaceDiagram[i, j] = distance;
                }
                else
                {
                    freeSpaceDiagram[i, j] = double.MaxValue;
                }
            }
        }

        return freeSpaceDiagram;
    }
}