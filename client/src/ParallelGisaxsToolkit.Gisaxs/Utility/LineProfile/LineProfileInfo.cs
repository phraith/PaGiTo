#nullable enable

namespace ParallelGisaxsToolkit.Gisaxs.Utility.LineProfile
{
    public class LineProfileInfo
    {
        public Coordinate EndRel { get; }
        public Coordinate StartRel { get; }

        public LineProfileInfo(Coordinate startRel, Coordinate endRel)
        {
            EndRel = endRel;
            StartRel = startRel;
        }


        public Coordinate AbsoluteStart(int width, int height)
        {
            return new Coordinate(StartRel.X * width, StartRel.Y * height);
        }

        public Coordinate AbsoluteEnd(int width, int height)
        {
            return new Coordinate(EndRel.X * width, EndRel.Y * height);
        }
    }
}