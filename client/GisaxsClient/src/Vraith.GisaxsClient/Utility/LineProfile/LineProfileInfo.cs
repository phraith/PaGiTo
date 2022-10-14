#nullable enable

namespace Vraith.GisaxsClient.Utility.LineProfile
{
    internal class LineProfileInfo
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
            return new Coordinate { X = StartRel.X * width, Y = StartRel.Y * height };
        }

        public Coordinate AbsoluteEnd(int width, int height)
        {
            return new Coordinate { X = EndRel.X * width, Y = EndRel.Y * height };
        }
    }
}
