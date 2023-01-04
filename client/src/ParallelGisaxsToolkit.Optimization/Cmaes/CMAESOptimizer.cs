using MathNet.Numerics.LinearAlgebra;

namespace ParallelGisaxsToolkit.Optimization.Cmaes
{

    public record Solution()
    {
        public Vector<double>? Parameters { get; init; }
        public double Fitness { get; init; }

    }

    public class CmaesOptimizer
    {
        private readonly Cma _cma;
        private readonly Func<double[], double> _function;
        private readonly int _maxIteration;


        public double[] ResultVector { get; private set; }

        public double ResultValue { get; private set; }

        public CmaesOptimizer(Func<double[], double> function, double[] initial, double sigma, int randSeed = 0)
        {
            _function = function;
            _maxIteration = initial.Length * 200;

            _cma = new Cma(initial, sigma, seed: randSeed);

            ResultValue = double.MaxValue;
        }


        public CmaesOptimizer(Func<double[], double> function, double[] initial, double sigma, double[] lowerBounds, double[] upperBounds, int randSeed = 0)
        {
            if (initial.Length != lowerBounds.Length)
            {
                throw new ArgumentException("Length of lowerBounds must be equal to that of initial.");
            }
            if (initial.Length != upperBounds.Length)
            {
                throw new ArgumentException("Length of upperBounds must be equal to that of initial");
            }

            _function = function;
            _maxIteration = initial.Length * 2000;

            Matrix<double> bounds = Matrix<double>.Build.Dense(initial.Length, 2);
            bounds.SetColumn(0, lowerBounds.ToArray());
            bounds.SetColumn(1, upperBounds.ToArray());



            ResultValue = double.MaxValue;
        }


        public void Optimize()
        {
            while (true)
            {
                var solutions = new List<Solution>();
                for (int i = 0; i < _cma.PopulationSize; i++)
                {
                    Vector<double> parameterVector = _cma.Ask();
                    double fitness = _function(parameterVector.AsArray());
                    solutions.Add(new Solution { Parameters = parameterVector, Fitness = fitness });
                }

                _cma.Tell(solutions);
                if (_cma.ShouldStop())
                {
                    foreach (var solution in solutions)
                    {
                        Console.WriteLine($"{solution.Fitness} | {string.Join(',', solution.Parameters)}");
                    }
                    break;
                }
            }
        }

    }
}
