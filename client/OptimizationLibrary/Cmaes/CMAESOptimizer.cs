using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OptimizationLibrary.Cmaes
{

    public record Solution()
    {
        public Vector<double>? Parameters { get; init; }
        public double Fitness { get; init; }

    }

    public class CMAESOptimizer
    {
        private readonly CMA cma;
        private readonly Func<double[], double> function;
        private readonly int maxIteration;


        public double[] ResultVector { get; private set; }

        public double ResultValue { get; private set; }

        public CMAESOptimizer(Func<double[], double> function, double[] initial, double sigma, int randSeed = 0)
        {
            this.function = function;
            maxIteration = initial.Length * 200;

            cma = new CMA(initial, sigma, seed: randSeed);

            ResultValue = double.MaxValue;
        }


        public CMAESOptimizer(Func<double[], double> function, double[] initial, double sigma, double[] lowerBounds, double[] upperBounds, int randSeed = 0)
        {
            if (initial.Length != lowerBounds.Length)
            {
                throw new ArgumentException("Length of lowerBounds must be equal to that of initial.");
            }
            if (initial.Length != upperBounds.Length)
            {
                throw new ArgumentException("Length of upperBounds must be equal to that of initial");
            }

            this.function = function;
            maxIteration = initial.Length * 2000;

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
                for (int i = 0; i < cma.PopulationSize; i++)
                {
                    Vector<double> parameterVector = cma.Ask();
                    double fitness = function(parameterVector.AsArray());
                    solutions.Add(new Solution { Parameters = parameterVector, Fitness = fitness });
                }

                cma.Tell(solutions);
                if (cma.ShouldStop())
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
