using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;

namespace ParallelGisaxsToolkit.Optimization.Cmaes
{
    public class Cma
    {
        private readonly int _mu;
        private readonly double _muEff;
        private readonly double _cc;
        private readonly double _c1;
        private readonly double _cmu;
        private readonly double _cSigma;
        private readonly double _dSigma;
        private readonly int _cm;
        private readonly double _chiN;
        private readonly Vector<double> _weights;
        private Vector<double> _pSigma;
        private Vector<double> _pc;
        private Vector<double> _mean;
        private Matrix<double> _c;
        private double _sigma;
        private Vector<double> _d;
        private Matrix<double> _b;
        private Matrix<double> _bounds;
        private readonly int _nMaxResampling;
        private readonly Normal _normalDistribution;
        private readonly double _epsilon;
        private readonly double _tolSigma;
        private readonly double _tolC;
        private readonly int _funhistTerm;

        private const double Tolxup = 1e4;
        private const double Tolfun = 1e-12;
        private const double Tolconditioncov = 1e14;

        private readonly double _tolx;

        private double[] _funhistValues;

        public int Dim { get; }

        public int PopulationSize { get; private set; }

        public int Generation { get; private set; }


        public Cma(IList<double> mean, double sigma, Matrix<double> bounds = null, int nMaxResampling = 100,
            int seed = 0, double tolSigma = 1e-4, double tolC = 1e-4)
        {
            if (!(sigma > 0))
            {
                throw new ArgumentOutOfRangeException("sigma must be non-zero positive value");
            }

            int nDim = mean.Count;
            if (!(nDim > 1))
            {
                throw new ArgumentOutOfRangeException("The dimension of mean must be larger than 1");
            }

            int populationSize = 4 + (int)Math.Floor(3 * Math.Log(nDim)); // # (eq. 48)

            int mu = populationSize / 2;

            Vector<double> weightsPrime = Vector<double>.Build.Dense(populationSize);
            for (int i = 0; i < populationSize; i++)
            {
                weightsPrime[i] = Math.Log((populationSize + 1) / (double)2) - Math.Log(i + 1);
            }

            Vector<double> weightsPrimeMuEff = Vector<double>.Build.Dense(weightsPrime.Take(mu).ToArray());
            double muEff = Math.Pow(weightsPrimeMuEff.Sum(), 2) / Math.Pow(weightsPrimeMuEff.L2Norm(), 2);
            Vector<double> weightsPrimeMuEffMinus = Vector<double>.Build.Dense(weightsPrime.Skip(mu).ToArray());
            double muEffMinus = Math.Pow(weightsPrimeMuEffMinus.Sum(), 2) /
                                Math.Pow(weightsPrimeMuEffMinus.L2Norm(), 2);

            int alphacCov = 2;
            double c1 = alphacCov / (Math.Pow(nDim + 1.3, 2) + muEff);
            double cmu = Math.Min(1 - c1,
                alphacCov * (muEff - 2 + 1 / muEff) / (Math.Pow(nDim + 2, 2) + alphacCov * muEff / 2));
            if (!(c1 <= 1 - cmu))
            {
                throw new Exception("invalid learning rate for the rank-one update");
            }

            if (!(cmu <= 1 - c1))
            {
                throw new Exception("invalid learning rate for the rank-μ update");
            }

            double minAlpha = Math.Min(1 + c1 / cmu,
                Math.Min(1 + 2 * muEffMinus / (muEff + 2), (1 - c1 - cmu) / (nDim * cmu)));

            double positiveSum = weightsPrime.Where(x => x > 0).Sum();
            double negativeSum = Math.Abs(weightsPrime.Where(x => x < 0).Sum());

            Vector<double> weights = Vector<double>.Build.Dense(weightsPrime.Count);
            weightsPrime.CopyTo(weights);
            bool[] weightsIsNotNegative = weightsPrime.Select(x => x >= 0).ToArray();
            for (int i = 0; i < weights.Count; i++)
            {
                weights[i] = weightsIsNotNegative[i]
                    ? 1 / positiveSum * weightsPrime[i]
                    : minAlpha / negativeSum * weightsPrime[i];
            }

            int cm = 1;

            double cSigma = (muEff + 2) / (nDim + muEff + 5);
            double dSigma = 1 + 2 * Math.Max(0, Math.Sqrt((muEff - 1) / (nDim + 1)) - 1) + cSigma;
            if (!(cSigma < 1))
            {
                throw new Exception("invalid learning rate for cumulation for the step-size control");
            }

            double cc = (4 + muEff / nDim) / (nDim + 4 + 2 * muEff / nDim);
            if (!(cc <= 1))
            {
                throw new Exception("invalid learning rate for cumulation for the rank-one update");
            }

            Dim = nDim;
            PopulationSize = populationSize;
            _mu = mu;
            _muEff = muEff;

            _cc = cc;
            _c1 = c1;
            _cmu = cmu;
            _cSigma = cSigma;
            _dSigma = dSigma;
            _cm = cm;

            _chiN = Math.Sqrt(Dim) * (1.0 - 1.0 / (4.0 * Dim) + 1.0 / (21.0 * Math.Pow(Dim, 2)));

            _weights = weights;

            _pSigma = Vector<double>.Build.Dense(Dim, 0);
            _pc = Vector<double>.Build.Dense(Dim, 0);

            _mean = Vector<double>.Build.DenseOfArray(mean.ToArray());
            _c = Matrix<double>.Build.DenseIdentity(Dim, Dim);
            _sigma = sigma;

            if (!(bounds == null || bounds.RowCount == Dim && bounds.ColumnCount == 2))
            {
                throw new Exception("bounds should be (n_dim, 2)-dim matrix");
            }

            _bounds = bounds;
            _nMaxResampling = nMaxResampling;

            Generation = 0;

            _normalDistribution = new Normal(0, 1);

            _epsilon = 1e-8;

            _tolSigma = tolSigma;
            _tolC = tolC;

            _funhistTerm = 10 + (int)Math.Ceiling(30.0 * Dim / PopulationSize);
            _funhistValues = new double[_funhistTerm * 2];

            _tolx = 1e-12 * _sigma;
        }

        public (Matrix<double> B, Vector<double> D) EigenDecomposition()
        {
            if (_b != null && _d != null)
                return (_b, _d);

            _c = (_c + _c.Transpose()) / 2;
            Evd<double> evdC = _c.Evd();
            Matrix<double> b = evdC.EigenVectors;
            Vector<double> d =
                Vector<double>.Build.DenseOfArray(evdC.EigenValues.Select(x => RootTemp(x.Real)).ToArray());

            var diagSquaredD = Matrix<double>.Build.DiagonalOfDiagonalArray(d.AsArray().Select(x => x * x).ToArray());
            var bTimesDiagSquaredD = b * diagSquaredD;
            var bTimesDiagSquaredDTimesBt = bTimesDiagSquaredD * b.Transpose();
            _c = bTimesDiagSquaredDTimesBt;
            return (b, d);
        }

        public static (Matrix<double> B, Vector<double> D) EigenDecomposition2(Matrix<double> c)
        {
            c = (c + c.Transpose()) / 2;
            Evd<double> evdC = c.Evd();
            Matrix<double> b = evdC.EigenVectors;
            Vector<double> d =
                Vector<double>.Build.DenseOfArray(evdC.EigenValues.Select(x => RootTemp(x.Real)).ToArray());

            var diagSquaredD = Matrix<double>.Build.DiagonalOfDiagonalArray(d.AsArray().Select(x => x * x).ToArray());
            var bTimesDiagSquaredD = b * diagSquaredD;
            var bTimesDiagSquaredDTimesBt = bTimesDiagSquaredD * b.Transpose();
            c = bTimesDiagSquaredDTimesBt;
            return (b, d);
        }

        static double RootTemp(double input)
        {
            return input >= 0 ? Math.Sqrt(input) : Math.Sqrt(1e-8);
        }

        public bool ShouldStop()
        {
            (Matrix<double> b, Vector<double> d) = EigenDecomposition();

            var dC = _c.Diagonal();

            if (Generation > _funhistTerm && _funhistValues.Max() - _funhistValues.Min() < Tolfun)
            {
                return true;
            }

            if (dC.All(x => _sigma * x < _tolx) && _pc.All(x => _sigma * x < _tolx))
            {
                return true;
            }

            if (_sigma * d.Max() > Tolxup)
            {
                return true;
            }

            if (dC.Any(x => _mean == _mean + 0.2 * _sigma * Math.Sqrt(x)))
            {
                return true;
            }

            int i = Generation % Dim;

            if (b.Column(i).All(b => _mean == _mean + 0.1 * _sigma * d[i] * b))
            {
                return true;
            }

            double conditionCov = d.Max() / d.Min();
            if (conditionCov > Tolconditioncov)
            {
                return true;
            }

            return false;
        }

        public bool IsConverged()
        {
            return _sigma < _tolSigma && _c.L2Norm() < _tolC;
        }

        public void SetBounds(Matrix<double> bounds = null)
        {
            if (!(bounds == null || bounds.RowCount == Dim && bounds.ColumnCount == 2))
            {
                throw new Exception("bounds should be (n_dim, 2)-dim matrix");
            }

            _bounds = bounds;
        }

        public Vector<double> Ask()
        {
            for (int i = 0; i < _nMaxResampling; i++)
            {
                Vector<double> x = SampleSolution();
                if (IsFeasible(x))
                {
                    return x;
                }
            }

            Vector<double> xNew = SampleSolution();
            xNew = RepairInfeasibleParams(xNew);
            return xNew;
        }


        public void Tell(List<Solution> solutions)
        {
            if (solutions.Count != PopulationSize)
            {
                throw new ArgumentException("Must tell popsize-length solutions.");
            }

            Generation += 1;
            var sortedSolutions = solutions.OrderBy(x => x.Fitness).ToArray();

            int funhistIdx = 2 * (Generation % _funhistTerm);
            _funhistValues[funhistIdx] = sortedSolutions.First().Fitness;
            _funhistValues[funhistIdx + 1] = sortedSolutions.Last().Fitness;

            // Sample new population of search_points, for k=1, ..., popsize
            (Matrix<double> b, Vector<double> d) = EigenDecomposition();

            _b = null;
            _d = null;

            Matrix<double> xK = Matrix<double>.Build.DenseOfRowVectors(sortedSolutions.Select(x => x.Parameters));
            var yKRows = xK.EnumerateRows().Select((x, i) => (x - _mean) / _sigma).ToArray();
            Matrix<double> yK = Matrix<double>.Build.DenseOfRowVectors(yKRows);

            // Selection and recombination
            Vector<double> yW = PointwiseMultiplicationOnRows(yK.SubMatrix(0, _mu, 0, Dim).Transpose(),
                _weights.SubVector(0, _mu));
            _mean += _cm * _sigma * yW;

            var diag = Matrix<double>.Build.DenseOfDiagonalArray((1.0 / d).AsArray());
            Matrix<double> c2 = b * diag * b.Transpose();

            double sqrtFactor = Math.Sqrt(_cSigma * (2 - _cSigma) * _muEff);
            var g = sqrtFactor * (c2 * yW);
            var q = c2 * yW;
            _pSigma = (1 - _cSigma) * _pSigma + sqrtFactor * (c2 * yW);

            double normPSigma = _pSigma.L2Norm();
            _sigma *= Math.Exp(_cSigma / _dSigma * (normPSigma / _chiN - 1));
            //Console.WriteLine($"Sigma: | {string.Join(", ", _p_sigma)} | {norm_pSigma}, {Math.Exp(_c_sigma / _d_sigma * (norm_pSigma / _chi_n - 1))}, {_sigma}");
            double hSigmaCondLeft = normPSigma / Math.Sqrt(1 - Math.Pow(1 - _cSigma, 2 * (Generation + 1)));
            double hSigmaCondRight = (1.4 + 2 / (double)(Dim + 1)) * _chiN;
            double hSigma = hSigmaCondLeft < hSigmaCondRight ? 1.0 : 0.0;

            _pc = (1 - _cc) * _pc + hSigma * Math.Sqrt(_cc * (2 - _cc) * _muEff) * yW;

            Vector<double> wIo = Vector<double>.Build.Dense(_weights.Count, 1);
            Vector<double> wIee = (c2 * yK.Transpose()).ColumnNorms(2).PointwisePower(2);
            for (int i = 0; i < _weights.Count; i++)
            {
                if (_weights[i] >= 0)
                {
                    wIo[i] = _weights[i] * 1;
                }
                else
                {
                    wIo[i] = _weights[i] * Dim / (wIee[i] + _epsilon);
                }
            }

            double deltaHSigma = (1 - hSigma) * _cc * (2 - _cc);
            if (!(deltaHSigma <= 1))
            {
                throw new Exception("invalid value of delta_h_sigma");
            }

            Matrix<double> rankOne = _pc.OuterProduct(_pc);
            Matrix<double> rankMu = Matrix<double>.Build.Dense(yK.ColumnCount, yK.ColumnCount, 0);
            for (int i = 0; i < wIo.Count; i++)
            {
                rankMu += wIo[i] * yK.Row(i).OuterProduct(yK.Row(i));
            }

            _c =
                (
                    1
                    + _c1 * deltaHSigma
                    - _c1
                    - _cmu * _weights.Sum()
                )
                * _c
                + _c1 * rankOne
                + _cmu * rankMu
                ;
        }

        private Vector<double> RepairInfeasibleParams(Vector<double> param)
        {
            if (_bounds == null)
            {
                return param;
            }

            Vector<double> newParam = param.PointwiseMaximum(_bounds.Column(0));
            newParam = newParam.PointwiseMinimum(_bounds.Column(1));
            return newParam;
        }

        private bool IsFeasible(Vector<double> param)
        {
            if (_bounds == null)
            {
                return true;
            }

            bool isCorrectLower = true;
            bool isCorrectUpper = true;
            for (int i = 0; i < param.Count; i++)
            {
                isCorrectLower &= param[i] >= _bounds[i, 0];
                isCorrectUpper &= param[i] <= _bounds[i, 1];
            }

            return isCorrectLower & isCorrectUpper;
        }

        private Vector<double> SampleSolution()
        {
            (Matrix<double> b, Vector<double> d) = EigenDecomposition();

            Vector<double> z = Vector<double>.Build.Dense(Dim);
            for (int i = 0; i < z.Count; i++)
            {
                z[i] = _normalDistribution.Sample();
            }

            var h = b * Matrix<double>.Build.DenseOfDiagonalArray(d.AsArray());
            var y = PointwiseMultiplicationOnRows(h, z);
            Vector<double> x = _mean + _sigma * y;
            return x;
        }

        private Vector<double> PointwiseMultiplicationOnRows(Matrix<double> matrix, Vector<double> vector)
        {
            var rows = matrix.EnumerateRows().Select(x => x.PointwiseMultiply(vector)).ToArray();
            return Vector<double>.Build.DenseOfArray(rows.Select(x => x.Sum()).ToArray());
        }
    }
}