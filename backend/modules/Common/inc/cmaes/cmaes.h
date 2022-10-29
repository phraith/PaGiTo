#ifndef GISAXSMODELINGFRAMEWORK_CMAES_H
#define GISAXSMODELINGFRAMEWORK_CMAES_H

#include <eigen3/Eigen/Eigen>
#include <random>
#include <memory>
#include "solution.h"
#include "weights_information.h"

class Cmaes {
public:
    Cmaes(std::vector<double> mean, double sigma, std::shared_ptr<Eigen::MatrixX<double>> bounds = nullptr,
          int n_max_resampling = 100, int seed = 0, double tol_sigma = 1e-4, double tol_C = 1e-4);

    std::tuple<std::shared_ptr<Eigen::MatrixX<double>>, std::shared_ptr<Eigen::VectorX<double>>> EigenDecomposition();

    [[nodiscard]] int PopulationSize() const;

    bool ShouldStop();

    bool IsConverged();

    void SetBounds(std::shared_ptr<Eigen::MatrixX<double>> bounds = nullptr);

    Eigen::VectorX<double> Ask();

    void Tell(std::vector<Solution> &solutions);

private:
    [[nodiscard]] static double RootTemp(double input);

    Eigen::VectorX<double> SampleSolution();

    static Eigen::VectorX<double>
    PointwiseMultiplicationOnRows(Eigen::MatrixX<double> matrix, Eigen::VectorX<double> vector);

    bool IsFeasable(Eigen::VectorX<double> param);

    Eigen::VectorX<double> RepairInfeasableParams(Eigen::VectorX<double> params);

    static WeightsInformation CreateInitialWeights(int population_size, int mu, int dim);
    static double CalculateInitialCSigma(double mu_eff, int dim);
    static double CalculateInitialCc(double mu_eff, int dim);
    static double CalculateInitialDSigma(double c_sigma, double mu_eff, int dim);
    static double CalculateInitialChiN(int dim);



    double sigma_;
    int n_max_resampling_;
    double tolx_;
    double tol_sigma_;
    double tol_C_;
    std::shared_ptr<Eigen::MatrixX<double>> bounds_;
    int dim_;
    int population_size_;
    int generation_;
    int mu_;
    int cm_;
    std::mt19937 gen_;
    std::normal_distribution<> normal_distribution_;
    WeightsInformation weights_information_;
    double c_sigma_;
    double cc_;
    double d_sigma_;
    double chi_n_;

    int funhist_term_;

    Eigen::VectorX<double> p_sigma_;
    Eigen::VectorX<double> pc_;
    Eigen::VectorX<double> mean_;
    std::shared_ptr<Eigen::MatrixX<double>> C_;
    std::shared_ptr<Eigen::VectorX<double>> D_;
    std::shared_ptr<Eigen::MatrixX<double>> B_;

    std::vector<double> function_history_;
    const double epsilon_ = 1e-8;
    const double tolxup_ = 1e4;
    const double tolfun_ = 1e-12;
    const double tolconditioncov_ = 1e14;
};

#endif //GISAXSMODELINGFRAMEWORK_CMAES_H
