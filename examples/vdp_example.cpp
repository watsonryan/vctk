#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "distributions.hpp"
#include "vctk.hpp"

namespace {

Eigen::MatrixXd make_data() {
  constexpr int n_per_cluster = 100;
  constexpr int d = 2;
  constexpr int n_clusters = 3;
  const int n = n_per_cluster * n_clusters;

  Eigen::MatrixXd X(n, d);

  std::mt19937 rng(42);
  std::normal_distribution<double> z(0.0, 1.0);

  const std::vector<Eigen::Vector2d> means = {
      Eigen::Vector2d(-2.2, -1.5),
      Eigen::Vector2d(2.0, 1.8),
      Eigen::Vector2d(-0.2, 3.0),
  };

  const std::vector<Eigen::Matrix2d> chol = {
      (Eigen::Matrix2d() << 0.70, 0.15, 0.00, 0.45).finished(),
      (Eigen::Matrix2d() << 0.55, -0.05, 0.00, 0.60).finished(),
      (Eigen::Matrix2d() << 0.50, 0.20, 0.00, 0.40).finished(),
  };

  int row = 0;
  for (int k = 0; k < n_clusters; ++k) {
    for (int i = 0; i < n_per_cluster; ++i) {
      Eigen::Vector2d u(z(rng), z(rng));
      const Eigen::Vector2d sample = means[k] + chol[k] * u;
      X.row(row++) = sample.transpose();
    }
  }

  return X;
}

Eigen::MatrixXd make_data_batch2() {
  constexpr int n_per_cluster = 100;
  constexpr int d = 2;
  constexpr int n_clusters = 3;
  const int n = n_per_cluster * n_clusters;

  Eigen::MatrixXd X(n, d);

  std::mt19937 rng(7);
  std::normal_distribution<double> z(0.0, 1.0);

  // Slightly shifted means to emulate a streaming/incremental update.
  const std::vector<Eigen::Vector2d> means = {
      Eigen::Vector2d(-2.0, -1.2),
      Eigen::Vector2d(2.2, 2.0),
      Eigen::Vector2d(0.1, 3.2),
  };

  const std::vector<Eigen::Matrix2d> chol = {
      (Eigen::Matrix2d() << 0.75, 0.12, 0.00, 0.45).finished(),
      (Eigen::Matrix2d() << 0.50, -0.08, 0.00, 0.65).finished(),
      (Eigen::Matrix2d() << 0.48, 0.18, 0.00, 0.42).finished(),
  };

  int row = 0;
  for (int k = 0; k < n_clusters; ++k) {
    for (int i = 0; i < n_per_cluster; ++i) {
      Eigen::Vector2d u(z(rng), z(rng));
      const Eigen::Vector2d sample = means[k] + chol[k] * u;
      X.row(row++) = sample.transpose();
    }
  }

  return X;
}

void write_points_csv(const std::filesystem::path &path, const Eigen::MatrixXd &X,
                      const Eigen::MatrixXd &qZ) {
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("failed to open points CSV for writing");

  out << "x,y,cluster,max_prob\n";
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    Eigen::Index k = 0;
    const double p = qZ.row(i).maxCoeff(&k);
    out << X(i, 0) << ',' << X(i, 1) << ',' << k << ',' << p << '\n';
  }
}

void write_centers_csv(const std::filesystem::path &path,
                       const std::vector<distributions::GaussWish> &clusters,
                       const distributions::StickBreak &weights) {
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("failed to open centers CSV for writing");

  const Eigen::ArrayXd w = weights.Elogweight().exp();

  out << "cluster,mean_x,mean_y,weight\n";
  for (std::size_t k = 0; k < clusters.size(); ++k) {
    const auto &m = clusters[k].getmean();
    const double wk = (k < static_cast<std::size_t>(w.size())) ? w(k) : 0.0;
    out << k << ',' << m(0) << ',' << m(1) << ',' << wk << '\n';
  }
}

void write_merged_centers_csv(
    const std::filesystem::path &path,
    const std::vector<vctk::merge::MixtureComponent> &gmm) {
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("failed to open merged centers CSV for writing");

  out << "cluster,mean_x,mean_y,weight\n";
  for (std::size_t k = 0; k < gmm.size(); ++k) {
    out << k << ',' << gmm[k].mean(0) << ',' << gmm[k].mean(1) << ','
        << gmm[k].weight << '\n';
  }
}

} // namespace

int main() {
  try {
    const Eigen::MatrixXd X1 = make_data();
    const Eigen::MatrixXd X2 = make_data_batch2();

    Eigen::MatrixXd qZ1;
    Eigen::MatrixXd qZ2;
    distributions::StickBreak weights1;
    distributions::StickBreak weights2;
    std::vector<distributions::GaussWish> clusters1;
    std::vector<distributions::GaussWish> clusters2;

    vctk::VbemOptions opts;
    opts.max_clusters = 10;
    opts.max_vbem_iters = 100;
    opts.split_refine_iters = 20;

    const double F1 =
        vctk::learnVDP(X1, qZ1, weights1, clusters1, vctk::PRIORVAL, opts);
    const auto prior = vctk::merge::mergeMixtureModel(
        X1, qZ1, {}, clusters1, weights1, 0.05, 32);

    const double F2 =
        vctk::learnVDP(X2, qZ2, weights2, clusters2, vctk::PRIORVAL, opts);
    const auto merged = vctk::merge::mergeMixtureModel(
        X2, qZ2, prior, clusters2, weights2, 0.05, 32);

    const std::filesystem::path out_dir = "build/examples";
    std::filesystem::create_directories(out_dir);

    const auto points_csv = out_dir / "vdp_points.csv";
    const auto centers_csv = out_dir / "vdp_centers.csv";
    const auto merged_centers_csv = out_dir / "vdp_merged_centers.csv";

    write_points_csv(points_csv, X2, qZ2);
    write_centers_csv(centers_csv, clusters2, weights2);
    write_merged_centers_csv(merged_centers_csv, merged);

    std::cout << "VDP + Merge complete\n";
    std::cout << "Batch1: N=" << X1.rows() << ", D=" << X1.cols()
              << ", clusters=" << clusters1.size() << ", F=" << F1 << "\n";
    std::cout << "Batch2: N=" << X2.rows() << ", D=" << X2.cols()
              << ", clusters=" << clusters2.size() << ", F=" << F2 << "\n";
    std::cout << "Merged global components: " << merged.size() << "\n";
    std::cout << "Wrote: " << points_csv << "\n";
    std::cout << "Wrote: " << centers_csv << "\n";
    std::cout << "Wrote: " << merged_centers_csv << "\n";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
