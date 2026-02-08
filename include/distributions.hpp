/**
 * Various distributions used by VCTk
 * ----------------------------------
 *
 *                        _..._
 *                     .-'_..._''.
 *  .----.     .----..' .'      '.\          .
 *   \    \   /    // .'                   .'|
 *    '   '. /'   /. '               .|  .'  |
 *    |    |'    / | |             .' |_<    |
 *    |    ||    | | |           .'     ||   | ____
 *    '.   `'   .' . '          '--.  .-'|   | \ .'
 *     \        /   \ '.          .|  |  |   |/  .
 *      \      /     '. `._____.-'/|  |  |    /\  \
 *       '----'        `-.______ / |  '.'|   |  \  \
 *                              `  |   / '    \  \  \
 *                                 `'-' '------'  '---'
 *
 * @author watson
 * @date 2025-07-12
 */

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace distributions {

inline constexpr double BETAPRIOR = 1.0;
inline constexpr double NUPRIOR = 1.0;
inline constexpr double ALPHA1PRIOR = 1.0;
inline constexpr double ALPHA2PRIOR = 1.0;
inline constexpr double APRIOR = 1.0;

using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
using Index = Eigen::Index;

class WeightDist {
public:
  WeightDist(const WeightDist &) = default;
  WeightDist &operator=(const WeightDist &) = default;
  WeightDist(WeightDist &&) = default;
  WeightDist &operator=(WeightDist &&) = default;
  virtual ~WeightDist() = default;

  /// update posterior from observation counts
  virtual void update(const Eigen::ArrayXd &Nk) = 0;

  /// E_q[ log π ] for each weight component
  [[nodiscard]] virtual const Eigen::ArrayXd &Elogweight() const noexcept = 0;

  /// cached observation counts
  [[nodiscard]] const Eigen::ArrayXd &getNk() const noexcept { return Nk_; }

  /// free‑energy contribution
  [[nodiscard]] virtual double fenergy() const noexcept = 0;

protected:
  WeightDist() : Nk_(Eigen::ArrayXd::Zero(1)) {}

  Eigen::ArrayXd Nk_;
};

// ═════════════════════════════════════════════════════════════════════════════
// 2. Stick‑Breaking / Dirichlet‑family weight distributions
// ═════════════════════════════════════════════════════════════════════════════
class StickBreak : public WeightDist {
public:
  StickBreak();
  explicit StickBreak(double concentration);

  void update(const Eigen::ArrayXd &Nk) override;
  [[nodiscard]] const Eigen::ArrayXd &Elogweight() const noexcept override {
    return E_logpi_;
  }
  [[nodiscard]] double fenergy() const noexcept override;

protected:
  // prior
  double alpha1_p_{ALPHA1PRIOR};
  double alpha2_p_{ALPHA2PRIOR};
  double F_p_{0.0};

  // posterior
  Eigen::ArrayXd alpha1_;
  Eigen::ArrayXd alpha2_;
  Eigen::ArrayXd E_logv_;
  Eigen::ArrayXd E_lognv_;
  Eigen::ArrayXd E_logpi_;

  std::vector<std::pair<int, double>> ordvec_;

  void prior_fcalc() noexcept;
};

class GDirichlet final : public StickBreak {
public:
  using StickBreak::StickBreak; // inherit ctors
  void update(const Eigen::ArrayXd &Nk) override;
  double fenergy() const noexcept override;
};

class Dirichlet final : public WeightDist {
public:
  Dirichlet();
  explicit Dirichlet(double alpha);

  void update(const Eigen::ArrayXd &Nk) override;
  [[nodiscard]] const Eigen::ArrayXd &Elogweight() const noexcept override {
    return E_logpi_;
  }
  [[nodiscard]] double fenergy() const noexcept override;

private:
  // prior
  double alpha_p_{ALPHA1PRIOR};
  double F_p_{0.0};

  // posterior
  Eigen::ArrayXd alpha_;
  Eigen::ArrayXd E_logpi_;
};

// ═════════════════════════════════════════════════════════════════════════════
// 3. Cluster‑parameter base class
// ═════════════════════════════════════════════════════════════════════════════
class ClusterDist {
public:
  ClusterDist(const ClusterDist &) = default;
  ClusterDist &operator=(const ClusterDist &) = default;
  ClusterDist(ClusterDist &&) = default;
  ClusterDist &operator=(ClusterDist &&) = default;
  virtual ~ClusterDist() = default;

  // sufficient‑statistics interface
  virtual void addobs(Eigen::Ref<const Eigen::VectorXd> qZk,
                      Eigen::Ref<const Eigen::MatrixXd> X) = 0;
  virtual void update() = 0;
  virtual void clearobs() = 0;

  // likelihood / energy
  [[nodiscard]] virtual Eigen::VectorXd Eloglike(
      Eigen::Ref<const Eigen::MatrixXd> X) const = 0;
  [[nodiscard]] virtual double fenergy() const noexcept = 0;

  // splitting utility
  [[nodiscard]] virtual ArrayXb splitobs(
      Eigen::Ref<const Eigen::MatrixXd> X) const = 0;

  // metadata
  [[nodiscard]] double getN() const noexcept { return N_; }
  [[nodiscard]] double getprior() const noexcept { return prior_; }

protected:
  ClusterDist(double prior, std::size_t D) : D_(D), prior_(prior) {}

  std::size_t D_;
  double prior_;
  double N_{0.0};
};

// ═════════════════════════════════════════════════════════════════════════════
// 4. Concrete cluster distributions
// ═════════════════════════════════════════════════════════════════════════════
class GaussWish final : public ClusterDist {
public:
  GaussWish(double clustwidth, std::size_t D);

  void addobs(Eigen::Ref<const Eigen::VectorXd> qZk,
              Eigen::Ref<const Eigen::MatrixXd> X) override;
  void update() override;
  void clearobs() override;
  [[nodiscard]] Eigen::VectorXd
  Eloglike(Eigen::Ref<const Eigen::MatrixXd> X) const override;
  [[nodiscard]] ArrayXb splitobs(Eigen::Ref<const Eigen::MatrixXd> X) const
      override;
  [[nodiscard]] double fenergy() const noexcept override;

  [[nodiscard]] const Eigen::RowVectorXd &getmean() const noexcept {
    return m_;
  }
  [[nodiscard]] Eigen::MatrixXd getcov() const { return iW_ / nu_; }

private:
  // priors
  double nu_p_;
  double beta_p_;
  Eigen::RowVectorXd m_p_;
  Eigen::MatrixXd iW_p_;
  double logdW_p_{0.0};
  double F_p_{0.0};

  // posteriors
  double nu_;
  double beta_;
  Eigen::RowVectorXd m_;
  Eigen::MatrixXd iW_;
  double logdW_{0.0};

  // sufficient stats
  double N_s_{0.0};
  Eigen::RowVectorXd x_s_;
  Eigen::MatrixXd xx_s_;
};

class NormGamma final : public ClusterDist {
public:
  NormGamma(double clustwidth, std::size_t D);

  void addobs(Eigen::Ref<const Eigen::VectorXd> qZk,
              Eigen::Ref<const Eigen::MatrixXd> X) override;
  void update() override;
  void clearobs() override;
  [[nodiscard]] Eigen::VectorXd
  Eloglike(Eigen::Ref<const Eigen::MatrixXd> X) const override;
  [[nodiscard]] ArrayXb splitobs(Eigen::Ref<const Eigen::MatrixXd> X) const
      override;
  [[nodiscard]] double fenergy() const noexcept override;

  [[nodiscard]] const Eigen::RowVectorXd &getmean() const noexcept {
    return m_;
  }
  [[nodiscard]] Eigen::RowVectorXd getcov() const { return L_ * nu_; }

private:
  // priors
  double nu_p_;
  double beta_p_;
  Eigen::RowVectorXd m_p_;
  Eigen::RowVectorXd L_p_;
  double logL_p_{0.0};

  // posteriors
  double nu_;
  double beta_;
  Eigen::RowVectorXd m_;
  Eigen::RowVectorXd L_;
  double logL_{0.0};

  // sufficient stats
  double N_s_{0.0};
  Eigen::RowVectorXd x_s_;
  Eigen::RowVectorXd xx_s_;
};

class ExpGamma final : public ClusterDist {
public:
  ExpGamma(double obsmag, std::size_t D);

  void addobs(Eigen::Ref<const Eigen::VectorXd> qZk,
              Eigen::Ref<const Eigen::MatrixXd> X) override;
  void update() override;
  void clearobs() override;
  [[nodiscard]] Eigen::VectorXd
  Eloglike(Eigen::Ref<const Eigen::MatrixXd> X) const override;
  [[nodiscard]] ArrayXb splitobs(Eigen::Ref<const Eigen::MatrixXd> X) const
      override;
  [[nodiscard]] double fenergy() const noexcept override;

  [[nodiscard]] Eigen::RowVectorXd getrate() const { return a_ * ib_; }

private:
  // priors
  double a_p_;
  double b_p_;

  // posteriors
  double a_;
  Eigen::RowVectorXd ib_;
  double logb_{0.0};

  // sufficient stats
  double N_s_{0.0};
  Eigen::RowVectorXd x_s_;
};

} // namespace distributions
