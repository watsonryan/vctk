#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Dense>

#include "distributions.hpp"

namespace {

void test_ctor_validation() {
  bool threw = false;

  try {
    distributions::ExpGamma bad(0.0, 2);
    (void)bad;
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    distributions::GaussWish bad(1.0, 0);
    (void)bad;
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    distributions::NormGamma bad(1.0, 0);
    (void)bad;
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
}

void test_gdirichlet_empty_update_validation() {
  distributions::GDirichlet gd;
  bool threw = false;
  try {
    Eigen::ArrayXd empty(0);
    gd.update(empty);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
}

void test_read_path_dimension_validation() {
  const Eigen::MatrixXd badX = Eigen::MatrixXd::Zero(4, 3);
  bool threw = false;

  distributions::GaussWish gw(1.0, 2);
  try {
    (void)gw.Eloglike(badX);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)gw.splitobs(badX);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  distributions::NormGamma ng(1.0, 2);
  threw = false;
  try {
    (void)ng.Eloglike(badX);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)ng.splitobs(badX);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  distributions::ExpGamma eg(1.0, 2);
  threw = false;
  try {
    (void)eg.Eloglike(badX);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)eg.splitobs(badX);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
}

void test_eloglike_behavior() {
  Eigen::MatrixXd Xtrain(4, 1);
  Xtrain << -0.1, 0.0, 0.1, 0.2;
  Eigen::VectorXd q = Eigen::VectorXd::Ones(Xtrain.rows());

  Eigen::MatrixXd Xprobe(2, 1);
  Xprobe << 0.05, 3.0;

  distributions::GaussWish gw(1.0, 1);
  gw.addobs(q, Xtrain);
  gw.update();
  Eigen::VectorXd gw_ll = gw.Eloglike(Xprobe);
  assert(gw_ll.size() == Xprobe.rows());
  assert(gw_ll.allFinite());
  assert(gw_ll(0) > gw_ll(1));

  distributions::NormGamma ng(1.0, 1);
  ng.addobs(q, Xtrain);
  ng.update();
  Eigen::VectorXd ng_ll = ng.Eloglike(Xprobe);
  assert(ng_ll.size() == Xprobe.rows());
  assert(ng_ll.allFinite());
  assert(ng_ll(0) > ng_ll(1));

  Eigen::MatrixXd Xexp(4, 1);
  Xexp << 0.1, 0.2, 0.3, 0.4;
  distributions::ExpGamma eg(1.0, 1);
  eg.addobs(q, Xexp);
  eg.update();

  Eigen::MatrixXd Xprobe_exp(2, 1);
  Xprobe_exp << 0.15, 2.0;
  Eigen::VectorXd eg_ll = eg.Eloglike(Xprobe_exp);
  assert(eg_ll.size() == Xprobe_exp.rows());
  assert(eg_ll.allFinite());
  assert(eg_ll(0) > eg_ll(1));
}

} // namespace

int main() {
  test_ctor_validation();
  test_gdirichlet_empty_update_validation();
  test_read_path_dimension_validation();
  test_eloglike_behavior();

  std::cout << "All distributions tests passed ✔️\n";
  return 0;
}
