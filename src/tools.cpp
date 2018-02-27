#include <iostream>
#include <exception>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
  if (estimations.size() != ground_truth.size() || estimations.size() <= 0) {
    std::cout << "Error computing the Root Mean Squared Error." << std::endl;
  }

  // probe the dimensions. we can access at postion 0 since we checked the length
  // in the statement above
  unsigned int dimensions = estimations[0].rows();
  double tmp_difference = 0.0;

  VectorXd rmse(dimensions);
  rmse.setConstant(0.0);

  // iterate all elements
  for(unsigned int i = 0; i < estimations.size(); i++) {

    // iterate all dimensions
    for(unsigned int d = 0; d < dimensions; d++) {
      tmp_difference = (estimations[i](d) - ground_truth[i](d));
      rmse(d) +=  tmp_difference * tmp_difference;
    }
  }

  // compute mean and square root
  for(unsigned int d = 0; d < dimensions; d++) {
    rmse(d) /= static_cast<double>(estimations.size());
    rmse(d) = std::sqrt(static_cast<double>(rmse(d)));
  }

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
    * This is done with the help of the code snipped of
    * the course Lession 6: 19. Jacobian Matrix Part 2.
  */
  MatrixXd Hj(3, 4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculations
  float c1 = (px*px) + (py*py);
  float c2 = std::sqrt(c1);
  float c3 = (c1*c2);

  // checking div by zero
  if(fabs(c1) < 1e-4) {
    std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
    return Hj;
  }

  // compute the Jacobian matrix
  Hj << (px / c2), (py / c2), 0, 0,
        -(py / c1), (px / c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}

VectorXd Tools::ConvertPolarToCartesian(const VectorXd &polar_measurement) {

  int expected_dims = 3;
  int destination_dims = 4;

  VectorXd cartesian_measurement(destination_dims);
  cartesian_measurement.setConstant(0.0);

  if(polar_measurement.rows() != expected_dims) {
    throw("ConvertPolarToCartesian() input does not match expected dimensions!");
  }

  float rho = polar_measurement(0); // Distance
  float phi = polar_measurement(1); // Bearing
  float rho_dot = polar_measurement(2); // Range Rate

  cartesian_measurement(0) = std::cos(phi) * rho;
  cartesian_measurement(1) = std::sin(phi) * rho;

  return cartesian_measurement;
}


