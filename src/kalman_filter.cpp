#include "kalman_filter.h"
#include <exception>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;  // state
  P_ = P_in;  // state uncertainty
  F_ = F_in;  // state transition function
  H_ = H_in;  // measurement transition matrix
  R_ = R_in;  // measurement covariance matrix (uncertainties)
  Q_ = Q_in;  // process noise matrix
}

void KalmanFilter::Predict() {
  /**
    * predict the state
    * This is the implementation of the basic Kalman Filter
    * Algorithm (11), (12) on page 2 of cheat sheet.
  */

  // using state transition function:
  x_ = F_ * x_; // + u (u is set to zero and therefore ignored)

  // prediction to the state uncertainty
  P_ = F_ * P_ * F_.transpose() + Q_;
}



void KalmanFilter::Update_Generic(const Eigen::VectorXd &y) {

    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    MatrixXd K = P_ * H_.transpose() * S.inverse();         // Kalman Gain
    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());  // Identity matrix

    x_ = x_ + K * y;
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
    * equations (13) to (17) in cheat sheet.
  */
  VectorXd y = z - H_ * x_;

  this->Update_Generic(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  VectorXd y = z - this->H_Radar(x_);

  this->Update_Generic(y);
}

Eigen::VectorXd KalmanFilter::H_Radar(const Eigen::VectorXd &x_prime) {
  unsigned int expected_dim = 4;
  unsigned int destination_dim = 3;

  VectorXd h_x_prime = VectorXd(destination_dim);
  h_x_prime.setConstant(destination_dim);

  if(x_prime.rows() != expected_dim) {
    throw("H_Radar () input has wrong shape!");
  }
  const double &px = x_prime(0);
  const double &py = x_prime(1);
  const double &vx = x_prime(2);
  const double &vy = x_prime(3);

  h_x_prime(0) = std::sqrt((px*px)+(py*py));
  h_x_prime(1) = std::atan(py/px);

  // set the rho_dot only if no division by zero occurs!
  if(h_x_prime(0) > 1e-4) {
    h_x_prime(2) = ((px*vx) + (py*vy)) / h_x_prime(0);
  }

  return h_x_prime;
}

