#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // transition matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  // state initialization vector (px, py, vx, vy)
  VectorXd x_init = VectorXd(4);

  // state covariance matrix
  MatrixXd P_init = MatrixXd(4, 4);
  P_init << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

  // initial transition matrix F
  MatrixXd F_init = MatrixXd(4, 4);
  F_init << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;


  // initialize process noises_ax and noise_ay
  noise_ax = 9.0;
  noise_ay = 9.0;

  // Process noise matrix - is going to be recalculated at every measurement
  MatrixXd Q_init = MatrixXd(4, 4);
  Q_init.setConstant(0.0);

  // initialize Extended Kalman Filter class
  this->ekf_.Init(x_init, P_init, F_init, H_laser_, R_laser_, Q_init);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      this->ekf_.x_ = tools.ConvertPolarToCartesian(measurement_pack.raw_measurements_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      this->ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // set the previous timestamp
    this->previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // Update the state transition matrix F according to the new elapsed time.
  //    - Time is measured in seconds, so the delta is divided by 1000000
  float dt = (measurement_pack.timestamp_ - this->previous_timestamp_) / 1000000.0;
  this->previous_timestamp_ = measurement_pack.timestamp_;

  // Update state transition matrix F
  this->ekf_.F_(0, 2) = dt;
  this->ekf_.F_(1, 3) = dt;

  //    Update the process noise covariance matrix.
  //    Use noise_ax = 9 and noise_ay = 9 for your Q matrix.

  //    The process noise covariance matrix is initialized like discribed by
  //    Formula (40) in the sensor-fusion-ekf-reference.pdf on page 4.
  float dt2 = dt * dt;
  float dt3 = dt2 * dt;
  float dt4 = dt3 * dt;
  dt4 /= 4.0;
  dt3 /= 2.0;

  this->ekf_.Q_ << dt4 * this->noise_ax, 0, dt3 * this->noise_ax, 0,
                   0, dt4 * this->noise_ay, 0, dt3 * this->noise_ay,
                   dt3 * this->noise_ax, 0, dt2 * this->noise_ax, 0,
                   0, dt3 * this->noise_ay, 0, dt2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    // Set the H (measurement transition function) matrix
    // Set the R (measurement covariance) matrix
    this->ekf_.H_ = tools.CalculateJacobian(this->ekf_.x_);
    this->ekf_.R_ = this->R_radar_;
    this->ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    // Set the H (measurement transition function) matrix
    // Set the R (measurement covariance) matrix
    //this->ekf_.H_ = this->H_laser_;
    //this->ekf_.R_ = this->R_laser_;
    //this->ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
