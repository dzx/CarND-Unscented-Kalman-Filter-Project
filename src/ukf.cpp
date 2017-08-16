#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = .6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 6.;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_= false;

  n_x_ = 5;
  n_aug_= 7;
  time_us_= time(NULL);
  lambda_ = 3 - n_aug_;
  weights_= VectorXd(2 * n_aug_ + 1);
  weights_[0] = lambda_ / (lambda_ + n_aug_);
  for(int i = 1; i != weights_.size(); i++)
    weights_[i] = 1./2./(lambda_ + n_aug_);
  estimate_count_ = 0;
  nis_over_count_ = 0;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if(!is_initialized_){
    Initialize(meas_package);
    is_initialized_= true;
  }else{
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.;

    if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
      time_us_ = meas_package.timestamp_;
      Prediction(dt);
      UpdateLidar(meas_package);
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
      time_us_ = meas_package.timestamp_;
      Prediction(dt);
      UpdateRadar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

//  std::cout << "Predicting" << std::endl;

  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  const int aug_w = 2 * n_aug_ + 1;
  MatrixXd Xsig_aug = MatrixXd(n_aug_, aug_w);
  x_aug.head(n_x_) = x_;
  x_aug.tail(n_aug_ - n_x_).setZero();
  P_aug.bottomRows(n_aug_ - n_x_).setZero();
  P_aug.rightCols(n_aug_ - n_x_).setZero();
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ +1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
//  std::cout << "P_aug" << std::endl << P_aug << std::endl;

  MatrixXd P_sqr = P_aug.llt().matrixL();
//  std::cout << "P_sqr" << std::endl << P_sqr << std::endl;
  P_sqr *= sqrt(lambda_ + n_aug_);
//  std::cout << "P_sqr weighted" << std::endl << P_sqr << std::endl;
  Xsig_aug.col(0) = x_aug;
  for(int i = 0; i != n_aug_; i++){
    Xsig_aug.col(i + 1) = x_aug + P_sqr.col(i);
    Xsig_aug.col(n_aug_ + i + 1) = x_aug - P_sqr.col(i);
  }
  // so this is our augmented sigma matrix
//  std::cout << "x-sig-aug" << std::endl << Xsig_aug << std::endl;
  Xsig_pred_ = MatrixXd(n_x_, aug_w);
  double px, py, v, theta, theta_dot;
  double mu_a, mu_theta_dot;
  double px_p, py_p, v_p, theta_p, theta_dot_p;
  for(int i = 0; i != aug_w; i++){
    px = Xsig_aug(0, i); py = Xsig_aug(1, i); v = Xsig_aug(2, i); theta = Xsig_aug(3, i);
    theta_dot = Xsig_aug(4, i); mu_a = Xsig_aug(5, i); mu_theta_dot = Xsig_aug(6, i);
    // avoid division by zero
    if(fabs(theta_dot) > .0001){
      px_p = px + v / theta_dot * (sin(theta + theta_dot * delta_t) - sin(theta)) +
          pow(delta_t, 2) * cos(theta) * mu_a / 2.;
      py_p = py + v / theta_dot * (cos(theta) - cos(theta + theta_dot * delta_t)) +
          pow(delta_t, 2) * sin(theta) * mu_a / 2.;
    }else{
      px_p = px + cos(theta) * delta_t * (v + delta_t * mu_a / 2.);
      py_p = py + sin(theta) * delta_t * (v + delta_t * mu_a / 2.);
    }
    v_p = v + delta_t * mu_a;
    theta_p = theta + delta_t * (theta_dot + delta_t * mu_theta_dot / 2.);
    theta_dot_p = theta_dot + delta_t * mu_theta_dot;
    //write predicted sigma points into right column
    Xsig_pred_.col(i) << px_p, py_p, v_p, theta_p, theta_dot_p;
  }  x_.setZero();
  for(int i = 0; i != aug_w; i++){
    x_ = x_ + Xsig_pred_.col(i) * weights_[i];
  }
//  std::cout << x_ << std::endl;
//  std::cout << "Xsig_pred_" << std::endl << Xsig_pred_ << std::endl;
  P_.setZero();
  for(int i = 0; i != aug_w; i++){
    VectorXd dev = Xsig_pred_.col(i) - x_;
    while(dev[3] > M_PI) dev[3] -= 2. * M_PI;
    while(dev[3] < -M_PI) dev[3] += 2. * M_PI;
    MatrixXd devMat = dev * dev.transpose();
    P_ = P_ + devMat * weights_[i];
  }
//  std::cout << "P_ post" << std::endl << P_ << std::endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  //  std::cout << "Updating with lidar" << std::endl;

  int n_z = meas_package.raw_measurements_.size();
  int n_sig = 2 * n_aug_ + 1;
  MatrixXd Zsig = MatrixXd(n_z, n_sig);
  VectorXd z_pred = VectorXd(n_z).setZero();
  MatrixXd S = MatrixXd(n_z, n_z).setZero();
  Zsig = Xsig_pred_.topLeftCorner(n_z, n_sig);
  for(int i = 0; i != n_sig; i++)
    z_pred = z_pred + Zsig.col(i) * weights_(i);
  for(int i = 0; i != n_sig; i++){
    VectorXd dev = Zsig.col(i) - z_pred;
    S = S + dev * dev.transpose() * weights_(i);
  }

  S(0, 0) += (std_laspx_ * std_laspx_);
  S(1, 1) += (std_laspy_ * std_laspy_);
  const VectorXd z = meas_package.raw_measurements_;
  double nis = GetNIS(z, z_pred, S);
  RunNisStats(nis, 5.991);

  MatrixXd Tc = MatrixXd(n_x_, n_z).setZero();
  for(int i = 0; i != n_sig; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  MatrixXd K = Tc * S.inverse();
  VectorXd zDiff = z - z_pred;
  x_ = x_ + K * (zDiff);
  P_ = P_ - (K * S * K.transpose());
//  std::cout << "x_ " << x_ << std::endl;

//  std::cout << "Done Updading with lidar" << std::endl;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

//  std::cout << "Updating with radar" << std::endl;
  int n_z = meas_package.raw_measurements_.size();
  int n_sig = 2 * n_aug_ + 1;
  MatrixXd Zsig = MatrixXd(n_z, n_sig);
  VectorXd z_pred = VectorXd(n_z).setZero();
//  std::cout << "z_pred " << z_pred << std::endl;
  MatrixXd S = MatrixXd(n_z, n_z).setZero();
  for(int i = 0; i != n_sig; i++){
    Zsig(0, i) = sqrt(pow(Xsig_pred_(0, i), 2) + pow(Xsig_pred_(1, i), 2));
    Zsig(1, i) = atan2(Xsig_pred_(1, i), Xsig_pred_(0, i));
    Zsig(2, i) = Xsig_pred_(2, i) * (Xsig_pred_(0, i) * cos(Xsig_pred_(3, i)) +
                                      Xsig_pred_(1, i) * sin(Xsig_pred_(3, i))) /
                                (Zsig(0, i) > .0001 ? Zsig(0, i) : .0001);
    z_pred = z_pred + Zsig.col(i) * weights_(i);
  }
//  std::cout << "Computed Zsig" << std::endl << Zsig << std::endl;
//  std::cout << "z_pred " << z_pred << std::endl;
  for(int i = 0; i != n_sig; i++){
    VectorXd dev = Zsig.col(i) - z_pred;
    while(dev[1] > M_PI) dev[1] -= 2. * M_PI;
    while(dev[1] < -M_PI) dev[1] += 2. * M_PI;
    S = S + dev * dev.transpose() * weights_(i);
  }

  S(0, 0) += (std_radr_ * std_radr_);
  S(1, 1) += (std_radphi_ * std_radphi_);
  S(2, 2) += (std_radrd_ * std_radrd_);

//  std::cout << "Computed S" << std::endl;
  const VectorXd z = meas_package.raw_measurements_;
  
  VectorXd z_diff = z - z_pred;
  double nis = GetNIS(z, z_pred, S);
  RunNisStats(nis, 7.815);
  
  MatrixXd Tc = MatrixXd(n_x_, n_z).setZero();
  for(int i = 0; i != n_sig; i++){
//    std::cout << "Processing Zsig column " << i << "/" << n_sig << std::endl;
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while(x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while(x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
//  std::cout << "Computed Tc" << std::endl;
  MatrixXd K = Tc * S.inverse();
  VectorXd zDiff = z - z_pred;
  x_ = x_ + K * (zDiff);
  P_ = P_ - (K * S * K.transpose());

//  std::cout << "K" << std::endl << K << std::endl;
//  std::cout << "P_" << std::endl << P_ << std::endl;
//  std::cout << "x_ " << x_ << std::endl;
//  std::cout << "Done updating radar" << std::endl;

}

void UKF::Initialize(MeasurementPackage meas_package){
  std::cout << "Initializing " << meas_package.sensor_type_ << std::endl;
  x_.setZero();
  if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    x_[0] = meas_package.raw_measurements_[0];
    x_[1] = meas_package.raw_measurements_[1];
  }else if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    float rho, theta, rho_dot;
    rho = meas_package.raw_measurements_[0];
    theta = meas_package.raw_measurements_[1];
    rho_dot = meas_package.raw_measurements_[2];
    float vert_c = sin(theta);
    float horiz_c = cos(theta);
    x_[0] = rho * horiz_c;
    x_[1] = rho * vert_c;
    if(rho < .00001) rho = .00001;
    x_[2] = rho_dot * sqrt(x_[0] * x_[0] + x_[1] * x_[1]) / (horiz_c * horiz_c + vert_c * vert_c) / rho;
    //rho_dot * sqr(px^2 + py^2) / (cos(phi)^2 + sin(phi)^2) / rho
  }
    time_us_ = meas_package.timestamp_;
    P_ << 1., 0, 0, 0, 0,
        0, 1., 0, 0, 0,
        0, 0, 1., 0, 0,
        0, 0, 0, 1., 0,
        0, 0, 0, 0, 1.;
  return;
}

double UKF::GetNIS(const VectorXd& z, const VectorXd& pred, const MatrixXd& S){
  VectorXd z_diff = z - pred;
  return (z_diff.transpose() * S.inverse() * z_diff)[0];
}

void UKF::RunNisStats(double nis, double target){
  estimate_count_++;
  if(nis > target) nis_over_count_++;
  std::cout << "NIS: " << nis << " 95% Target: " << target << (nis > target ? " Over " : " Under" ) <<
    (nis_over_count_ * 100 / estimate_count_) << "% over." << std::endl;
}
