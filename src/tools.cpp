#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()  || estimations.size() == 0){
      std::cout << "Invalid data format" << std::endl;
      return rmse;
  }

  //accumulate squared residuals
  for (int i = 0; i < estimations.size(); ++i){

      VectorXd residual = estimations[i] - ground_truth[i];
      //coefficient-wise multiplication
      residual = residual.array()*residual.array();
      rmse += residual;
  }

  //calculate the mean
  rmse = rmse / estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Jacobian(3, 4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute to speed up
  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = (c1 * c2);
  float c4 = px / c2;
  float c5 = py / c2;
  float c6 = vx * py;
  float c7 = vy * px;

  //check division by zero
  if (fabs(c1) < 0.00001){
      std::cout << "Division by zero error" << std::endl;
      return Jacobian;
  }

  //compute the Jacobian matrix
  Jacobian << c4, c5, 0, 0,
        -(py / c1), (px / c1), 0, 0,
        py*(c6 - c7) / c3, px*(c7 - c6) / c3, c4, c5;

  return Jacobian;
}
