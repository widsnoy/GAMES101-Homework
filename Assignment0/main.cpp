#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>

int main() {
  float rotate = acos(-1.0) / 4.0;
  Eigen::Matrix3<float> trans;
  trans << std::cos(rotate), -std::sin(rotate), 1.0, std::sin(rotate),
      std::cos(rotate), 2.0, 0.0, 0.0, 1.0;
  Eigen::Vector3f p(2.0, 1.0, 1.0);
  std::cout << trans * p << std::endl;
  return 0;
}