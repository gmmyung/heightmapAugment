//
// Created by Gyungmin Myung on 2025/01/06.
//

#pragma once

#include <Reward.hpp>
#include <algorithm>
#include <raisim/raisim_message.hpp>

namespace raisim {

class MyRewards : public raisim::Reward {
public:
  MyRewards(const Yaml::Node &cfg, raisim::ArticulatedSystem *raibo,
            std::array<double, 3> &commands)
      : raibo_(raibo), commands_(commands) {
    initializeFromConfigurationFile(cfg);
    setupReward(cfg);
  }

  void update() override {
    Eigen::VectorXd gc, gv;
    raibo_->getState(gc, gv);

    //////////// Torque //////////
    record("torque", raibo_->getGeneralizedForce().squaredNorm());

    raisim::Vec<4> quat{gc[3], gc[4], gc[5], gc[6]};
    raisim::Mat<3, 3> rot;
    raisim::quatToRotMat(quat, rot);

    Eigen::Vector3d bodyLinearVel = rot.e().transpose() * gv.head<3>();
    Eigen::Vector3d bodyAngularVel = rot.e().transpose() * gv.segment<3>(3);

    ////////// Command Velocity //////////

    double target_vel_x = commands_[0] * command_max_x_;
    double target_vel_y = commands_[1] * command_max_y_;
    double target_vel_ang = commands_[2] * command_max_y_;

    double error_x = bodyLinearVel[0] - target_vel_x;
    double error_y = bodyLinearVel[1] - target_vel_y;
    double error_ang = bodyAngularVel[2] - target_vel_ang;
    double error = std::exp(-std::pow(error_x, 2)) +
                   std::exp(-std::pow(error_y, 2)) +
                   std::exp(-std::pow(error_ang, 2));

    record("command_vel_error", error);
  }

private:
  raisim::ArticulatedSystem *raibo_ = nullptr;
  const std::array<double, 3> &commands_;
  double command_max_x_, command_max_y_, command_max_ang_;

  void setupReward(const Yaml::Node &cfg) {
    command_max_x_ = cfg["command_vel_error"]["max_x"].As<double>();
    command_max_y_ = cfg["command_vel_error"]["max_y"].As<double>();
    command_max_ang_ = cfg["command_vel_error"]["max_ang"].As<double>();
  }
};

} // namespace raisim
