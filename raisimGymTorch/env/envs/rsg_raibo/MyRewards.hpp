//
// Created by Gyungmin Myung on 2025/01/06.
//

#pragma once

#include <Reward.hpp>
#include <raisim/raisim_message.hpp>

namespace raisim {

class MyRewards : public raisim::Reward {
public:
  MyRewards(const Yaml::Node &cfg, raisim::ArticulatedSystem *raibo,
            std::array<double, 3> &commands, std::set<size_t> &foot_indices,
            const double control_dt)
      : raibo_(raibo), commands_(commands), control_dt_(control_dt),
        foot_indices_(foot_indices.begin(), foot_indices.end()),
        contacts_(foot_indices_.size(), false),
        first_contacts_(foot_indices_.size(), false),
        air_times_(foot_indices_.size(), 0.) {
    initializeFromConfigurationFile(cfg);
    setupReward(cfg);
  }

  void update() override {
    Eigen::VectorXd gc, gv;
    raibo_->getState(gc, gv);

    ///// Torque /////
    record("torque", raibo_->getGeneralizedForce().squaredNorm());

    raisim::Vec<4> quat{gc[3], gc[4], gc[5], gc[6]};
    raisim::Mat<3, 3> rot;
    raisim::quatToRotMat(quat, rot);

    Eigen::Vector3d bodyLinearVel = rot.e().transpose() * gv.head<3>();
    Eigen::Vector3d bodyAngularVel = rot.e().transpose() * gv.segment<3>(3);

    ///// Command Velocity /////
    double target_vel_x = commands_[0] * command_max_x_;
    double target_vel_y = commands_[1] * command_max_y_;
    double target_vel_ang = commands_[2] * command_max_ang_;

    double error_x = bodyLinearVel[0] - target_vel_x;
    double error_y = bodyLinearVel[1] - target_vel_y;
    double error_ang = bodyAngularVel[2] - target_vel_ang;
    double error = std::exp(-std::pow(error_x, 2)) +
                   std::exp(-std::pow(error_y, 2)) +
                   std::exp(-std::pow(error_ang, 2));

    record("command_vel_error", error);

    ///// Feet Air Time /////
    auto contacts = raibo_->getContacts();
    for (const auto &contact : contacts) {
      int index = contact.getlocalBodyIndex();
      // Update foot contact status
      for (size_t i = 0; i < foot_indices_.size(); ++i) {
        if (foot_indices_[i] == index) {
          contacts_[i] = true;
          break;
        } else {
          contacts_[i] = false;
        }
      }
    }

    double air_time_sum{0.};

    for (size_t i = 0; i < foot_indices_.size(); ++i) {
      first_contacts_[i] = air_times_[i] != 0. && contacts_[i];
      if (first_contacts_[i]) {
        air_time_sum += air_times_[i] - 0.5;
      }
      if (contacts_[i]) {
        air_times_[i] = 0.;
      } else {
        air_times_[i] += control_dt_;
      }
    }
    record("air_time", air_time_sum);
  }

  void getFootHolds(Eigen::Ref<Eigen::VectorXf> foot_holds) override {
    for (size_t i = 0; i < foot_indices_.size(); ++i) {
      if (first_contacts_[i]) {
        raisim::Vec<3> foot_pos;
        raibo_->getBodyPosition(foot_indices_[i], foot_pos);
        foot_holds.segment<3>(3 * i) << foot_pos[0], foot_pos[1], foot_pos[2];
      } else {
        foot_holds.segment<3>(3 * i) << 0, 0, 0;
      }
    }
  }

private:
  raisim::ArticulatedSystem *raibo_{nullptr};
  double control_dt_;
  const std::array<double, 3> &commands_;
  const std::vector<size_t> foot_indices_;
  std::vector<bool> contacts_;
  std::vector<bool> first_contacts_;
  std::vector<double> air_times_;
  double command_max_x_, command_max_y_, command_max_ang_;

  void setupReward(const Yaml::Node &cfg) {
    command_max_x_ = cfg["command_vel_error"]["max_x"].As<double>();
    command_max_y_ = cfg["command_vel_error"]["max_y"].As<double>();
    command_max_ang_ = cfg["command_vel_error"]["max_ang"].As<double>();
  }
};

} // namespace raisim
