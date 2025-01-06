//
// Created by Gyungmin Myung on 2025/01/06.
//

#pragma once

#include <Reward.hpp>
#include <raisim/raisim_message.hpp>

namespace raisim {

class MyRewards : public raisim::Reward {
public:
  MyRewards(const Yaml::Node &cfg, raisim::ArticulatedSystem *raibo) {
    initializeFromConfigurationFile(cfg);
    raibo_ = raibo;
  }

  void update() override {
    Eigen::VectorXd gc, gv;
    raibo_->getState(gc, gv);
    // Extract rotation matrix
    raisim::Vec<4> quat{gc[3], gc[4], gc[5], gc[6]};
    raisim::Mat<3, 3> rot;
    raisim::quatToRotMat(quat, rot);

    Eigen::Vector3d bodyLinearVel = rot.e().transpose() * gv.head<3>();
    Eigen::Vector3d bodyAngularVel = rot.e().transpose() * gv.segment<3>(3);

    record("torque", raibo_->getGeneralizedForce().squaredNorm());
    record("forwardVel", std::min(4.0, bodyLinearVel[0]));
  }

private:
  raisim::ArticulatedSystem *raibo_ = nullptr;
};

} // namespace raisim
