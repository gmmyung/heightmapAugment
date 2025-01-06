//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include "../../RaisimGymEnv.hpp"
#include <random> // For std::mt19937, std::normal_distribution
#include <set>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {
public:
  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg,
                       bool visualizable)
      : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable),
        normDist_(0.0, 1.0) {

    createWorld();
    addRobotAndGround();
    initializeDimensions();
    initializeInitialStates();
    configureRobot(cfg);
    setupObservationAndAction(cfg);
    setupReward(cfg);
    defineFootIndices();
    launchServerIfVisualizable();
  }

  void init() final {
    // Nothing to do here at the moment
  }

  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec> &action) final {
    applyAction(action);
    simulate();
    updateObservation();

    // Record rewards
    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards_.record("forwardVel", std::min(4.0, bodyLinearVel_[0]));

    return rewards_.sum();
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    // Convert observation to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float &terminalReward) final {
    terminalReward = static_cast<float>(terminalRewardCoeff_);

    // If a contact body is not one of the feet, episode terminates
    for (auto &contact : anymal_->getContacts()) {
      if (footIndices_.find(contact.getlocalBodyIndex()) ==
          footIndices_.end()) {
        return true;
      }
    }

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() final {
    // No curriculum strategy for now
  }

private:
  // ----------------------------------------------------------------------------
  // Helper functions
  // ----------------------------------------------------------------------------

  void createWorld() { world_ = std::make_unique<raisim::World>(); }

  void addRobotAndGround() {
    anymal_ =
        world_->addArticulatedSystem(resourceDir_ + "/raibo2/urdf/raibo2.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();
  }

  void initializeDimensions() {
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;
  }

  void initializeInitialStates() {
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);

    // Nominal configuration of Raibo2
    gc_init_ << 0, 0, 0.481, 1.0, 0.0, 0.0, 0.0, 0, 0.580099, -1.195, 0,
        0.580099, -1.195, 0, 0.580099, -1.195, 0, 0.580099, -1.195;
  }

  void configureRobot(const Yaml::Node &cfg) {
    // Set PD gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointDgain.setZero();

    double joint_p_gain, joint_d_gain;
    READ_YAML(double, joint_p_gain, cfg["joint_p_gain"]);
    READ_YAML(double, joint_d_gain, cfg["joint_d_gain"]);

    jointPgain.tail(nJoints_).setConstant(joint_p_gain);
    jointDgain.tail(nJoints_).setConstant(joint_d_gain);

    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
  }

  void setupObservationAndAction(const Yaml::Node &cfg) {
    // Must be done for all environments
    obDim_ = 34;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    // Action scaling
    actionMean_ = gc_init_.tail(nJoints_);

    double action_std;
    READ_YAML(double, action_std, cfg["action_std"]); // example usage
    actionStd_.setConstant(action_std);
  }

  void setupReward(const Yaml::Node &cfg) {
    // Reward coefficients
    rewards_.initializeFromConfigurationFile(cfg["reward"]);
  }

  void defineFootIndices() {
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));
  }

  void launchServerIfVisualizable() {
    if (!visualizable_)
      return;
    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
    server_->launchServer();
    server_->focusOn(anymal_);
  }

  void applyAction(const Eigen::Ref<EigenVec> &action) {
    // Scale and shift the action
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_) + actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);
  }

  void simulate() {
    // Step the simulation for control_dt_ / simulation_dt_ steps
    const int steps = static_cast<int>(control_dt_ / simulation_dt_ + 1e-10);
    for (int i = 0; i < steps; ++i) {
      if (server_)
        server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_)
        server_->unlockVisualizationServerMutex();
    }
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);

    // Extract rotation matrix
    raisim::Vec<4> quat{gc_[3], gc_[4], gc_[5], gc_[6]};
    raisim::Mat<3, 3> rot;
    raisim::quatToRotMat(quat, rot);

    // Compute body velocity in body frame
    bodyLinearVel_ = rot.e().transpose() * gv_.head<3>();
    bodyAngularVel_ = rot.e().transpose() * gv_.segment<3>(3);

    obDouble_ << gc_[2],            // body height
        rot.e().row(2).transpose(), // body orientation (z-axis)
        gc_.tail(12),               // joint angles
        bodyLinearVel_,             // body linear velocity
        bodyAngularVel_,            // body angular velocity
        gv_.tail(12);               // joint velocities
  }

  // ----------------------------------------------------------------------------
  // Member variables
  // ----------------------------------------------------------------------------
private:
  // Robot, environment
  raisim::ArticulatedSystem *anymal_{nullptr};
  bool visualizable_{false};

  // Dimensions
  int gcDim_{0}, gvDim_{0}, nJoints_{0};

  // Initial and current states
  Eigen::VectorXd gc_init_, gv_init_;
  Eigen::VectorXd gc_, gv_;

  // PD targets
  Eigen::VectorXd pTarget_, pTarget12_, vTarget_;

  // Action scaling
  Eigen::VectorXd actionMean_, actionStd_;

  // Observations
  Eigen::VectorXd obDouble_;
  int obDim_{0}, actionDim_{0};

  // Body velocities
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;

  // Foot indices
  std::set<size_t> footIndices_;

  // Reward
  double terminalRewardCoeff_{-10.0};

  // Random number generator
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};

// Definition of the static thread_local generator
thread_local std::mt19937 ENVIRONMENT::gen_;

} // namespace raisim
