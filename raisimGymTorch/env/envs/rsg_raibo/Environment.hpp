//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include "MyRewards.hpp"
#include "RaisimGymEnv.hpp"
#include <random> // For std::mt19937, std::normal_distribution

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {
public:
  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg,
                       bool visualizable)
      : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable),
        normDist_(0.0, 1.0), unifDist_(0.0, 1.0) {
    createWorld();
    addRobotAndGround();
    initializeDimensions();
    initializeInitialStates();
    configureRobot(cfg);
    setupObservationAndAction(cfg);
    setupReward(cfg);
    setupCommand(cfg);
    defineFootIndices();
    launchServerIfVisualizable();
  }

  void init() final {
    // Nothing to do here at the moment
  }

  void reset() final {
    raibo_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec> &action) final {
    applyAction(action);
    simulate();
    updateCommands();
    updateObservation();

    rewards_->update();

    return rewards_->sum();
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    // Convert observation to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float &terminalReward) final {
    terminalReward = static_cast<float>(terminalRewardCoeff_);

    // If a contact body is not one of the feet, episode terminates
    for (auto &contact : raibo_->getContacts()) {
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

  void setSeed(int seed) final { gen_.seed(seed); }

private:
  // ----------------------------------------------------------------------------
  // Helper functions
  // ----------------------------------------------------------------------------

  void createWorld() { world_ = std::make_unique<raisim::World>(); }

  void addRobotAndGround() {
    raibo_ =
        world_->addArticulatedSystem(resourceDir_ + "/raibo2/urdf/raibo2.urdf");
    raibo_->setName("raibo");
    raibo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();
  }

  void initializeDimensions() {
    gcDim_ = raibo_->getGeneralizedCoordinateDim();
    gvDim_ = raibo_->getDOF();
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

    raibo_->setPdGains(jointPgain, jointDgain);
    raibo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
  }

  void setupObservationAndAction(const Yaml::Node &cfg) {
    // Must be done for all environments
    // 2 * 12 joints + 3 lin_vel + 3 ang_vel + 3 z_axis + 1 height + 3 commands
    obDim_ = 37;
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
    rewards_ = std::make_unique<MyRewards>(cfg["reward"], raibo_, commands_);
  }

  void setupCommand(const Yaml::Node &cfg) {
    command_interval_ = cfg["command_interval"].As<double>();
  }

  void updateCommands() {
    command_counter_ += 1;
    if (command_length_ < command_counter_) {
      commands_[0] = (unifDist_(gen_) - 0.5) * 2; // x
      commands_[1] = unifDist_(gen_);             // y
      commands_[2] = (unifDist_(gen_) - 0.5) * 2; // z
      command_counter_ = 0;
      command_length_ = command_interval_ / control_dt_ * unifDist_(gen_) * 2;
    }
  }

  void defineFootIndices() {
    footIndices_.insert(raibo_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(raibo_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(raibo_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(raibo_->getBodyIdx("RH_SHANK"));
  }

  void launchServerIfVisualizable() {
    if (!visualizable_)
      return;
    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
    server_->launchServer();
    server_->focusOn(raibo_);
  }

  void applyAction(const Eigen::Ref<EigenVec> &action) {
    // Scale and shift the action
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_) + actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    raibo_->setPdTarget(pTarget_, vTarget_);
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
    raibo_->getState(gc_, gv_);

    // Extract rotation matrix
    raisim::Vec<4> quat{gc_[3], gc_[4], gc_[5], gc_[6]};
    raisim::Mat<3, 3> rot;
    raisim::quatToRotMat(quat, rot);

    // Compute body velocity in body frame
    bodyLinearVel_ = rot.e().transpose() * gv_.head<3>();
    bodyAngularVel_ = rot.e().transpose() * gv_.segment<3>(3);
    Eigen::Vector3d commands(commands_[0], commands_[1], commands_[2]);

    obDouble_ << gc_[2],            // body height
        rot.e().row(2).transpose(), // body orientation (z-axis)
        bodyLinearVel_,             // body linear velocity
        bodyAngularVel_,            // body angular velocity
        gv_.tail(12),               // joint velocities
        commands;                   // commands
  }

  // ----------------------------------------------------------------------------
  // Member variables
  // ----------------------------------------------------------------------------
private:
  // Robot, environment
  raisim::ArticulatedSystem *raibo_{nullptr};
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

  // Body velocities
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;

  // Foot indices
  std::set<size_t> footIndices_;

  // Command change counter
  size_t command_counter_{0};

  // Reward
  double terminalRewardCoeff_{-10.0};

  // Commands
  std::array<double, 3> commands_{0.0, 0.0, 0.0};
  double command_interval_{0.0};
  size_t command_length_{0};

  // Random number generator
  std::normal_distribution<double> normDist_;
  std::uniform_real_distribution<double> unifDist_;
  thread_local static std::mt19937 gen_;
};

thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

} // namespace raisim
