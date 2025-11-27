#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/Dense>
struct Quaternion {
    double q_w;
    double q_x;
    double q_y;
    double q_z;
};
struct PosTraj {
    std::vector<double> pos_x;
    std::vector<double> pos_y;
    std::vector<double> pos_z;
};
struct RotTraj {
    std::vector<double> quat_w;
    std::vector<double> quat_x;
    std::vector<double> quat_y;
    std::vector<double> quat_z;
};
struct AngularVelocity {
    std::vector<double> omega_x;
    std::vector<double> omega_y;
    std::vector<double> omega_z;
};
struct AngularAcceleration {
    std::vector<double> omega_dot_x;
    std::vector<double> omega_dot_y;
    std::vector<double> omega_dot_z;
};
struct Trajectory {
    std::vector<double> time_us;
    PosTraj position;
    RotTraj orientation;
    AngularVelocity angular_velocity;    
    AngularAcceleration angular_acceleration;  
};
class DMP {
public:
    enum class Type {
        Standard,
        Corrected
    };
    DMP(int n_basis_ = 100,
        Type type_ = Type::Standard,
        double alpha_z_ = 60.0,
        double beta_z_ = 15.0,
        double alpha_x_ = 4.0,
        double tau_ = 1.0,
        double dt_ = 0.01)
        : n_basis(n_basis_),
          type(type_),
          alpha_z(alpha_z_),
          beta_z(beta_z_),
          alpha_x(alpha_x_),
          dt(dt_),
          tau(tau_)
    {
        centers.resize(n_basis);
        widths.resize(n_basis);
        initializeBasisFunctions();
        weights_px.assign(n_basis, 0.0);
        weights_py.assign(n_basis, 0.0);
        weights_pz.assign(n_basis, 0.0);
        weights_rx.assign(n_basis, 0.0);
        weights_ry.assign(n_basis, 0.0);
        weights_rz.assign(n_basis, 0.0);
    }
    void train(const Trajectory& demo_traj);
    void rollout(const std::vector<double>& p_0,
                 const Quaternion& q_0,
                 const std::vector<double>& p_goal,
                 const Quaternion& q_goal,
                 double time_factor = 1.0);
    Trajectory get_demo_trajectory() const {
        return demo_trajectory;
    }
    Trajectory get_rollout_trajectory() const {
        return rollout_trajectory;
    }
    Trajectory read_trajectory(const std::string& csv_path);
    std::vector<double> get_centers() const { return centers; }
    std::vector<double> get_widths() const { return widths; }
    std::vector<double> get_weights_px() const { return weights_px; }
    std::vector<double> get_weights_py() const { return weights_py; }
    std::vector<double> get_weights_pz() const { return weights_pz; }
    std::vector<double> get_weights_rx() const { return weights_rx; }
    std::vector<double> get_weights_ry() const { return weights_ry; }
    std::vector<double> get_weights_rz() const { return weights_rz; }
    void set_dt(double new_dt) { dt = new_dt; }
    void set_tau(double new_tau) { tau = new_tau; } 
    void set_n_basis(int new_n_basis) { 
        n_basis = new_n_basis; 
        initializeBasisFunctions(); 
    }
    void set_type(Type new_type) { type = new_type; }
    void set_alpha_z(double new_alpha_z) { alpha_z = new_alpha_z; } 
    void set_beta_z(double new_beta_z) { beta_z = new_beta_z; } 
    void set_alpha_x(double new_alpha_x) { alpha_x = new_alpha_x; } 
private:
    int n_basis;
    Type type;
    double alpha_z, beta_z;
    double alpha_x;
    double dt;
    double tau;
    std::vector<double> weights_px;
    std::vector<double> weights_py;
    std::vector<double> weights_pz;
    std::vector<double> weights_rx;
    std::vector<double> weights_ry;
    std::vector<double> weights_rz;
    std::vector<double> centers;
    std::vector<double> widths;
    Trajectory demo_trajectory;
    Trajectory rollout_trajectory;
    void initializeBasisFunctions();
    void computeDerivatives(const std::vector<double>& y_demo, 
                           std::vector<double>& dy_demo, 
                           std::vector<double>& ddy_demo) const;
    double phase_update(double x_prev, double dt) const;
    double basis_function(double x, int i) const;
    double interpolate(double t, 
                      const std::vector<double>& time_vec, 
                      const std::vector<double>& value_vec) const;
    Eigen::Matrix3d quaternionToRotationMatrix(double qw, double qx, double qy, double qz) const;
    Eigen::Vector3d extractOmegaFromSkewSymmetric(const Eigen::Matrix3d& omega_hat) const;
    void train_translation(const Trajectory& demo_traj);
    void train_orientation(const Trajectory& demo_traj);
    void train_orientation_corrected(const Trajectory& demo_traj);
    std::vector<std::vector<double>> rollout_translation(
        const std::vector<double>& p_0,
        const std::vector<double>& g,
        double time_factor);
    std::vector<Quaternion> rollout_orientation(
        const Quaternion& q_0, 
        const Quaternion& q_goal, 
        double time_factor);
    std::vector<Quaternion> rollout_orientation_corrected(const Quaternion& q_0, const Quaternion& q_goal, double time_factor);
    void save_translation_rollout_csv(const std::vector<std::vector<double>>& pos_traj,
                                      const std::string& output_path);
    void save_full_rollout_csv(const std::vector<double>& pos_traj, 
                               const std::vector<Quaternion>& quat_traj, 
                               const std::string& output_path);
    void save_rollout_csv(const std::vector<double>& rollout_traj, 
                          const std::string& output_path);
};
Quaternion Quat_conjugate(const Quaternion& Q1);
Quaternion Quat_product(const Quaternion& Q1, const Quaternion& Q2);
std::vector<double> Quat_log(const Quaternion& q);
Quaternion Quat_exp(const std::vector<double>& r);
Eigen::MatrixXd Quat_Jacobian(const Quaternion& Q);
Eigen::MatrixXd Quat_Log_Jacobian(const Quaternion& Q);