#include "dmplib/dmp.hpp"
#include <algorithm>
#include <Eigen/Dense>
#include <stdexcept>
Eigen::MatrixXd Quat_Log_Jacobian(const Quaternion &Q)
{
    Eigen::MatrixXd J_log_q(4, 3);
    double qw_clamped = std::max(-1.0, std::min(1.0, Q.q_w));
    double theta = std::acos(qw_clamped);
    const double epsilon = 1e-6;
    if (theta < epsilon)
    {
        J_log_q.row(0) << 0.0, 0.0, 0.0;
        J_log_q.block<3, 3>(1, 0) = Eigen::Matrix3d::Identity();
    }
    else
    {
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        Eigen::Vector3d n;
        n << Q.q_x / sin_theta,
            Q.q_y / sin_theta,
            Q.q_z / sin_theta;
        J_log_q.row(0) = -sin_theta * n.transpose();
        Eigen::Matrix3d n_nT = n * n.transpose();
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        J_log_q.block<3, 3>(1, 0) = (sin_theta / theta) * (I - n_nT) + cos_theta * n_nT;
    }
    return J_log_q;
}
Eigen::MatrixXd Quat_Jacobian(const Quaternion &Q)
{
    Eigen::MatrixXd J_q(3, 4);
    double qw_clamped = std::max(-1.0, std::min(1.0, Q.q_w));
    double theta = std::acos(qw_clamped);
    const double epsilon = 1e-6;
    if (theta < epsilon)
    {
        J_q << 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
    }
    else
    {
        double sin_theta = std::sin(theta);
        Eigen::Vector3d n;
        n << Q.q_x / sin_theta, Q.q_y / sin_theta, Q.q_z / sin_theta;
        double first_col_scale = (-sin_theta + theta * std::cos(theta)) / (sin_theta * sin_theta);
        double remaining_scale = theta / sin_theta;
        J_q.col(0) = first_col_scale * n;
        J_q.block<3, 3>(0, 1) = remaining_scale * Eigen::Matrix3d::Identity();
    }
    return J_q;
}
Quaternion Quat_conjugate(const Quaternion &Q1)
{
    return {Q1.q_w, -Q1.q_x, -Q1.q_y, -Q1.q_z};
}
Quaternion Quat_product(const Quaternion &Q1, const Quaternion &Q2)
{
    double w1 = Q1.q_w, x1 = Q1.q_x, y1 = Q1.q_y, z1 = Q1.q_z;
    double w2 = Q2.q_w, x2 = Q2.q_x, y2 = Q2.q_y, z2 = Q2.q_z;
    double w_res = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    double x_res = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    double y_res = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    double z_res = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    return {w_res, x_res, y_res, z_res};
}
std::vector<double> Quat_log(const Quaternion &q)
{
    std::vector<double> log_q(3, 0.0);
    double qx = q.q_x;
    double qy = q.q_y;
    double qz = q.q_z;
    double v_norm = std::sqrt(qx * qx + qy * qy + qz * qz);
    if (v_norm < 1e-6)
    {
        return log_q;
    }
    else
    {
        if (q.q_w > 1.0)
            std::cout << "Warning: improper quaternion > 1.0\n";
        double qw_clamped = std::max(-1.0, std::min(1.0, q.q_w));
        double theta = std::acos(qw_clamped);
        double scale_factor = theta / v_norm;
        log_q[0] = qx * scale_factor;
        log_q[1] = qy * scale_factor;
        log_q[2] = qz * scale_factor;
        return log_q;
    }
}
Quaternion Quat_exp(const std::vector<double> &r)
{
    if (r.size() != 3)
    {
        throw std::invalid_argument("Input vector 'r' must have 3 elements for quaternion exponential.");
    }
    double rx = r[0];
    double ry = r[1];
    double rz = r[2];
    double norm_r = std::sqrt(rx * rx + ry * ry + rz * rz);
    const double epsilon = 1e-6;
    if (norm_r < epsilon)
    {
        return {1.0, 0.0, 0.0, 0.0};
    }
    else
    {
        double qw = std::cos(norm_r);
        double scale_factor = std::sin(norm_r) / norm_r;
        double qx = rx * scale_factor;
        double qy = ry * scale_factor;
        double qz = rz * scale_factor;
        return {qw, qx, qy, qz};
    }
}
void DMP::initializeBasisFunctions()
{
    centers.resize(n_basis);
    widths.resize(n_basis);
    for (int i = 0; i < n_basis; ++i)
    {
        double t_i = static_cast<double>(i) / (n_basis - 1.0);
        centers[i] = std::exp(-alpha_x * t_i);
    }
    std::reverse(centers.begin(), centers.end());
    std::cout << "RBF Centers initialized.\n";
    for (int i = 0; i < n_basis; ++i)
    {
        if (i < n_basis - 1)
        {
            double delta_c = centers[i + 1] - centers[i];
            widths[i] = 1.0 / std::pow(delta_c * 0.5, 2);
        }
        else
        {
            widths[i] = widths[i - 1];
        }
    }
    std::cout << "RBF Widths initialized based on center spacing.\n";
}
void DMP::computeDerivatives(const std::vector<double> &y, std::vector<double> &dy, std::vector<double> &ddy) const
{
    int n_steps = y.size();
    dy.assign(n_steps, 0.0);
    ddy.assign(n_steps, 0.0);
    for (int i = 1; i < n_steps - 1; ++i)
    {
        dy[i] = (y[i + 1] - y[i - 1]) / (2.0 * dt);
    }
    if (n_steps >= 2)
    {
        dy[0] = (y[1] - y[0]) / dt;
        dy[n_steps - 1] = (y[n_steps - 1] - y[n_steps - 2]) / dt;
    }
    for (int i = 1; i < n_steps - 1; ++i)
    {
        ddy[i] = (dy[i + 1] - dy[i - 1]) / (2.0 * dt);
    }
}
double DMP::phase_update(double x_prev, double dt) const
{
    double dx_dt = -alpha_x / tau * x_prev;
    double x_next = x_prev + dx_dt * dt;
    return std::max(0.0, x_next);
}
double DMP::basis_function(double x, int i) const
{
    return std::exp(-widths[i] * std::pow(x - centers[i], 2));
}
void DMP::train_translation(const Trajectory &demo_traj)
{
    if (demo_traj.position.pos_x.empty() ||
        demo_traj.position.pos_y.empty() ||
        demo_traj.position.pos_z.empty())
    {
        std::cerr << "Error: Incomplete trajectory data for translation training.\n";
        return;
    }
    int N_steps = demo_traj.position.pos_x.size();
    if (N_steps < 3)
    {
        std::cerr << "Error: Trajectory too short for training.\n";
        return;
    }
    tau = N_steps * dt;
    std::cout << "DMP Translation training: tau set to " << tau << " seconds.\n";
    double y0_x = demo_traj.position.pos_x[0];
    double g_x = demo_traj.position.pos_x[N_steps - 1];
    double amplitude_x = g_x - y0_x;
    double y0_y = demo_traj.position.pos_y[0];
    double g_y = demo_traj.position.pos_y[N_steps - 1];
    double amplitude_y = g_y - y0_y;
    double y0_z = demo_traj.position.pos_z[0];
    double g_z = demo_traj.position.pos_z[N_steps - 1];
    double amplitude_z = g_z - y0_z;
    std::vector<double> dy_x, ddy_x;
    std::vector<double> dy_y, ddy_y;
    std::vector<double> dy_z, ddy_z;
    computeDerivatives(demo_traj.position.pos_x, dy_x, ddy_x);
    computeDerivatives(demo_traj.position.pos_y, dy_y, ddy_y);
    computeDerivatives(demo_traj.position.pos_z, dy_z, ddy_z);
    Eigen::VectorXd f_target_scaled_x(N_steps);
    Eigen::VectorXd f_target_scaled_y(N_steps);
    Eigen::VectorXd f_target_scaled_z(N_steps);
    std::vector<double> x_track(N_steps);
    double x = 1.0;
    double tau_sq = std::pow(tau, 2);
    for (int t = 0; t < N_steps; ++t)
    {
        x_track[t] = x;
        double y_t_x = demo_traj.position.pos_x[t];
        double dy_t_x = dy_x[t];
        double ddy_t_x = ddy_x[t];
        double f_target_x = tau_sq * ddy_t_x - alpha_z * (beta_z * (g_x - y_t_x) - tau * dy_t_x);
        f_target_scaled_x(t) = (std::abs(amplitude_x) < 1e-6) ? 0.0 : f_target_x / amplitude_x;
        double y_t_y = demo_traj.position.pos_y[t];
        double dy_t_y = dy_y[t];
        double ddy_t_y = ddy_y[t];
        double f_target_y = tau_sq * ddy_t_y - alpha_z * (beta_z * (g_y - y_t_y) - tau * dy_t_y);
        f_target_scaled_y(t) = (std::abs(amplitude_y) < 1e-6) ? 0.0 : f_target_y / amplitude_y;
        double y_t_z = demo_traj.position.pos_z[t];
        double dy_t_z = dy_z[t];
        double ddy_t_z = ddy_z[t];
        double f_target_z = tau_sq * ddy_t_z - alpha_z * (beta_z * (g_z - y_t_z) - tau * dy_t_z);
        f_target_scaled_z(t) = (std::abs(amplitude_z) < 1e-6) ? 0.0 : f_target_z / amplitude_z;
        x = phase_update(x, dt);
    }
    Eigen::MatrixXd Psi_norm(N_steps, n_basis);
    for (int t = 0; t < N_steps; ++t)
    {
        double x_t = x_track[t];
        double sum_psi = 0.0;
        for (int j = 0; j < n_basis; ++j)
        {
            sum_psi += basis_function(x_t, j);
        }
        for (int i = 0; i < n_basis; ++i)
        {
            double psi_t = basis_function(x_t, i);
            double gated_psi = psi_t * x_t; 
            Psi_norm(t, i) = gated_psi / sum_psi;
        }
    }
    Eigen::VectorXd weights_x_eig = Psi_norm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f_target_scaled_x);
    weights_px.assign(weights_x_eig.data(), weights_x_eig.data() + weights_x_eig.size());
    Eigen::VectorXd weights_y_eig = Psi_norm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f_target_scaled_y);
    weights_py.assign(weights_y_eig.data(), weights_y_eig.data() + weights_y_eig.size());
    Eigen::VectorXd weights_z_eig = Psi_norm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f_target_scaled_z);
    weights_pz.assign(weights_z_eig.data(), weights_z_eig.data() + weights_z_eig.size());
    std::cout << "DMP Translation Training Complete. " << n_basis << " weights learned for X, Y, Z.\n";
}
std::vector<std::vector<double>> DMP::rollout_translation(
    const std::vector<double> &p_0, 
    const std::vector<double> &g,   
    double time_factor)
{
    if (p_0.size() != 3 || g.size() != 3)
    {
        throw std::invalid_argument("Start and goal positions must have 3 elements (x, y, z)");
    }
    tau = tau * time_factor;
    int num_steps = static_cast<int>(tau / dt);
    std::vector<std::vector<double>> pos_out(3);
    pos_out[0].resize(num_steps); 
    pos_out[1].resize(num_steps); 
    pos_out[2].resize(num_steps); 
    std::vector<double> y = {p_0[0], p_0[1], p_0[2]};
    std::vector<double> dy = {0.0, 0.0, 0.0};
    std::vector<double> ddy = {0.0, 0.0, 0.0};
    double x = 1.0; 
    pos_out[0][0] = y[0];
    pos_out[1][0] = y[1];
    pos_out[2][0] = y[2];
    for (int t = 1; t < num_steps; ++t)
    {
        x = phase_update(x, dt);
        for (int axis = 0; axis < 3; ++axis)
        {
            double psi_sum = 0.0;
            double psi_weighted_sum = 0.0;
            const std::vector<double> *weights_axis;
            if (axis == 0)
                weights_axis = &weights_px;
            else if (axis == 1)
                weights_axis = &weights_py;
            else
                weights_axis = &weights_pz;
            for (int i = 0; i < n_basis; ++i)
            {
                double psi = basis_function(x, i);
                psi_sum += psi;
                psi_weighted_sum += psi * (*weights_axis)[i];
            }
            double f = 0.0;
            if (psi_sum > 1e-10)
            {
                f = psi_weighted_sum / psi_sum;
            }
            double amplitude = g[axis] - p_0[axis];
            double F = f * x * amplitude;
            ddy[axis] = (alpha_z * (beta_z * (g[axis] - y[axis]) - tau * dy[axis]) + F) / (tau * tau);
            dy[axis] = dy[axis] + ddy[axis] * dt;
            y[axis] = y[axis] + dy[axis] * dt;
            pos_out[axis][t] = y[axis];
        }
    }
    return pos_out;
}
double DMP::interpolate(double t, const std::vector<double> &time_vec,
                        const std::vector<double> &value_vec) const
{
    if (time_vec.size() != value_vec.size() || time_vec.empty())
    {
        throw std::runtime_error("Invalid input vectors for interpolation");
    }
    if (t <= time_vec.front())
        return value_vec.front();
    if (t >= time_vec.back())
        return value_vec.back();
    for (size_t i = 0; i < time_vec.size() - 1; ++i)
    {
        if (t >= time_vec[i] && t <= time_vec[i + 1])
        {
            double t0 = time_vec[i];
            double t1 = time_vec[i + 1];
            double v0 = value_vec[i];
            double v1 = value_vec[i + 1];
            double alpha = (t - t0) / (t1 - t0);
            return v0 + alpha * (v1 - v0);
        }
    }
    return value_vec.back();
}
Eigen::Matrix3d DMP::quaternionToRotationMatrix(double qw, double qx, double qy, double qz) const
{
    Eigen::Matrix3d R;
    double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    qw /= norm;
    qx /= norm;
    qy /= norm;
    qz /= norm;
    R(0, 0) = 1 - 2 * (qy * qy + qz * qz);
    R(0, 1) = 2 * (qx * qy - qw * qz);
    R(0, 2) = 2 * (qx * qz + qw * qy);
    R(1, 0) = 2 * (qx * qy + qw * qz);
    R(1, 1) = 1 - 2 * (qx * qx + qz * qz);
    R(1, 2) = 2 * (qy * qz - qw * qx);
    R(2, 0) = 2 * (qx * qz - qw * qy);
    R(2, 1) = 2 * (qy * qz + qw * qx);
    R(2, 2) = 1 - 2 * (qx * qx + qy * qy);
    return R;
}
Eigen::Vector3d DMP::extractOmegaFromSkewSymmetric(const Eigen::Matrix3d &omega_hat) const
{
    Eigen::Vector3d omega;
    omega(0) = omega_hat(2, 1);
    omega(1) = omega_hat(0, 2);
    omega(2) = omega_hat(1, 0);
    return omega;
}
Trajectory DMP::read_trajectory(const std::string &csv_path)
{
    double dt_seconds = dt;
    std::ifstream file(csv_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + csv_path);
    }
    Trajectory original_traj;
    std::string header;
    std::getline(file, header);
    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (std::getline(ss, value, ','))
        {
            row.push_back(std::stod(value));
        }
        if (row.size() == 8)
        {
            original_traj.time_us.push_back(row[0]);
            original_traj.position.pos_x.push_back(row[1]);
            original_traj.position.pos_y.push_back(row[2]);
            original_traj.position.pos_z.push_back(row[3]);
            original_traj.orientation.quat_w.push_back(row[4]);
            original_traj.orientation.quat_x.push_back(row[5]);
            original_traj.orientation.quat_y.push_back(row[6]);
            original_traj.orientation.quat_z.push_back(row[7]);
        }
    }
    file.close();
    if (original_traj.time_us.empty())
    {
        throw std::runtime_error("No data read from CSV file");
    }
    double dt_us = dt_seconds * 1e6;
    Trajectory resampled_traj;
    double t_start = original_traj.time_us.front();
    double t_end = original_traj.time_us.back();
    for (double t = t_start; t <= t_end; t += dt_us)
    {
        resampled_traj.time_us.push_back(t);
        resampled_traj.position.pos_x.push_back(
            interpolate(t, original_traj.time_us, original_traj.position.pos_x));
        resampled_traj.position.pos_y.push_back(
            interpolate(t, original_traj.time_us, original_traj.position.pos_y));
        resampled_traj.position.pos_z.push_back(
            interpolate(t, original_traj.time_us, original_traj.position.pos_z));
        resampled_traj.orientation.quat_w.push_back(
            interpolate(t, original_traj.time_us, original_traj.orientation.quat_w));
        resampled_traj.orientation.quat_x.push_back(
            interpolate(t, original_traj.time_us, original_traj.orientation.quat_x));
        resampled_traj.orientation.quat_y.push_back(
            interpolate(t, original_traj.time_us, original_traj.orientation.quat_y));
        resampled_traj.orientation.quat_z.push_back(
            interpolate(t, original_traj.time_us, original_traj.orientation.quat_z));
    }
    int N = resampled_traj.time_us.size();
    std::cout << "Computing angular velocities and accelerations..." << std::endl;
    resampled_traj.angular_velocity.omega_x.resize(N, 0.0);
    resampled_traj.angular_velocity.omega_y.resize(N, 0.0);
    resampled_traj.angular_velocity.omega_z.resize(N, 0.0);
    for (int i = 0; i < N; ++i)
    {
        Eigen::Matrix3d R = quaternionToRotationMatrix(
            resampled_traj.orientation.quat_w[i],
            resampled_traj.orientation.quat_x[i],
            resampled_traj.orientation.quat_y[i],
            resampled_traj.orientation.quat_z[i]);
        Eigen::Matrix3d R_dot;
        if (i == 0)
        {
            Eigen::Matrix3d R_next = quaternionToRotationMatrix(
                resampled_traj.orientation.quat_w[i + 1],
                resampled_traj.orientation.quat_x[i + 1],
                resampled_traj.orientation.quat_y[i + 1],
                resampled_traj.orientation.quat_z[i + 1]);
            R_dot = (R_next - R) / dt_seconds;
        }
        else if (i == N - 1)
        {
            Eigen::Matrix3d R_prev = quaternionToRotationMatrix(
                resampled_traj.orientation.quat_w[i - 1],
                resampled_traj.orientation.quat_x[i - 1],
                resampled_traj.orientation.quat_y[i - 1],
                resampled_traj.orientation.quat_z[i - 1]);
            R_dot = (R - R_prev) / dt_seconds;
        }
        else
        {
            Eigen::Matrix3d R_next = quaternionToRotationMatrix(
                resampled_traj.orientation.quat_w[i + 1],
                resampled_traj.orientation.quat_x[i + 1],
                resampled_traj.orientation.quat_y[i + 1],
                resampled_traj.orientation.quat_z[i + 1]);
            Eigen::Matrix3d R_prev = quaternionToRotationMatrix(
                resampled_traj.orientation.quat_w[i - 1],
                resampled_traj.orientation.quat_x[i - 1],
                resampled_traj.orientation.quat_y[i - 1],
                resampled_traj.orientation.quat_z[i - 1]);
            R_dot = (R_next - R_prev) / (2.0 * dt_seconds);
        }
        Eigen::Matrix3d omega_hat = R_dot * R.transpose();
        Eigen::Vector3d omega = extractOmegaFromSkewSymmetric(omega_hat);
        resampled_traj.angular_velocity.omega_x[i] = omega(0);
        resampled_traj.angular_velocity.omega_y[i] = omega(1);
        resampled_traj.angular_velocity.omega_z[i] = omega(2);
    }
    resampled_traj.angular_acceleration.omega_dot_x.resize(N, 0.0);
    resampled_traj.angular_acceleration.omega_dot_y.resize(N, 0.0);
    resampled_traj.angular_acceleration.omega_dot_z.resize(N, 0.0);
    for (int i = 0; i < N; ++i)
    {
        if (i == 0)
        {
            resampled_traj.angular_acceleration.omega_dot_x[i] =
                (resampled_traj.angular_velocity.omega_x[i + 1] - resampled_traj.angular_velocity.omega_x[i]) / dt_seconds;
            resampled_traj.angular_acceleration.omega_dot_y[i] =
                (resampled_traj.angular_velocity.omega_y[i + 1] - resampled_traj.angular_velocity.omega_y[i]) / dt_seconds;
            resampled_traj.angular_acceleration.omega_dot_z[i] =
                (resampled_traj.angular_velocity.omega_z[i + 1] - resampled_traj.angular_velocity.omega_z[i]) / dt_seconds;
        }
        else if (i == N - 1)
        {
            resampled_traj.angular_acceleration.omega_dot_x[i] =
                (resampled_traj.angular_velocity.omega_x[i] - resampled_traj.angular_velocity.omega_x[i - 1]) / dt_seconds;
            resampled_traj.angular_acceleration.omega_dot_y[i] =
                (resampled_traj.angular_velocity.omega_y[i] - resampled_traj.angular_velocity.omega_y[i - 1]) / dt_seconds;
            resampled_traj.angular_acceleration.omega_dot_z[i] =
                (resampled_traj.angular_velocity.omega_z[i] - resampled_traj.angular_velocity.omega_z[i - 1]) / dt_seconds;
        }
        else
        {
            resampled_traj.angular_acceleration.omega_dot_x[i] =
                (resampled_traj.angular_velocity.omega_x[i + 1] - resampled_traj.angular_velocity.omega_x[i - 1]) / (2.0 * dt_seconds);
            resampled_traj.angular_acceleration.omega_dot_y[i] =
                (resampled_traj.angular_velocity.omega_y[i + 1] - resampled_traj.angular_velocity.omega_y[i - 1]) / (2.0 * dt_seconds);
            resampled_traj.angular_acceleration.omega_dot_z[i] =
                (resampled_traj.angular_velocity.omega_z[i + 1] - resampled_traj.angular_velocity.omega_z[i - 1]) / (2.0 * dt_seconds);
        }
    }
    std::cout << "Angular velocity and acceleration computation complete!" << std::endl;
    std::string output_path = csv_path.substr(0, csv_path.find_last_of("/\\") + 1) +
                              "demo_traj_resampled.csv";
    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        throw std::runtime_error("Cannot create output file: " + output_path);
    }
    outfile << "time_us,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,omega_x,omega_y,omega_z,omega_dot_x,omega_dot_y,omega_dot_z\n";
    outfile << std::fixed;
    for (size_t i = 0; i < resampled_traj.time_us.size(); ++i)
    {
        outfile << resampled_traj.time_us[i] << ","
                << resampled_traj.position.pos_x[i] << ","
                << resampled_traj.position.pos_y[i] << ","
                << resampled_traj.position.pos_z[i] << ","
                << resampled_traj.orientation.quat_w[i] << ","
                << resampled_traj.orientation.quat_x[i] << ","
                << resampled_traj.orientation.quat_y[i] << ","
                << resampled_traj.orientation.quat_z[i] << ","
                << resampled_traj.angular_velocity.omega_x[i] << ","
                << resampled_traj.angular_velocity.omega_y[i] << ","
                << resampled_traj.angular_velocity.omega_z[i] << ","
                << resampled_traj.angular_acceleration.omega_dot_x[i] << ","
                << resampled_traj.angular_acceleration.omega_dot_y[i] << ","
                << resampled_traj.angular_acceleration.omega_dot_z[i] << "\n";
    }
    outfile.close();
    std::cout << "Resampled trajectory saved to: " << output_path << std::endl;
    std::cout << "Original samples: " << original_traj.time_us.size()
              << ", Resampled: " << resampled_traj.time_us.size() << std::endl;
    return resampled_traj;
}
void DMP::train_orientation(const Trajectory &demo_traj)
{
    if (demo_traj.orientation.quat_w.empty() ||
        demo_traj.angular_velocity.omega_x.empty() ||
        demo_traj.angular_acceleration.omega_dot_x.empty())
    {
        std::cerr << "Error: Incomplete trajectory data for orientation training.\n";
        return;
    }
    int N_steps = demo_traj.orientation.quat_w.size();
    if (N_steps < 3)
    {
        std::cerr << "Error: Trajectory too short for training.\n";
        return;
    }
    tau = N_steps * dt;
    std::cout << "DMP Orientation training: tau set to " << tau << " seconds.\n";
    Quaternion q_0 = {
        demo_traj.orientation.quat_w[0],
        demo_traj.orientation.quat_x[0],
        demo_traj.orientation.quat_y[0],
        demo_traj.orientation.quat_z[0]};
    Quaternion q_goal = {
        demo_traj.orientation.quat_w.back(),
        demo_traj.orientation.quat_x.back(),
        demo_traj.orientation.quat_y.back(),
        demo_traj.orientation.quat_z.back()};
    Eigen::MatrixXd f_target_matrix(N_steps, 3);
    std::vector<double> x_track(N_steps);
    double x = 1.0;
    for (int t = 0; t < N_steps; ++t)
    {
        x_track[t] = x;
        Quaternion q_curr = {
            demo_traj.orientation.quat_w[t],
            demo_traj.orientation.quat_x[t],
            demo_traj.orientation.quat_y[t],
            demo_traj.orientation.quat_z[t]};
        std::vector<double> omega = {
            demo_traj.angular_velocity.omega_x[t],
            demo_traj.angular_velocity.omega_y[t],
            demo_traj.angular_velocity.omega_z[t]};
        std::vector<double> omega_dot = {
            demo_traj.angular_acceleration.omega_dot_x[t],
            demo_traj.angular_acceleration.omega_dot_y[t],
            demo_traj.angular_acceleration.omega_dot_z[t]};
        std::vector<double> eta = {
            tau * omega[0],
            tau * omega[1],
            tau * omega[2]};
        std::vector<double> eta_dot = {
            tau * omega_dot[0],
            tau * omega_dot[1],
            tau * omega_dot[2]};
        Quaternion q_inv = Quat_conjugate(q_curr);
        Quaternion q_diff = Quat_product(q_goal, q_inv);
        std::vector<double> log_diff = Quat_log(q_diff);
        std::vector<double> error_term = {
            2.0 * log_diff[0],
            2.0 * log_diff[1],
            2.0 * log_diff[2]};
        for (int i = 0; i < 3; ++i)
        {
            f_target_matrix(t, i) = tau * eta_dot[i] - alpha_z * (beta_z * error_term[i] - eta[i]);
        }
        x = phase_update(x, dt);
    }
    Quaternion q_inv_0 = Quat_conjugate(q_0);
    Quaternion q_diff_0 = Quat_product(q_goal, q_inv_0);
    std::vector<double> log_diff_0 = Quat_log(q_diff_0);
    std::vector<double> amplitude = {
        2.0 * log_diff_0[0],
        2.0 * log_diff_0[1],
        2.0 * log_diff_0[2]};
    for (int i = 0; i < 3; ++i)
    {
        if (std::abs(amplitude[i]) < 1e-6)
        {
            amplitude[i] = 1e-6; 
        }
    }
    Eigen::MatrixXd f_target_scaled(N_steps, 3);
    for (int t = 0; t < N_steps; ++t)
    {
        for (int i = 0; i < 3; ++i)
        {
            f_target_scaled(t, i) = (std::abs(amplitude[i]) < 1e-6) ? 0.0 : f_target_matrix(t, i) / amplitude[i];
        }
    }
    Eigen::MatrixXd Psi_norm(N_steps, n_basis);
    for (int t = 0; t < N_steps; ++t)
    {
        double x_t = x_track[t];
        double sum_psi = 0.0;
        for (int j = 0; j < n_basis; ++j)
        {
            sum_psi += basis_function(x_t, j);
        }
        for (int i = 0; i < n_basis; ++i)
        {
            double psi_t = basis_function(x_t, i);
            double gated_psi = psi_t * x_t;
            if (sum_psi < 1e-10)
                sum_psi = 1e-10;
            Psi_norm(t, i) = gated_psi / sum_psi;
        }
    }
    Eigen::VectorXd f_x = f_target_scaled.col(0);
    Eigen::VectorXd w_x_eig = Psi_norm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f_x);
    weights_rx.assign(w_x_eig.data(), w_x_eig.data() + w_x_eig.size());
    Eigen::VectorXd f_y = f_target_scaled.col(1);
    Eigen::VectorXd w_y_eig = Psi_norm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f_y);
    weights_ry.assign(w_y_eig.data(), w_y_eig.data() + w_y_eig.size());
    Eigen::VectorXd f_z = f_target_scaled.col(2);
    Eigen::VectorXd w_z_eig = Psi_norm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f_z);
    weights_rz.assign(w_z_eig.data(), w_z_eig.data() + w_z_eig.size());
    std::cout << "DMP Orientation Training Complete. Weights learned for Rx, Ry, Rz.\n";
}
std::vector<Quaternion> DMP::rollout_orientation(const Quaternion &q_0, const Quaternion &q_goal, double time_factor)
{
    tau = tau * time_factor;
    int num_steps = static_cast<int>(tau / dt);
    std::vector<Quaternion> quat_out(num_steps);
    Quaternion q = q_0;
    std::vector<double> eta = {0.0, 0.0, 0.0};
    std::vector<double> eta_dot = {0.0, 0.0, 0.0};
    double x = 1.0;
    quat_out[0] = q;
    Quaternion q_inv_0 = Quat_conjugate(q_0);
    Quaternion q_diff_0 = Quat_product(q_goal, q_inv_0);
    std::vector<double> log_diff_0 = Quat_log(q_diff_0);
    std::vector<double> amplitude = {
        2.0 * log_diff_0[0],
        2.0 * log_diff_0[1],
        2.0 * log_diff_0[2]};
    for (int t = 1; t < num_steps; ++t)
    {
        x = phase_update(x, dt);
        std::vector<double> F(3, 0.0);
        for (int axis = 0; axis < 3; ++axis)
        {
            double psi_sum = 0.0;
            double psi_weighted_sum = 0.0;
            const std::vector<double> *weights_axis;
            if (axis == 0)
                weights_axis = &weights_rx;
            else if (axis == 1)
                weights_axis = &weights_ry;
            else
                weights_axis = &weights_rz;
            for (int i = 0; i < n_basis; ++i)
            {
                double psi = basis_function(x, i);
                psi_sum += psi;
                psi_weighted_sum += psi * (*weights_axis)[i];
            }
            double f = 0.0;
            if (psi_sum > 1e-10)
            {
                f = psi_weighted_sum / psi_sum;
            }
            F[axis] = f * x * amplitude[axis];
        }
        Quaternion q_inv = Quat_conjugate(q);
        Quaternion q_diff = Quat_product(q_goal, q_inv);
        std::vector<double> log_diff = Quat_log(q_diff);
        std::vector<double> error_term = {
            2.0 * log_diff[0],
            2.0 * log_diff[1],
            2.0 * log_diff[2]};
        for (int i = 0; i < 3; ++i)
        {
            eta_dot[i] = (alpha_z * (beta_z * error_term[i] - eta[i]) + F[i]) / tau;
        }
        for (int i = 0; i < 3; ++i)
        {
            eta[i] = eta[i] + eta_dot[i] * dt;
        }
        double scale_factor = (dt / 2.0) * (1.0 / tau);
        std::vector<double> scaled_eta = {
            scale_factor * eta[0],
            scale_factor * eta[1],
            scale_factor * eta[2]};
        q = Quat_product(Quat_exp(scaled_eta), q);
        quat_out[t] = q;
    }
    return quat_out;
}
void DMP::train(const Trajectory &demo_traj)
{
    std::cout << "=== Starting DMP Training ===" << std::endl;
    demo_trajectory = demo_traj;
    std::cout << "\n--- Training Translation DMP ---" << std::endl;
    train_translation(demo_traj);
    std::cout << "\n--- Training Orientation DMP ---" << std::endl;
    train_orientation(demo_traj);
}
void DMP::rollout(const std::vector<double> &p_0,
                  const Quaternion &q_0,
                  const std::vector<double> &p_goal,
                  const Quaternion &q_goal,
                  double time_factor)
{
    std::cout << "\n--- Generating Rollout Trajectory ---" << std::endl;
    double tau_original = tau;
    std::vector<std::vector<double>> pos_rollout = rollout_translation(p_0, p_goal, time_factor);
    tau = tau_original;
    std::vector<Quaternion> quat_rollout = rollout_orientation(q_0, q_goal, time_factor);
    int num_steps = pos_rollout[0].size();
    rollout_trajectory.time_us.clear();
    rollout_trajectory.position.pos_x.clear();
    rollout_trajectory.position.pos_y.clear();
    rollout_trajectory.position.pos_z.clear();
    rollout_trajectory.orientation.quat_w.clear();
    rollout_trajectory.orientation.quat_x.clear();
    rollout_trajectory.orientation.quat_y.clear();
    rollout_trajectory.orientation.quat_z.clear();
    double dt_us = dt * 1e6; 
    for (int i = 0; i < num_steps; ++i)
    {
        rollout_trajectory.time_us.push_back(i * dt_us);
        rollout_trajectory.position.pos_x.push_back(pos_rollout[0][i]);
        rollout_trajectory.position.pos_y.push_back(pos_rollout[1][i]);
        rollout_trajectory.position.pos_z.push_back(pos_rollout[2][i]);
        rollout_trajectory.orientation.quat_w.push_back(quat_rollout[i].q_w);
        rollout_trajectory.orientation.quat_x.push_back(quat_rollout[i].q_x);
        rollout_trajectory.orientation.quat_y.push_back(quat_rollout[i].q_y);
        rollout_trajectory.orientation.quat_z.push_back(quat_rollout[i].q_z);
    }
    std::cout << "\n=== DMP Training Complete ===" << std::endl;
    std::cout << "Demo trajectory steps: " << demo_trajectory.time_us.size() << std::endl;
    std::cout << "Rollout trajectory steps: " << rollout_trajectory.time_us.size() << std::endl;
}
void DMP::save_rollout_csv(const std::vector<double> &rollout_traj,
                           const std::string &output_path)
{
    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        throw std::runtime_error("Cannot create output file: " + output_path);
    }
    outfile << "time_us,pos_x\n";
    double dt_us = dt * 1e6;
    outfile << std::fixed;
    for (size_t i = 0; i < rollout_traj.size(); ++i)
    {
        double time_us = i * dt_us;
        outfile << time_us << "," << rollout_traj[i] << "\n";
    }
    outfile.close();
    std::cout << "Rollout trajectory saved to: " << output_path << std::endl;
}
void DMP::save_full_rollout_csv(const std::vector<double> &pos_traj,
                                const std::vector<Quaternion> &quat_traj,
                                const std::string &output_path)
{
    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        throw std::runtime_error("Cannot create output file: " + output_path);
    }
    size_t num_steps = std::min(pos_traj.size(), quat_traj.size());
    outfile << "time_us,pos_x,quat_w,quat_x,quat_y,quat_z\n";
    double dt_us = dt * 1e6;
    outfile << std::fixed;
    for (size_t i = 0; i < num_steps; ++i)
    {
        double time_us = i * dt_us;
        outfile << time_us << ","
                << pos_traj[i] << ","
                << quat_traj[i].q_w << ","
                << quat_traj[i].q_x << ","
                << quat_traj[i].q_y << ","
                << quat_traj[i].q_z << "\n";
    }
    outfile.close();
    std::cout << "Full rollout trajectory (position + orientation) saved to: " << output_path << std::endl;
}
void DMP::save_translation_rollout_csv(const std::vector<std::vector<double>> &pos_traj,
                                       const std::string &output_path)
{
    std::ofstream outfile(output_path);
    if (!outfile.is_open())
    {
        throw std::runtime_error("Cannot create output file: " + output_path);
    }
    outfile << "time_us,pos_x,pos_y,pos_z\n";
    double dt_us = dt * 1e6;
    size_t num_steps = pos_traj[0].size();
    outfile << std::fixed;
    for (size_t i = 0; i < num_steps; ++i)
    {
        double time_us = i * dt_us;
        outfile << time_us << ","
                << pos_traj[0][i] << ","
                << pos_traj[1][i] << ","
                << pos_traj[2][i] << "\n";
    }
    outfile.close();
    std::cout << "Translation rollout trajectory saved to: " << output_path << std::endl;
}
void DMP::train_orientation_corrected(const Trajectory &demo_traj)
{
    if (demo_traj.orientation.quat_w.empty() ||
        demo_traj.angular_velocity.omega_x.empty())
    {
        std::cerr << "Error: Incomplete trajectory data for orientation training.\n";
        return;
    }
    int N_steps = demo_traj.orientation.quat_w.size();
    if (N_steps < 3)
    {
        std::cerr << "Error: Trajectory too short for training.\n";
        return;
    }
    tau = N_steps * dt;
    std::cout << "DMP Orientation (Corrected) training: tau set to " << tau << " seconds.\n";
    Quaternion q_0 = {
        demo_traj.orientation.quat_w[0],
        demo_traj.orientation.quat_x[0],
        demo_traj.orientation.quat_y[0],
        demo_traj.orientation.quat_z[0]};
    Quaternion q_goal = {
        demo_traj.orientation.quat_w.back(),
        demo_traj.orientation.quat_x.back(),
        demo_traj.orientation.quat_y.back(),
        demo_traj.orientation.quat_z.back()};
    std::vector<std::vector<double>> e_Q(N_steps, std::vector<double>(3));
    std::vector<std::vector<double>> e_Q_dot(N_steps, std::vector<double>(3));
    std::vector<std::vector<double>> e_Q_dot_dot(N_steps, std::vector<double>(3, 0.0));
    std::vector<double> x_track(N_steps);
    double x = 1.0;
    for (int t = 0; t < N_steps; ++t)
    {
        x_track[t] = x;
        Quaternion q_curr = {
            demo_traj.orientation.quat_w[t],
            demo_traj.orientation.quat_x[t],
            demo_traj.orientation.quat_y[t],
            demo_traj.orientation.quat_z[t]};
        std::vector<double> omega = {
            demo_traj.angular_velocity.omega_x[t],
            demo_traj.angular_velocity.omega_y[t],
            demo_traj.angular_velocity.omega_z[t]};
        Quaternion omega_quat = {0.0, omega[0], omega[1], omega[2]};
        Quaternion Q_dot_temp = Quat_product(omega_quat, q_curr);
        Quaternion Q_dot = {
            0.5 * Q_dot_temp.q_w,
            0.5 * Q_dot_temp.q_x,
            0.5 * Q_dot_temp.q_y,
            0.5 * Q_dot_temp.q_z};
        Quaternion q_inv = Quat_conjugate(q_curr);
        Quaternion temp1 = Quat_product(q_goal, q_inv);
        Quaternion temp2 = Quat_product(Q_dot, q_inv);
        Quaternion quat_chain = Quat_product(temp1, temp2);
        Quaternion q_diff = Quat_product(q_goal, q_inv);
        Eigen::MatrixXd J_q = Quat_Jacobian(q_diff);
        Eigen::MatrixXd jacobian_term = -2.0 * J_q;
        Eigen::Vector4d quat_chain_vec;
        quat_chain_vec << quat_chain.q_w, quat_chain.q_x, quat_chain.q_y, quat_chain.q_z;
        Eigen::Vector3d e_Q_dot_vec = jacobian_term * quat_chain_vec;
        e_Q_dot[t][0] = e_Q_dot_vec(0);
        e_Q_dot[t][1] = e_Q_dot_vec(1);
        e_Q_dot[t][2] = e_Q_dot_vec(2);
        std::vector<double> log_diff = Quat_log(q_diff);
        e_Q[t][0] = 2.0 * log_diff[0];
        e_Q[t][1] = 2.0 * log_diff[1];
        e_Q[t][2] = 2.0 * log_diff[2];
        x = phase_update(x, dt);
    }
    for (int t = 1; t < N_steps - 1; ++t)
    {
        for (int i = 0; i < 3; ++i)
        {
            e_Q_dot_dot[t][i] = (e_Q_dot[t + 1][i] - e_Q_dot[t - 1][i]) / (2.0 * dt);
        }
    }
    if (N_steps >= 2)
    {
        for (int i = 0; i < 3; ++i)
        {
            e_Q_dot_dot[N_steps - 1][i] = (e_Q_dot[N_steps - 1][i] - e_Q_dot[N_steps - 2][i]) / dt;
        }
    }
    Eigen::MatrixXd F_target(N_steps, 3);
    double tau_sq = tau * tau;
    for (int t = 0; t < N_steps; ++t)
    {
        for (int i = 0; i < 3; ++i)
        {
            F_target(t, i) = tau_sq * e_Q_dot_dot[t][i] +
                             alpha_z * (beta_z * e_Q[t][i] + e_Q_dot[t][i]);
        }
    }
    Quaternion q_inv_0 = Quat_conjugate(q_0);
    Quaternion q_diff_0 = Quat_product(q_goal, q_inv_0);
    std::vector<double> log_diff_0 = Quat_log(q_diff_0);
    std::vector<double> amplitude = {
        2.0 * log_diff_0[0],
        2.0 * log_diff_0[1],
        2.0 * log_diff_0[2]};
    for (int i = 0; i < 3; ++i)
    {
        if (std::abs(amplitude[i]) < 1e-6)
        {
            amplitude[i] = 1e-6;
        }
    }
    Eigen::MatrixXd F_target_scaled(N_steps, 3);
    for (int t = 0; t < N_steps; ++t)
    {
        for (int i = 0; i < 3; ++i)
        {
            F_target_scaled(t, i) = F_target(t, i) / amplitude[i];
        }
    }
    Eigen::MatrixXd Psi_norm(N_steps, n_basis);
    for (int t = 0; t < N_steps; ++t)
    {
        double x_t = x_track[t];
        double sum_psi = 0.0;
        for (int j = 0; j < n_basis; ++j)
        {
            sum_psi += basis_function(x_t, j);
        }
        for (int i = 0; i < n_basis; ++i)
        {
            double psi_t = basis_function(x_t, i);
            double gated_psi = psi_t * x_t;
            if (sum_psi < 1e-10)
                sum_psi = 1e-10;
            Psi_norm(t, i) = gated_psi / sum_psi;
        }
    }
    Eigen::VectorXd f_x = F_target_scaled.col(0);
    Eigen::VectorXd w_x_eig = Psi_norm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f_x);
    weights_rx.assign(w_x_eig.data(), w_x_eig.data() + w_x_eig.size());
    Eigen::VectorXd f_y = F_target_scaled.col(1);
    Eigen::VectorXd w_y_eig = Psi_norm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f_y);
    weights_ry.assign(w_y_eig.data(), w_y_eig.data() + w_y_eig.size());
    Eigen::VectorXd f_z = F_target_scaled.col(2);
    Eigen::VectorXd w_z_eig = Psi_norm.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(f_z);
    weights_rz.assign(w_z_eig.data(), w_z_eig.data() + w_z_eig.size());
    std::cout << "DMP Orientation (Corrected) Training Complete. Weights learned for Rx, Ry, Rz.\n";
}
std::vector<Quaternion> DMP::rollout_orientation_corrected(
    const Quaternion &q_0,
    const Quaternion &q_goal,
    double time_factor)
{
    tau = tau * time_factor;
    int num_steps = static_cast<int>(tau / dt);
    std::vector<Quaternion> quat_out(num_steps);
    Quaternion q_inv_0 = Quat_conjugate(q_0);
    Quaternion q_diff_0 = Quat_product(q_goal, q_inv_0);
    std::vector<double> log_diff_0 = Quat_log(q_diff_0);
    std::vector<double> e_Q = {
        2.0 * log_diff_0[0],
        2.0 * log_diff_0[1],
        2.0 * log_diff_0[2]};
    std::vector<double> amplitude = e_Q; 
    std::vector<double> e_Q_dot = {0.0, 0.0, 0.0};
    std::vector<double> e_Q_dot_dot = {0.0, 0.0, 0.0};
    double x = 1.0;
    Quaternion q = q_0;
    quat_out[0] = q;
    double tau_sq = tau * tau;
    for (int t = 1; t < num_steps; ++t)
    {
        x = phase_update(x, dt);
        std::vector<double> F(3, 0.0);
        for (int axis = 0; axis < 3; ++axis)
        {
            double psi_sum = 0.0;
            double psi_weighted_sum = 0.0;
            const std::vector<double> *weights_axis;
            if (axis == 0)
                weights_axis = &weights_rx;
            else if (axis == 1)
                weights_axis = &weights_ry;
            else
                weights_axis = &weights_rz;
            for (int i = 0; i < n_basis; ++i)
            {
                double psi = basis_function(x, i);
                psi_sum += psi;
                psi_weighted_sum += psi * (*weights_axis)[i];
            }
            double f = 0.0;
            if (psi_sum > 1e-10)
            {
                f = psi_weighted_sum / psi_sum;
            }
            F[axis] = f * x * amplitude[axis];
        }
        for (int i = 0; i < 3; ++i)
        {
            e_Q_dot_dot[i] = (1.0 / tau_sq) *
                             (-alpha_z * (beta_z * e_Q[i] + e_Q_dot[i]) + F[i]);
        }
        for (int i = 0; i < 3; ++i)
        {
            e_Q_dot[i] = e_Q_dot[i] + e_Q_dot_dot[i] * dt;
        }
        for (int i = 0; i < 3; ++i)
        {
            e_Q[i] = e_Q[i] + e_Q_dot[i] * dt;
        }
        std::vector<double> half_e_Q = {0.5 * e_Q[0], 0.5 * e_Q[1], 0.5 * e_Q[2]};
        Quaternion exp_half_e_Q = Quat_exp(half_e_Q);
        Quaternion exp_half_e_Q_conj = Quat_conjugate(exp_half_e_Q);
        q = Quat_product(exp_half_e_Q_conj, q_goal);
        quat_out[t] = q;
    }
    return quat_out;
}