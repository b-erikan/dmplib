#include "dmplib/dmp.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
int main(int argc, char* argv[])
{
    try {
        std::cout << "=== DMP Trajectory Generator ===" << std::endl;
        DMP dmp_instance;
        dmp_instance.set_dt(0.01);
        dmp_instance.set_alpha_x(4.6052);
        dmp_instance.set_alpha_z(80.0);
        dmp_instance.set_beta_z(20.0);
        dmp_instance.set_n_basis(60);
        std::string csv_path = "../demo_trajectory.csv";
        if (argc > 1) {
            csv_path = argv[1];
        }
        double time_factor = 1.0; 
        std::cout << "\n[INFO] Reading and resampling trajectory from: " << csv_path << std::endl;
        Trajectory demo_traj = dmp_instance.read_trajectory(csv_path);
        std::cout << "[INFO] Trajectory loaded with " << demo_traj.position.pos_x.size() 
                  << " samples" << std::endl;
        std::vector<double> p_0 = {
            demo_traj.position.pos_x.front(),
            demo_traj.position.pos_y.front(),
            demo_traj.position.pos_z.front()
        };
        Quaternion q_0 = {
            demo_traj.orientation.quat_w.front(),
            demo_traj.orientation.quat_x.front(),
            demo_traj.orientation.quat_y.front(),
            demo_traj.orientation.quat_z.front()
        };
        std::vector<double> p_goal_demo = {
            demo_traj.position.pos_x.back(),
            demo_traj.position.pos_y.back(),
            demo_traj.position.pos_z.back()
        };
        Quaternion q_goal_demo = {
            demo_traj.orientation.quat_w.back(),
            demo_traj.orientation.quat_x.back(),
            demo_traj.orientation.quat_y.back(),
            demo_traj.orientation.quat_z.back()
        };
        std::vector<double> p_goal = {
            p_goal_demo[0] + 0.1,
            p_goal_demo[1] - 0.1,
            p_goal_demo[2] + 0.1
        };
        double roll_rad = 10.0 * M_PI / 180.0;
        double half_roll = roll_rad / 2.0;
        Quaternion q_roll = {std::cos(half_roll), std::sin(half_roll), 0.0, 0.0};
        double pitch_rad = -10.0 * M_PI / 180.0;
        double half_pitch = pitch_rad / 2.0;
        Quaternion q_pitch = {std::cos(half_pitch), 0.0, std::sin(half_pitch), 0.0};
        double yaw_rad = -10.0 * M_PI / 180.0;
        double half_yaw = yaw_rad / 2.0;
        Quaternion q_yaw = {std::cos(half_yaw), 0.0, 0.0, std::sin(half_yaw)};
        Quaternion q_temp = Quat_product(q_yaw, q_pitch);
        Quaternion q_additional = Quat_product(q_temp, q_roll);
        Quaternion q_goal = Quat_product(q_additional, q_goal_demo);
        std::cout << "\n=== DEMONSTRATION TRAJECTORY ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Demo Initial position: [" << p_0[0] << ", " << p_0[1] << ", " << p_0[2] << "]" << std::endl;
        std::cout << "Demo Goal position: [" << p_goal_demo[0] << ", " << p_goal_demo[1] << ", " << p_goal_demo[2] << "]" << std::endl;
        std::cout << "Demo Goal orientation: [" << q_goal_demo.q_w << ", " << q_goal_demo.q_x << ", " 
                  << q_goal_demo.q_y << ", " << q_goal_demo.q_z << "]" << std::endl;
        std::cout << "\n=== MODIFIED GOALS FOR ROLLOUT ===" << std::endl;
        std::cout << "Rollout Initial position: [" << p_0[0] << ", " << p_0[1] << ", " << p_0[2] << "]" << std::endl;
        std::cout << "Rollout Goal position: [" << p_goal[0] << ", " << p_goal[1] << ", " << p_goal[2] 
                  << "] (+0.1m, -0.1m, +0.1m)" << std::endl;
        std::cout << "Rollout Goal orientation: [" << q_goal.q_w << ", " << q_goal.q_x << ", " 
                  << q_goal.q_y << ", " << q_goal.q_z << "] (+10° roll, -10° pitch, +10° yaw)" << std::endl;
        std::cout << "\n[INFO] Training DMP on demonstration trajectory..." << std::endl;
        dmp_instance.train(demo_traj);
        std::cout << "[INFO] DMP training complete!" << std::endl;
        std::cout << "\n[INFO] Rolling out DMP with modified goals..." << std::endl;
        dmp_instance.rollout(p_0, q_0, p_goal_demo, q_goal_demo, time_factor);
        Trajectory rollout_traj = dmp_instance.get_rollout_trajectory();
        std::cout << "[INFO] Rollout trajectory generated with " << rollout_traj.position.pos_x.size() 
                  << " points" << std::endl;
        std::cout << "\n=== ROLLOUT RESULTS ===" << std::endl;
        std::cout << "Final position: [" << rollout_traj.position.pos_x.back() << ", "
                  << rollout_traj.position.pos_y.back() << ", "
                  << rollout_traj.position.pos_z.back() << "]" << std::endl;
        std::cout << "Final orientation: [" << rollout_traj.orientation.quat_w.back() << ", "
                  << rollout_traj.orientation.quat_x.back() << ", "
                  << rollout_traj.orientation.quat_y.back() << ", "
                  << rollout_traj.orientation.quat_z.back() << "]" << std::endl;
        std::string rollout_csv_path = "rollout_trajectory.csv";
        std::ofstream outfile(rollout_csv_path);
        if (!outfile.is_open()) {
            throw std::runtime_error("Cannot create output file: " + rollout_csv_path);
        }
        outfile << "time_us,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z\n";
        outfile << std::fixed;
        for (size_t i = 0; i < rollout_traj.time_us.size(); ++i) {
            outfile << rollout_traj.time_us[i] << ","
                    << rollout_traj.position.pos_x[i] << ","
                    << rollout_traj.position.pos_y[i] << ","
                    << rollout_traj.position.pos_z[i] << ","
                    << rollout_traj.orientation.quat_w[i] << ","
                    << rollout_traj.orientation.quat_x[i] << ","
                    << rollout_traj.orientation.quat_y[i] << ","
                    << rollout_traj.orientation.quat_z[i] << "\n";
        }
        outfile.close();
        std::cout << "\n[SUCCESS] Full rollout trajectory saved to: " << rollout_csv_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}