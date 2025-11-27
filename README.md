# Dynamic Movement Primitives (DMP) Library

`dmplib` is a C++ library designed to automate the learning and generation of **full-pose (position and orientation)** robot trajectories using **Dynamic Movement Primitives (DMPs)**. It accepts a demonstration trajectory (position and orientation data) and allows for robust generalization and modification of the movement, such as changing the goal position or orientation.

---

## Features

- **Full Pose Support:** Learns and reproduces both 3D position and quaternion orientation movements.
- **Automated Process:** Handles trajectory reading, resampling, DMP training, and rollout generation.
- **Configurable:** Key DMP parameters (`alpha_z`, `beta_z`, `n_basis`, `dt` etc.) can be easily tuned via public setter methods.
- **File I/O:** Built-in functions to read demonstration trajectories from CSV files and save the generated rollout trajectories.

---

## DMP Formulation

### **1. Position DMPs (Translation)**  
The core structure follows:

A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal,  
*“Dynamical movement primitives: Learning attractor models for motor behaviors,”*  
Neural Computations, vol. 25, no. 2, pp. 328–373, 2013.

---

### **2. Orientation DMPs (Rotation)**  
Supports two formulations via `DMP::Type`:

| Formulation | Enum | Reference |
|------------|------|-----------|
| Standard | `DMP::Type::Standard` | A. Ude et al., *Orientation in Cartesian Space Dynamic Movement Primitives*, ICRA 2014 |
| Corrected | `DMP::Type::Corrected` | Leonidas Koutras & Zoe Doulgeri, *A correct formulation for the Orientation Dynamic Movement Primitives for robot control in the Cartesian space.* |

---

## Usage Example

```cpp
// 1. Setup and Configuration
DMP dmp_instance;
dmp_instance.set_dt(0.01);
dmp_instance.set_n_basis(60);

// 2. Load Demonstration Data
Trajectory demo_traj = dmp_instance.read_trajectory("../demo_trajectory.csv");

// 3. Define Modified Goal Poses
std::vector<double> p_0 = {demo_traj.position.pos_x.front(), ...};
Quaternion q_0 = {demo_traj.orientation.quat_w.front(), ...};
std::vector<double> p_goal = {modified_x, modified_y, modified_z};
Quaternion q_goal = {modified_w, modified_x, modified_y, modified_z};

// 4. Train the DMP
dmp_instance.train(demo_traj);

// 5. Rollout the New Trajectory
dmp_instance.rollout(p_0, q_0, p_goal, q_goal, 1.0);

// 6. Retrieve and Save Results
Trajectory rollout_traj = dmp_instance.get_rollout_trajectory();
// ... code to save rollout_traj to CSV
