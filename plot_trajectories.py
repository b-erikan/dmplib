#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def quaternion_to_euler(qw, qx, qy, qz):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def plot_trajectories():
    # Define file paths
    workspace_path = "/home/berke/dmplib"
    demo_path = os.path.join(workspace_path, "demo_traj_resampled.csv")
    rollout_path = os.path.join(workspace_path, "build/rollout_trajectory.csv")
    
    # Check if files exist
    if not os.path.exists(demo_path):
        print(f"Error: Demo trajectory file not found at {demo_path}")
        return
    
    if not os.path.exists(rollout_path):
        print(f"Error: Rollout trajectory file not found at {rollout_path}")
        return
    
    # Read CSV files
    demo_data = pd.read_csv(demo_path)
    rollout_data = pd.read_csv(rollout_path)
    
    # Convert time from microseconds to seconds
    demo_time = demo_data['time_us'].values / 1e6
    rollout_time = rollout_data['time_us'].values / 1e6
    
    # Extract position data (X, Y, Z)
    demo_pos_x = demo_data['pos_x'].values
    demo_pos_y = demo_data['pos_y'].values
    demo_pos_z = demo_data['pos_z'].values
    
    rollout_pos_x = rollout_data['pos_x'].values
    rollout_pos_y = rollout_data['pos_y'].values
    rollout_pos_z = rollout_data['pos_z'].values
    
    # Extract quaternion data and convert to Euler angles
    demo_roll, demo_pitch, demo_yaw = quaternion_to_euler(
        demo_data['quat_w'].values,
        demo_data['quat_x'].values,
        demo_data['quat_y'].values,
        demo_data['quat_z'].values
    )
    
    rollout_roll, rollout_pitch, rollout_yaw = quaternion_to_euler(
        rollout_data['quat_w'].values,
        rollout_data['quat_x'].values,
        rollout_data['quat_y'].values,
        rollout_data['quat_z'].values
    )
    
    # Convert radians to degrees for better readability
    demo_roll_deg = np.rad2deg(demo_roll)
    demo_pitch_deg = np.rad2deg(demo_pitch)
    demo_yaw_deg = np.rad2deg(demo_yaw)
    
    rollout_roll_deg = np.rad2deg(rollout_roll)
    rollout_pitch_deg = np.rad2deg(rollout_pitch)
    rollout_yaw_deg = np.rad2deg(rollout_yaw)
    
    # ========== FIGURE 1: Position Trajectories (X, Y, Z) ==========
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot Position X
    axes1[0].plot(demo_time, demo_pos_x, 'b-', linewidth=2.5, label='Demo', alpha=0.8)
    axes1[0].plot(rollout_time, rollout_pos_x, 'r--', linewidth=2.5, label='DMP Rollout', alpha=0.8)
    axes1[0].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes1[0].set_ylabel('Position X (meters)', fontsize=12, fontweight='bold')
    axes1[0].set_title('Position X Trajectory', fontsize=14, fontweight='bold')
    axes1[0].legend(fontsize=11, loc='best')
    axes1[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot Position Y
    axes1[1].plot(demo_time, demo_pos_y, 'b-', linewidth=2.5, label='Demo', alpha=0.8)
    axes1[1].plot(rollout_time, rollout_pos_y, 'r--', linewidth=2.5, label='DMP Rollout', alpha=0.8)
    axes1[1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes1[1].set_ylabel('Position Y (meters)', fontsize=12, fontweight='bold')
    axes1[1].set_title('Position Y Trajectory', fontsize=14, fontweight='bold')
    axes1[1].legend(fontsize=11, loc='best')
    axes1[1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot Position Z
    axes1[2].plot(demo_time, demo_pos_z, 'b-', linewidth=2.5, label='Demo', alpha=0.8)
    axes1[2].plot(rollout_time, rollout_pos_z, 'r--', linewidth=2.5, label='DMP Rollout', alpha=0.8)
    axes1[2].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes1[2].set_ylabel('Position Z (meters)', fontsize=12, fontweight='bold')
    axes1[2].set_title('Position Z Trajectory', fontsize=14, fontweight='bold')
    axes1[2].legend(fontsize=11, loc='best')
    axes1[2].grid(True, alpha=0.3, linestyle='--')
    
    # Main title for position figure
    fig1.suptitle('DMP Position Trajectories: Demo vs Rollout (X, Y, Z)', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save position plot
    pos_output_path = os.path.join(workspace_path, "position_comparison.png")
    plt.savefig(pos_output_path, dpi=300, bbox_inches='tight')
    print(f"Position plot saved to: {pos_output_path}")
    
    # ========== FIGURE 2: Orientation Trajectories (Roll, Pitch, Yaw) ==========
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot Roll
    axes2[0].plot(demo_time, demo_roll_deg, 'b-', linewidth=2.5, label='Demo', alpha=0.8)
    axes2[0].plot(rollout_time, rollout_roll_deg, 'r--', linewidth=2.5, label='DMP Rollout', alpha=0.8)
    axes2[0].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes2[0].set_ylabel('Roll (degrees)', fontsize=12, fontweight='bold')
    axes2[0].set_title('Roll Angle', fontsize=14, fontweight='bold')
    axes2[0].legend(fontsize=11, loc='best')
    axes2[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot Pitch
    axes2[1].plot(demo_time, demo_pitch_deg, 'b-', linewidth=2.5, label='Demo', alpha=0.8)
    axes2[1].plot(rollout_time, rollout_pitch_deg, 'r--', linewidth=2.5, label='DMP Rollout', alpha=0.8)
    axes2[1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes2[1].set_ylabel('Pitch (degrees)', fontsize=12, fontweight='bold')
    axes2[1].set_title('Pitch Angle', fontsize=14, fontweight='bold')
    axes2[1].legend(fontsize=11, loc='best')
    axes2[1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot Yaw
    axes2[2].plot(demo_time, demo_yaw_deg, 'b-', linewidth=2.5, label='Demo', alpha=0.8)
    axes2[2].plot(rollout_time, rollout_yaw_deg, 'r--', linewidth=2.5, label='DMP Rollout', alpha=0.8)
    axes2[2].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes2[2].set_ylabel('Yaw (degrees)', fontsize=12, fontweight='bold')
    axes2[2].set_title('Yaw Angle', fontsize=14, fontweight='bold')
    axes2[2].legend(fontsize=11, loc='best')
    axes2[2].grid(True, alpha=0.3, linestyle='--')
    
    # Main title for orientation figure
    fig2.suptitle('DMP Orientation Trajectories: Demo vs Rollout (Roll, Pitch, Yaw)', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save orientation plot
    ori_output_path = os.path.join(workspace_path, "orientation_comparison.png")
    plt.savefig(ori_output_path, dpi=300, bbox_inches='tight')
    print(f"Orientation plot saved to: {ori_output_path}")
    
    # Display both plots
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAJECTORY STATISTICS")
    print("="*60)
    
    print(f"\nTrajectory Info:")
    print(f"  Demo trajectory: {len(demo_pos_x)} points, duration: {demo_time[-1]:.2f}s")
    print(f"  Rollout trajectory: {len(rollout_pos_x)} points, duration: {rollout_time[-1]:.2f}s")
    
    print(f"\n{'Position Start/End Values':^60}")
    print("-"*60)
    print(f"  X-axis:")
    print(f"    Demo:    start = {demo_pos_x[0]:+.6f} m,  end = {demo_pos_x[-1]:+.6f} m")
    print(f"    Rollout: start = {rollout_pos_x[0]:+.6f} m,  end = {rollout_pos_x[-1]:+.6f} m")
    print(f"  Y-axis:")
    print(f"    Demo:    start = {demo_pos_y[0]:+.6f} m,  end = {demo_pos_y[-1]:+.6f} m")
    print(f"    Rollout: start = {rollout_pos_y[0]:+.6f} m,  end = {rollout_pos_y[-1]:+.6f} m")
    print(f"  Z-axis:")
    print(f"    Demo:    start = {demo_pos_z[0]:+.6f} m,  end = {demo_pos_z[-1]:+.6f} m")
    print(f"    Rollout: start = {rollout_pos_z[0]:+.6f} m,  end = {rollout_pos_z[-1]:+.6f} m")
    
    print(f"\n{'Orientation Start/End Values (degrees)':^60}")
    print("-"*60)
    print(f"  Roll:")
    print(f"    Demo:    start = {demo_roll_deg[0]:+8.2f}°,  end = {demo_roll_deg[-1]:+8.2f}°")
    print(f"    Rollout: start = {rollout_roll_deg[0]:+8.2f}°,  end = {rollout_roll_deg[-1]:+8.2f}°")
    print(f"  Pitch:")
    print(f"    Demo:    start = {demo_pitch_deg[0]:+8.2f}°,  end = {demo_pitch_deg[-1]:+8.2f}°")
    print(f"    Rollout: start = {rollout_pitch_deg[0]:+8.2f}°,  end = {rollout_pitch_deg[-1]:+8.2f}°")
    print(f"  Yaw:")
    print(f"    Demo:    start = {demo_yaw_deg[0]:+8.2f}°,  end = {demo_yaw_deg[-1]:+8.2f}°")
    print(f"    Rollout: start = {rollout_yaw_deg[0]:+8.2f}°,  end = {rollout_yaw_deg[-1]:+8.2f}°")
    
    # Calculate errors (if same length)
    if len(demo_pos_x) == len(rollout_pos_x):
        pos_x_error = np.mean(np.abs(demo_pos_x - rollout_pos_x))
        pos_y_error = np.mean(np.abs(demo_pos_y - rollout_pos_y))
        pos_z_error = np.mean(np.abs(demo_pos_z - rollout_pos_z))
        roll_error = np.mean(np.abs(demo_roll_deg - rollout_roll_deg))
        pitch_error = np.mean(np.abs(demo_pitch_deg - rollout_pitch_deg))
        yaw_error = np.mean(np.abs(demo_yaw_deg - rollout_yaw_deg))
        
        pos_x_max_error = np.max(np.abs(demo_pos_x - rollout_pos_x))
        pos_y_max_error = np.max(np.abs(demo_pos_y - rollout_pos_y))
        pos_z_max_error = np.max(np.abs(demo_pos_z - rollout_pos_z))
        roll_max_error = np.max(np.abs(demo_roll_deg - rollout_roll_deg))
        pitch_max_error = np.max(np.abs(demo_pitch_deg - rollout_pitch_deg))
        yaw_max_error = np.max(np.abs(demo_yaw_deg - rollout_yaw_deg))
        
        print(f"\n{'Mean Absolute Errors':^60}")
        print("-"*60)
        print(f"  Position:")
        print(f"    X: {pos_x_error:.6f} m")
        print(f"    Y: {pos_y_error:.6f} m")
        print(f"    Z: {pos_z_error:.6f} m")
        print(f"  Orientation:")
        print(f"    Roll:  {roll_error:6.3f}°")
        print(f"    Pitch: {pitch_error:6.3f}°")
        print(f"    Yaw:   {yaw_error:6.3f}°")
        
        print(f"\n{'Maximum Absolute Errors':^60}")
        print("-"*60)
        print(f"  Position:")
        print(f"    X: {pos_x_max_error:.6f} m")
        print(f"    Y: {pos_y_max_error:.6f} m")
        print(f"    Z: {pos_z_max_error:.6f} m")
        print(f"  Orientation:")
        print(f"    Roll:  {roll_max_error:6.3f}°")
        print(f"    Pitch: {pitch_max_error:6.3f}°")
        print(f"    Yaw:   {yaw_max_error:6.3f}°")
    else:
        print(f"\nWarning: Trajectory lengths differ - cannot compute errors")
        print(f"  Demo length: {len(demo_pos_x)}")
        print(f"  Rollout length: {len(rollout_pos_x)}")
    
 
    
    
if __name__ == "__main__":
    plot_trajectories()