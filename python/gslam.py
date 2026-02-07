import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --- 1. Configuration & Constants ---
DT = 1.0                # Time step
SIM_STEPS = 100          # Number of steps
V_CMD = 1.0             # Linear velocity command
W_CMD = 0.1             # Angular velocity command (turn rate)

# Noise parameters
# R_NOISE = 0.05          # Observation range noise (std dev)
# PHI_NOISE = 0.03        # Observation bearing noise (std dev)
R_NOISE = 0.2          # Observation range noise (std dev)
PHI_NOISE = 0.1        # Observation bearing noise (std dev)
V_NOISE = 0.1           # Control velocity noise
W_NOISE = 0.05          # Control angular noise
ODOM_V_NOISE = 0.02     # Odometry model noise
ODOM_W_NOISE = 0.02

# Data Association
MAHALANOBIS_THRESHOLD = 2.0  # Threshold to accept a match
OBS_COV = np.diag([R_NOISE**2, PHI_NOISE**2]) # Sensor covariance

# --- 2. Helper Functions ---

def normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def motion_model(pose, control):
    """Predict next pose given current pose and control [v, w]."""
    x, y, theta = pose
    v, w = control
    
    # Avoid division by zero for straight lines
    if abs(w) < 1e-5:
        new_x = x + v * np.cos(theta) * DT
        new_y = y + v * np.sin(theta) * DT
        new_theta = theta
    else:
        new_x = x + (v/w) * (np.sin(theta + w*DT) - np.sin(theta))
        new_y = y + (v/w) * (np.cos(theta) - np.cos(theta + w*DT))
        new_theta = normalize_angle(theta + w*DT)
        
    return np.array([new_x, new_y, new_theta])

def observe_landmarks(true_pose, true_landmarks, sensor_range=15.0):
    """Generate noisy observations for landmarks within range."""
    observations = [] # List of [range, bearing, true_id (hidden from slam)]
    x, y, theta = true_pose
    
    for i, lm in enumerate(true_landmarks):
        dx = lm[0] - x
        dy = lm[1] - y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist <= sensor_range:
            # Add noise
            dist_n = dist + np.random.normal(0, R_NOISE)
            
            # True bearing relative to robot
            angle_global = np.arctan2(dy, dx)
            bearing_true = normalize_angle(angle_global - theta)
            bearing_n = normalize_angle(bearing_true + np.random.normal(0, PHI_NOISE))
            
            observations.append({'z': np.array([dist_n, bearing_n]), 'true_id': i})
            
    return observations

def get_expected_obs(robot_pose, landmark_pos):
    """Calculate expected range and bearing from pose to landmark."""
    dx = landmark_pos[0] - robot_pose[0]
    dy = landmark_pos[1] - robot_pose[1]
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx) - robot_pose[2]
    return np.array([r, normalize_angle(phi)])

def inverse_sensor_model(robot_pose, observation):
    """Calculate global landmark position from robot pose and observation."""
    x, y, theta = robot_pose
    r, phi = observation
    lx = x + r * np.cos(theta + phi)
    ly = y + r * np.sin(theta + phi)
    return np.array([lx, ly])

# --- 3. SLAM Logic ---

def data_association(pred_pose, observation, map_landmarks):
    """
    Match observation to existing landmarks or return None (new landmark).
    Uses Mahalanobis distance.
    """
    if not map_landmarks:
        return None # No landmarks yet
        
    min_dist = float('inf')
    best_id = -1
    
    inv_cov = np.linalg.inv(OBS_COV)
    
    for i, lm_pos in enumerate(map_landmarks):
        pred_z = get_expected_obs(pred_pose, lm_pos)
        diff = observation - pred_z
        diff[1] = normalize_angle(diff[1])
        
        # Mahalanobis distance
        d_mah = diff.T @ inv_cov @ diff
        
        if d_mah < min_dist:
            min_dist = d_mah
            best_id = i
            
    if min_dist < MAHALANOBIS_THRESHOLD:
        return best_id
    else:
        return None # New landmark

def build_error_vector(params, n_poses, n_landmarks, measurements, controls):
    """
    Cost function for Least Squares Optimization.
    Computes residuals for all motion and observation constraints.
    """
    # Unpack parameters
    est_poses = params[:n_poses*3].reshape((n_poses, 3))
    est_lms = params[n_poses*3:].reshape((n_landmarks, 2))
    
    residuals = []
    
    # 1. Motion residuals (Odometry constraints)
    # Note: poses[0] is fixed (prior), start error check from poses[1]
    weight_motion = [1.0/ODOM_V_NOISE, 1.0/ODOM_V_NOISE, 1.0/ODOM_W_NOISE] 
    
    for t in range(1, n_poses):
        prev_p = est_poses[t-1]
        curr_p = est_poses[t]
        u = controls[t-1] # Control that moved t-1 to t
        
        # Predicted current pose based on prev pose + control
        pred_p = motion_model(prev_p, u)
        
        res = curr_p - pred_p
        res[2] = normalize_angle(res[2])
        
        # Apply simple weighting
        residuals.extend(res * weight_motion)

    # 2. Observation residuals
    weight_obs = [1.0/R_NOISE, 1.0/PHI_NOISE]
    
    for meas in measurements:
        pose_idx = meas['pose_idx']
        lm_idx = meas['lm_idx']
        z = meas['z']
        
        # Where we think the landmark is vs where we saw it
        pred_z = get_expected_obs(est_poses[pose_idx], est_lms[lm_idx])
        
        res = z - pred_z
        res[1] = normalize_angle(res[1])
        
        residuals.extend(res * weight_obs)
        
    return np.array(residuals)

# --- 4. Main Simulation ---

def main():
    # A. Setup Ground Truth
    true_landmarks = np.array([
        [10, 10], [10, -5], [-5, 15], [-10, -10], [15, 0]
    ])
    
    true_pose = np.array([0.0, 0.0, 0.0]) # x, y, theta
    true_path = [true_pose]
    
    # B. Initialization
    # Estimate Variables (SLAM State)
    est_pose = np.array([0.0, 0.0, 0.0])
    est_path = [est_pose]
    est_landmarks = [] # List of [x, y]
    
    # History for Graph
    controls = []      # u_0, u_1 ...
    measurements = []  # List of dicts {pose_idx, lm_idx, z}
    
    # C. Run Loop
    print(f"Simulating {SIM_STEPS} steps...")
    
    for t in range(SIM_STEPS):
        # 1. Move Robot (Ground Truth)
        # Add noise to command for actual movement
        v_act = V_CMD + np.random.normal(0, V_NOISE)
        w_act = W_CMD + np.random.normal(0, W_NOISE)
        true_pose = motion_model(true_pose, [v_act, w_act])
        true_path.append(true_pose)
        
        # 2. Odometry (Dead Reckoning Estimate)
        # What the robot *thinks* it did
        u_noisy = [V_CMD, W_CMD] # Simplified: usually you read encoders
        est_pose = motion_model(est_pose, u_noisy)
        est_path.append(est_pose)
        controls.append(u_noisy)
        
        # 3. Sense
        obs = observe_landmarks(true_pose, true_landmarks)
        
        # 4. SLAM Front-End (Data Association)
        for observation in obs:
            z = observation['z']
            
            # Try to associate with existing map
            lm_id = data_association(est_pose, z, est_landmarks)
            
            if lm_id is None:
                # New Landmark!
                lm_global = inverse_sensor_model(est_pose, z)
                est_landmarks.append(lm_global)
                lm_id = len(est_landmarks) - 1
                # print(f"Step {t}: New Landmark ID {lm_id} found.")
            
            # Add measurement to graph
            measurements.append({
                'pose_idx': t + 1, # +1 because pose 0 is start
                'lm_idx': lm_id,
                'z': z
            })

    # D. SLAM Back-End (Optimization)
    print("Optimizing Graph...")
    
    # Flatten initial guess into single parameter vector
    n_poses = len(est_path)
    n_lms = len(est_landmarks)
    
    x0_poses = np.array(est_path).flatten()
    x0_lms = np.array(est_landmarks).flatten()
    x0 = np.concatenate((x0_poses, x0_lms))
    
    # Optimize
    res = least_squares(
        build_error_vector, 
        x0, 
        args=(n_poses, n_lms, measurements, controls),
        verbose=1
    )
    
    # Unpack optimized result
    opt_params = res.x
    opt_poses = opt_params[:n_poses*3].reshape((n_poses, 3))
    opt_lms = opt_params[n_poses*3:].reshape((n_lms, 2))
    
    print("Optimization Complete.")

    # --- 5. Plotting ---
    true_path = np.array(true_path)
    est_path = np.array(est_path)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Landmarks
    plt.scatter(true_landmarks[:,0], true_landmarks[:,1], c='k', marker='*', s=200, label='True Landmarks')
    if len(est_landmarks) > 0:
        est_lms_arr = np.array(est_landmarks)
        plt.scatter(est_lms_arr[:,0], est_lms_arr[:,1], c='r', marker='x', s=100, label='Dead Reckoning LMs')
    plt.scatter(opt_lms[:,0], opt_lms[:,1], c='b', marker='o', s=100, facecolors='none', label='SLAM Optimized LMs')

    # Plot Paths
    plt.plot(true_path[:,0], true_path[:,1], 'k--', label='Ground Truth Path')
    plt.plot(est_path[:,0], est_path[:,1], 'r-', label='Dead Reckoning (Odom)')
    plt.plot(opt_poses[:,0], opt_poses[:,1], 'b-', linewidth=2, label='SLAM Optimized Path')
    
    plt.title("Graph SLAM with Unknown Landmark Correspondence")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
