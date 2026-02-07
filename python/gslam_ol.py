import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --- 1. Configuration ---
DT = 1.0
SIM_STEPS = 60
V_CMD = 1.0
W_CMD = 0.15     # Slightly tighter turn to ensure loop closure
R_NOISE = 0.05
PHI_NOISE = 0.03
ODOM_V_NOISE = 0.02
ODOM_W_NOISE = 0.02
MAHALANOBIS_THRESHOLD = 3.0

# --- 2. Helper Functions (Math) ---
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def motion_model(pose, control):
    x, y, theta = pose
    v, w = control
    if abs(w) < 1e-5:
        return np.array([x + v*np.cos(theta)*DT, y + v*np.sin(theta)*DT, theta])
    
    new_x = x + (v/w) * (np.sin(theta + w*DT) - np.sin(theta))
    new_y = y + (v/w) * (np.cos(theta) - np.cos(theta + w*DT))
    new_theta = normalize_angle(theta + w*DT)
    return np.array([new_x, new_y, new_theta])

def get_expected_obs(pose, lm_pos):
    dx = lm_pos[0] - pose[0]
    dy = lm_pos[1] - pose[1]
    r = np.sqrt(dx**2 + dy**2)
    phi = normalize_angle(np.arctan2(dy, dx) - pose[2])
    return np.array([r, phi])

def inverse_sensor_model(pose, z):
    x, y, theta = pose
    r, phi = z
    return np.array([x + r*np.cos(theta+phi), y + r*np.sin(theta+phi)])

# --- 3. Online SLAM Class ---

class OnlineGraphSLAM:
    def __init__(self, initial_pose):
        # State: List of poses [[x,y,th], ...] and Landmarks [[x,y], ...]
        self.est_poses = [initial_pose]
        self.est_lms = []
        
        # Graph History
        self.controls = []      # List of [v, w]
        self.measurements = []  # List of {pose_idx, lm_idx, z}
        
        # Covariances for weighting
        self.odom_weight = np.array([1.0/ODOM_V_NOISE, 1.0/ODOM_V_NOISE, 1.0/ODOM_W_NOISE])
        self.obs_weight = np.array([1.0/R_NOISE, 1.0/PHI_NOISE])

    def predict(self, control):
        """1. Prediction Step (Odometry)"""
        last_pose = self.est_poses[-1]
        new_pose = motion_model(last_pose, control)
        
        self.est_poses.append(new_pose)
        self.controls.append(control)
        return new_pose

    def update(self, observations):
        """2. Update Step (Data Association + Optimization)"""
        current_pose_idx = len(self.est_poses) - 1
        current_pose = self.est_poses[-1]
        
        # A. Data Association
        for z_raw in observations:
            z = z_raw['z']
            
            best_id = -1
            min_dist = float('inf')
            
            # Check against existing landmarks
            for i, lm in enumerate(self.est_lms):
                pred_z = get_expected_obs(current_pose, lm)
                diff = z - pred_z
                diff[1] = normalize_angle(diff[1])
                # Simple Mahalanobis-like distance (ignoring full covariance matrix for speed)
                dist = np.sum((diff * self.obs_weight)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_id = i
            
            # Threshold check
            if min_dist > MAHALANOBIS_THRESHOLD**2 or len(self.est_lms) == 0:
                # NEW LANDMARK
                new_lm = inverse_sensor_model(current_pose, z)
                self.est_lms.append(new_lm)
                best_id = len(self.est_lms) - 1
            
            # Add constraint to graph
            self.measurements.append({
                'pose_idx': current_pose_idx,
                'lm_idx': best_id,
                'z': z
            })
            
        # B. Online Optimization
        # Only run if we have enough constraints to make it worthwhile
        if len(self.measurements) > 0:
            self.optimize_graph()

    def optimize_graph(self):
        """Solves the full graph accumulated so far."""
        n_poses = len(self.est_poses)
        n_lms = len(self.est_lms)
        
        # Flatten state into 1D array
        x0_poses = np.array(self.est_poses).flatten()
        x0_lms = np.array(self.est_lms).flatten()
        x0 = np.concatenate((x0_poses, x0_lms))
        
        # Run Levenberg-Marquardt
        res = least_squares(
            self.error_function, 
            x0, 
            args=(n_poses, n_lms),
            xtol=1e-4, ftol=1e-4, verbose=0 
        )
        
        # Update State with optimized results
        opt_poses = res.x[:n_poses*3].reshape((n_poses, 3))
        opt_lms = res.x[n_poses*3:].reshape((n_lms, 2))
        
        self.est_poses = list(opt_poses)
        self.est_lms = list(opt_lms)

    def error_function(self, params, n_poses, n_lms):
        # Unpack
        p = params[:n_poses*3].reshape((n_poses, 3))
        l = params[n_poses*3:].reshape((n_lms, 2))
        
        residuals = []
        
        # Motion Errors
        for t in range(1, n_poses):
            prev = p[t-1]
            curr = p[t]
            u = self.controls[t-1]
            pred = motion_model(prev, u)
            err = curr - pred
            err[2] = normalize_angle(err[2])
            residuals.extend(err * self.odom_weight)
            
        # Observation Errors
        for m in self.measurements:
            pose_idx = m['pose_idx']
            lm_idx = m['lm_idx']
            z = m['z']
            pred_z = get_expected_obs(p[pose_idx], l[lm_idx])
            err = z - pred_z
            err[1] = normalize_angle(err[1])
            residuals.extend(err * self.obs_weight)
            
        return np.array(residuals)

# --- 4. Simulation & Animation ---

def run_online_slam():
    # Setup
    true_lms = np.array([[10, 10], [10, -5], [-5, 15], [-10, -10], [15, 0], [0, 12]])
    true_pose = np.array([0.0, 0.0, 0.0])
    
    slam = OnlineGraphSLAM(initial_pose=true_pose)
    
    # Visualization Init
    plt.ion() # Interactive Mode On
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for t in range(SIM_STEPS):
        # 1. Simulate Reality
        v_act = V_CMD + np.random.normal(0, ODOM_V_NOISE)
        w_act = W_CMD + np.random.normal(0, ODOM_W_NOISE)
        true_pose = motion_model(true_pose, [v_act, w_act])
        
        # Generate Observations
        obs_data = []
        for i, lm in enumerate(true_lms):
            dist = np.linalg.norm(lm - true_pose[:2])
            if dist < 15.0: # Sensor Range
                # Add noise
                d_n = dist + np.random.normal(0, R_NOISE)
                b_true = np.arctan2(lm[1]-true_pose[1], lm[0]-true_pose[0]) - true_pose[2]
                b_n = normalize_angle(b_true + np.random.normal(0, PHI_NOISE))
                obs_data.append({'z': np.array([d_n, b_n])})

        # 2. Run SLAM
        # A. Predict
        slam.predict([V_CMD, W_CMD]) # Pass noisy control in real usage
        # B. Update
        slam.update(obs_data)
        
        # 3. Visualization
        if t % 2 == 0: # Update plot every 2 steps for speed
            ax.clear()
            
            # Ground Truth
            ax.scatter(true_lms[:,0], true_lms[:,1], c='k', marker='*', s=150, label='True Landmarks')
            
            # SLAM Estimate
            poses = np.array(slam.est_poses)
            ax.plot(poses[:,0], poses[:,1], 'b-', linewidth=2, label='Online Trajectory')
            
            # Current Robot
            ax.arrow(poses[-1,0], poses[-1,1], 
                     2*np.cos(poses[-1,2]), 2*np.sin(poses[-1,2]), 
                     head_width=1.0, color='b')
            
            if len(slam.est_lms) > 0:
                lms = np.array(slam.est_lms)
                ax.scatter(lms[:,0], lms[:,1], c='r', marker='o', s=80, facecolors='none', label='Estimated LMs')
                
            ax.legend(loc='upper right')
            ax.set_title(f"Online Graph SLAM - Step {t}/{SIM_STEPS}")
            ax.axis('equal')
            ax.grid(True)
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_online_slam()
