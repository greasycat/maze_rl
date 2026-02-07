import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --- 1. Configuration ---
DT = 1.0
SIM_STEPS = 100
V_CMD = 1.0
W_CMD = 0.15     # Slightly tighter turn to ensure loop closure
R_NOISE = 0.03
PHI_NOISE = 0.03
ODOM_V_NOISE = 0.02
ODOM_W_NOISE = 0.01
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
                print(f"Drift Alert! Creating NEW landmark. Min Dist was {min_dist:.2f} (Threshold: {MAHALANOBIS_THRESHOLD**2})")
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
        """Solves the full graph accumulated so far (Method 1: Fixed Gauge)."""
        n_poses = len(self.est_poses)
        n_lms = len(self.est_lms)
        
        # --- CHANGE 1: Prepare x0 without the first pose ---
        # We slice [1:] to skip the first pose (the anchor)
        x0_poses = np.array(self.est_poses[1:]).flatten()
        x0_lms = np.array(self.est_lms).flatten()
        x0 = np.concatenate((x0_poses, x0_lms))
        
        # Run Levenberg-Marquardt
        # We still pass 'n_poses' (total) so error_function knows how many to reconstruct
        res = least_squares(
            self.error_function, 
            x0, 
            args=(n_poses, n_lms),
            xtol=1e-4, ftol=1e-4, verbose=0 
        )
        
        # --- CHANGE 2: Unpack and re-insert the anchor ---
        # Calculate how many poses were actually optimized
        n_opt = n_poses - 1
        
        # Get the optimized part
        opt_poses_part = res.x[:n_opt*3].reshape((n_opt, 3))
        opt_lms = res.x[n_opt*3:].reshape((n_lms, 2))
        
        # Reconstruct the full pose list: [Start] + [Optimized Rest]
        # Note: We use self.est_poses[0] as the source of truth for the start
        full_poses = np.vstack([self.est_poses[0], opt_poses_part])
        
        # Update state
        self.est_poses = list(full_poses)
        self.est_lms = list(opt_lms)

    def error_function(self, params, n_poses, n_lms):
        # --- CHANGE 3: Reconstruct Full Trajectory ---
        
        # 1. Determine optimized count (Total - 1)
        n_opt = n_poses - 1
        
        # 2. Unpack the variables (Skipping the anchor)
        p_opt = params[:n_opt*3].reshape((n_opt, 3))
        l = params[n_opt*3:].reshape((n_lms, 2))
        
        # 3. Create the Anchor (The Fixed Start Pose)
        # Ideally, use the class's actual start, but strictly following your snippet:
        start_pose = np.array([0.0, 0.0, 0.0]) 
        
        # 4. Stack them together: p becomes the full list of N poses again
        p = np.vstack([start_pose, p_opt])
        
        residuals = []
        
        # --- Logic below remains exactly the same ---
        # Because 'p' is now the full size (n_poses), all indexing works as before.
        
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

def get_square_control(t):
    """
    Returns [v, w] based on the current time step t to form a square.
    Pattern: 15 steps straight, 5 steps turning.
    """
    period = 20         # Total duration of one side + one corner
    step_in_cycle = t % period
    
    # 1. Move Straight (Steps 0 to 14)
    if step_in_cycle < 15:
        v = 1.0
        w = 0.0
    # 2. Turn 90 degrees (Steps 15 to 19)
    else:
        # We need to turn pi/2 radians in 5 steps.
        # w = angle / (steps * dt)
        turn_steps = 5
        target_angle = np.pi / 2
        v = 0.1             # Move slightly while turning (optional, but smoother)
        w = target_angle / (turn_steps * 1.0) # Assuming DT=1.0
        
    return [v, w]

def run_online_square_slam():
    # 1. Setup - Square Configuration
    # We increase steps to 100 to ensure we complete the full square (4 sides * 20 steps = 80)
    
    # Place landmarks in a square formation so we have things to see
    true_lms = np.array([
        [5, 5], [10, 5], [15, 5],    # Top side
        [15, -5], [15, -10],         # Right side
        [10, -10], [5, -10], [0, -10], # Bottom side
        [0, -5], [0, 0]              # Left side
    ])
    
    true_pose = np.array([0.0, 0.0, 0.0]) # Start at 0,0
    
    slam = OnlineGraphSLAM(initial_pose=true_pose)
    
    # Visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    print("Starting Square Simulation...")

    for t in range(SIM_STEPS):
        # --- A. Determine Control for this Step ---
        # Get the perfect control for a square
        clean_control = get_square_control(t)
        
        # Add noise to creating "Actual" movement (Reality)
        v_act = clean_control[0] + np.random.normal(0, ODOM_V_NOISE)
        w_act = clean_control[1] + np.random.normal(0, ODOM_W_NOISE)
        
        # Move "True" Robot
        true_pose = motion_model(true_pose, [v_act, w_act])
        
        # --- B. Generate Observations ---
        obs_data = []
        for i, lm in enumerate(true_lms):
            # Calculate distance to landmark
            dist = np.linalg.norm(lm - true_pose[:2])
            
            # Simulate Sensor Range (e.g., 10 meters)
            if dist < 10.0: 
                # Add sensor noise
                d_n = dist + np.random.normal(0, R_NOISE)
                b_true = np.arctan2(lm[1]-true_pose[1], lm[0]-true_pose[0]) - true_pose[2]
                b_n = normalize_angle(b_true + np.random.normal(0, PHI_NOISE))
                obs_data.append({'z': np.array([d_n, b_n])})

        # --- C. Run SLAM ---
        # Note: We pass the "clean" control (or read from encoders) to the SLAM prediction
        # We do NOT pass the actual noisy control.
        slam.predict(clean_control) 
        slam.update(obs_data)
        
        # --- D. Visualize ---
        if t % 2 == 0:
            ax.clear()
            
            # 1. Plot Landmarks
            ax.scatter(true_lms[:,0], true_lms[:,1], c='k', marker='*', s=150, label='True Landmarks')
            
            # 2. Plot SLAM Trajectory
            # This will look wobbly at first, then snap into a square upon loop closure
            poses = np.array(slam.est_poses)
            ax.plot(poses[:,0], poses[:,1], 'b-', linewidth=2, label='SLAM Trajectory')
            
            # 3. Plot Dead Reckoning (Just for comparison, optional)
            # You can see how bad the square would look without SLAM
            # (Requires tracking a separate list of pure odometry poses, omitted here for brevity)
            
            # 4. Current Robot Heading
            ax.arrow(poses[-1,0], poses[-1,1], 
                     2*np.cos(poses[-1,2]), 2*np.sin(poses[-1,2]), 
                     head_width=0.5, color='b')
            
            # 5. Estimated Landmarks
            if len(slam.est_lms) > 0:
                lms = np.array(slam.est_lms)
                ax.scatter(lms[:,0], lms[:,1], c='r', marker='o', s=80, facecolors='none', label='Estimated LMs')
                
            ax.legend(loc='upper right')
            ax.set_title(f"Online Square SLAM - Step {t}/{SIM_STEPS}")
            ax.axis('equal')
            ax.grid(True)
            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_online_square_slam()
