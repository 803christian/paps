import sys
sys.path.append('../')

import numpy as np
import open3d as o3d
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import trimesh as tm
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import time
#import math
import os
import shutil
#from scipy.spatial.transform import Rotation
from jax import vmap, lax
import jax.numpy as jnp
from scipy.spatial import distance_matrix
import cv2
import mediapy as media
from scipy.spatial.transform import Rotation as R

from emmd.dynamic_emmd import Dynamic_EMMD
import sfm.image_to_cloud as image_to_cloud
import sfm.opensfm as opensfm

# /usr/local/cuda/lib64/libcudnn
# echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64/cudnn${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
# source ~/.bashrc



class SFM_EMMD():
    def __init__(self, args):
        self.args = args
        self.T = self.args['tf'] / self.args['dt']

        self.load_mesh()
        
    def load_mesh(self, mesh_path = None):
        self.mesh_path = self.args['mesh_path']

        if mesh_path is not None:
            self.mesh = tm.load_mesh(mesh_path)
        else:
            self.mesh = tm.load_mesh(self.mesh_path)
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces

        if self.args['scaling'] == 1:
            self.scale_mesh()

        self.mesh_bound = np.max(self.mesh.bounds)

    def scale_mesh(self):
        # Get the bounding box of the mesh
        bounds_min, bounds_max = self.mesh.bounds
        extents = bounds_max - bounds_min

        # Find the maximum extent (largest dimension)
        max_extent = np.max(extents)

        # Compute the scaling factor
        self.scaling_factor = 1.0 / max_extent

        # Scale the vertices of the mesh
        self.mesh.vertices *= self.scaling_factor

    def solve_trajectory(self):
        if self.args['utility'] == 'uniform':
            self.args['info_dist'] = lambda x : 1
            self.P_XI = vmap(self.args['info_dist'], in_axes=(0,))#(jnp.arange(len(self.mesh.vertices)))
            self.args['P_XI'] = self.P_XI
            self.P_XI = self.P_XI(self.mesh.vertices)
            self.P_XI = self.P_XI / jnp.sum(self.P_XI)

        elif self.args['utility'] == 'heuristic':
            mesh_points = jnp.asarray(self.mesh.vertices)
            mesh_normals = jnp.asarray(self.mesh.vertex_normals)

            def compute_local_features(vertex_idx, k_neighbors=10):
                # Compute distances to all other points
                distances = jnp.linalg.norm(mesh_points - mesh_points[vertex_idx], axis=1)
                
                # Get k nearest neighbors (excluding self)
                _, neighbor_indices = lax.top_k(-distances, k_neighbors + 1)
                neighbor_indices = neighbor_indices[1:]  # Exclude self
                neighbor_distances = distances[neighbor_indices]
                
                # Compute local point density
                local_density = jnp.mean(neighbor_distances)
                
                # Compute normal consistency
                center_normal = mesh_normals[vertex_idx]
                neighbor_normals = mesh_normals[neighbor_indices]
                normal_consistency = jnp.mean(jnp.abs(jnp.dot(neighbor_normals, center_normal)))
                
                # Compute edge sharpness
                edge_vectors = mesh_points[neighbor_indices] - mesh_points[vertex_idx]
                edge_directions = edge_vectors / jnp.linalg.norm(edge_vectors, axis=1, keepdims=True)
                edge_sharpness = jnp.std(jnp.abs(jnp.dot(edge_directions, center_normal)))
                
                return local_density, normal_consistency, edge_sharpness

            # Compute features for all vertices
            features = vmap(compute_local_features)(jnp.arange(len(mesh_points)))
            local_densities, normal_consistencies, edge_sharpnesses = features

            # Normalize features to [0, 1] range
            def normalize(x):
                return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))

            norm_densities = normalize(local_densities)
            norm_consistencies = normalize(normal_consistencies)
            norm_sharpnesses = normalize(edge_sharpnesses)

            # Combine features to create utility
            utility = (0.1 * (1 - norm_densities) +  # Invert density so holes have high utility
                    0.6 * (1 - norm_consistencies) +  # Invert consistency so discontinuities have high utility
                    0.3 * norm_sharpnesses)  # Sharp edges still contribute, but less

            # Normalize final utility to [0, 1] range
            self.P_XI = normalize(utility)
            self.args['P_XI'] = self.P_XI
        elif self.args['utility'] == 'distribution':
            """
            def generate_utility_distribution_3d(data, spread_factor=0.25):
                
                def utility_distribution(point):
                    #Compute the utility value at a single point based on Gaussian peaks in `data`.
                    
                    def single_gaussian(peak):
                        x_coord, y_coord, z_coord, amplitude = peak
                        dist_sq = (
                            (point[0] - x_coord) ** 2 +
                            (point[1] - y_coord) ** 2 +
                            (point[2] - z_coord) ** 2
                        )
                        gaussian_value = amplitude * jnp.exp(-dist_sq / (2 * spread_factor ** 2))
                        return gaussian_value

                    # Apply the Gaussian calculation across all peaks in `data`
                    gaussian_values = vmap(single_gaussian)(jnp.array(data))
                    # Take the maximum value across all Gaussian contributions for this point
                    return jnp.max(gaussian_values)

                # Return a lambda function to compute the utility distribution on any set of 3D points
                return lambda points: vmap(utility_distribution)(points)

            self.P_XI = generate_utility_distribution_3d(self.args['data'])
            #self.P_XI = vmap(lambda x: self.args['info_dist'](x, self.args['data']), in_axes=(0,))(jnp.array([[vertex[0], vertex[1]] for vertex in self.mesh.vertices]))#(jnp.arange(len(self.mesh.vertices))) #self.args['P_XI']
            self.args['P_XI'] = self.P_XI
            """
            def info_distr(x):
                P_XI = []
                data = np.array(self.args['data'])
                for n in range(len(x)):
                    value = []
                    for i in range(len(data)):
                        x0 = data[i][0]
                        y0 = data[i][1]
                        sigma = data[i][3]
                        amplitude = data[i][2]
                        value.append(amplitude * np.exp(-((x[n][0] - x0) ** 2 + (x[n][2] - y0) ** 2) / (2 * sigma ** 2)))
                    P_XI.append(np.max(value))
                return jnp.array(P_XI)
            self.P_XI = info_distr(self.mesh.vertices)
            self.P_XI = self.P_XI / jnp.sum(self.P_XI)
            self.args['P_XI'] = info_distr

        if self.args['solver'] == 'emmd':
            emmd = Dynamic_EMMD(self.args)
            self.trajectory = emmd.solve()
        elif self.args['solver'] == 'tsp':
            if self.args['utility'] == 'uniform':
                epsilon = 0
            else:
                epsilon = 0.0002            

            print(f'Sampling {self.args["num_points"]} points...')

            valid_indices = np.where(self.P_XI > epsilon)[0]

            filtered_points = self.mesh.vertices[valid_indices]     

            if filtered_points.shape[0] > self.args['num_points']:
                vertex_indices = np.random.choice(len(filtered_points), self.args['num_points'], replace=False)
                filtered_points = filtered_points[vertex_indices]

            # TSP: Distance matrix computation
            dist_matrix = distance_matrix(filtered_points, filtered_points)

            # Solve the TSP using the nearest neighbor heuristic or linear sum assignment
            def solve_tsp(dist_matrix):
                # Use a simple nearest neighbor heuristic for demonstration
                num_points = dist_matrix.shape[0]
                visited = np.zeros(num_points, dtype=bool)
                tsp_path = [0]  # Start from the first point
                visited[0] = True

                for _ in range(1, num_points):
                    last = tsp_path[-1]
                    next_point = np.argmin(dist_matrix[last, ~visited])
                    tsp_path.append(np.where(~visited)[0][next_point])
                    visited[tsp_path[-1]] = True
                
                # Close the loop by returning to the starting point
                tsp_path.append(tsp_path[0])
                return np.array(tsp_path)

            # Get the TSP path for the filtered points
            path = solve_tsp(dist_matrix)
            self.trajectory = filtered_points[path] 

        elif self.args['solver'] == 'info_max':
            # Number of points in the trajectory
            n_points = self.args.get('n_trajectory_points', int(self.T))
            
            # Initialize the trajectory
            trajectory = []
            visited = set()

            # Start from the point with the highest information
            current_index = int(jnp.argmax(self.P_XI))  # Convert to Python int
            trajectory.append(self.mesh.vertices[current_index])
            visited.add(current_index)

            for _ in range(1, n_points):
                # Find the neighbors of the current point
                distances = jnp.linalg.norm(jnp.array(self.mesh.vertices) - trajectory[-1], axis=1)
                neighbor_indices = jnp.argsort(distances)[:20].tolist()  # Convert to Python list
                
                # Filter out visited neighbors
                unvisited_neighbors = [idx for idx in neighbor_indices if idx not in visited]
                
                if not unvisited_neighbors:
                    # If all neighbors are visited, jump to the highest information unvisited point
                    unvisited_indices = list(set(range(len(self.mesh.vertices))) - visited)
                    if not unvisited_indices:  # Check if there are no unvisited points
                        print("All points have been visited. Ending trajectory generation early.")
                        break
                    current_index = max(unvisited_indices, key=lambda idx: self.P_XI[idx])
                else:
                    # Choose the unvisited neighbor with the highest information
                    current_index = max(unvisited_neighbors, key=lambda idx: self.P_XI[idx])
                
                trajectory.append(self.mesh.vertices[current_index])
                visited.add(current_index)

            self.trajectory = np.array(trajectory)


    def interpolate_trajectory(self, trajectory, new_length):
        # Get the original number of points (n)
        n = trajectory.shape[0]
        new_length = int(new_length)
        
        # Debug: check the shape of trajectory
        print(f"\nOriginal trajectory shape: {trajectory.shape}")

        # Ensure trajectory has at least two points to interpolate
        if n < 2:
            raise ValueError("Trajectory must have at least two points for interpolation.")
        
        # Create the original index range (from 0 to n-1)
        original_indices = np.linspace(0, n - 1, n)
        
        # Create the new index range for interpolation (from 0 to n-1 but with new_length points)
        new_indices = np.linspace(0, n - 1, new_length)
        
        # Interpolate each axis (x, y, z) separately using scipy's interp1d
        interpolated_trajectory = np.zeros((new_length, 3))
        
        for i in range(3):  # Interpolating x, y, and z coordinates
            # Check if the trajectory dimension is correct
            if trajectory[:, i].shape[0] != n:
                raise ValueError(f"Mismatch in lengths: expected {n}, got {trajectory[:, i].shape[0]}")
            
            interpolator = interp1d(original_indices, trajectory[:, i], kind='linear')
            interpolated_trajectory[:, i] = interpolator(new_indices)
        
        self.trajectory = interpolated_trajectory

        print(f"Interpolated trajectory shape: {self.trajectory.shape}")

    def extract_directions(self):
        # Build a KD-tree for fast nearest-neighbor search
        kdtree = KDTree(self.mesh.vertices)

        # Ensure trajectory is 2D: (n_points, 3)
        self.trajectory = self.trajectory.reshape(-1, 3)

        # Find the nearest mesh points for each trajectory point
        trajectory_indices = kdtree.query(self.trajectory)[1]

        # Get normals corresponding to the nearest mesh points
        mesh_normals = np.asarray(self.mesh.vertex_normals)
        
        # Ensure normals are correctly aligned with trajectory
        traj_direction = -mesh_normals[trajectory_indices]  # Inverse the normals for arrow direction
        
        # Normalize each vector
        traj_direction = traj_direction / np.linalg.norm(traj_direction, axis=1, keepdims=True)
        self.direction = traj_direction

    def extract_gradient(self): # sub extract_directions (gradients method)
        def compute_kernel_gradients_single(x, xp, h=0.01):
            # x is a single 3D point, xp is an array of 3D points
            # Ensure xp is a numpy array
            xp = jnp.array(xp)
            
            # Compute the distance between the single point x and all points in xp
            dist = jnp.sum((xp - x)**2, axis=1)
            kernel_value = jnp.exp(-dist / h)
            
            # Gradient with respect to x
            grad_x = (2 * (x - xp) * kernel_value[:, None]) / h
            
            return grad_x
        
        def compute_kernel_gradients(x, xp, h=0.01):
            
            x = jnp.array(x)  # Ensure x is a numpy array
            xp = jnp.array(xp)  # Ensure xp is a numpy array
            
            # Use vmap to apply compute_kernel_gradients_single to each point in x
            return vmap(lambda x_i: compute_kernel_gradients_single(x_i, xp, h))(x)
        
        direction = np.zeros((len(self.trajectory), 3))
        traj_jax = np.array(self.trajectory)
        kernel_gradients = compute_kernel_gradients(traj_jax, traj_jax, h=0.01)
        traj_gradients = np.array(kernel_gradients)
        for i in range(len(self.trajectory)):
            direction[i] = - traj_gradients[i].mean(axis=0)
            if np.linalg.norm(direction[i]) > 0:
                direction[i] /= np.linalg.norm(direction[i])

        self.direction = direction

    def direction_to_quaternion(self):
        self.quat = np.zeros((len(self.direction), 4))
        
        # Define a fixed reference direction (e.g., [0, 0, 1] for 'up')
        reference_direction = np.array([0, -1, 0])
        
        for i in range(len(self.direction)):
            direction = np.array(self.direction[i]).squeeze()
            
            # Normalize the direction vector
            direction = direction / np.linalg.norm(direction)
            
            # Compute the rotation axis
            rotation_axis = np.cross(reference_direction, direction)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm < 1e-6:
                # The vectors are either parallel or anti-parallel
                if np.dot(reference_direction, direction) > 0:
                    # Vectors are parallel, no rotation needed
                    self.quat[i] = [1, 0, 0, 0]
                else:
                    # Vectors are anti-parallel, rotate 180 degrees around any perpendicular axis
                    self.quat[i] = [0, 1, 0, 0]  # 180 degree rotation around x-axis
            else:
                # Normalize the rotation axis
                rotation_axis = rotation_axis / rotation_axis_norm
                
                # Compute the rotation angle
                cos_angle = np.clip(np.dot(reference_direction, direction), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Compute the quaternion
                sin_half_angle = np.sin(angle / 2)
                cos_half_angle = np.cos(angle / 2)
                
                self.quat[i] = [
                    cos_half_angle,
                    rotation_axis[0] * sin_half_angle,
                    rotation_axis[1] * sin_half_angle,
                    rotation_axis[2] * sin_half_angle
                ]
            
            # Ensure the quaternion is normalized
            self.quat[i] = self.quat[i] / np.linalg.norm(self.quat[i])
    
    def mjc_sim(self, model_path='../meshes/vid_cam/scene.xml', camera_name='block_cam'):
        def update_progress(percent):
            """Update the progress bar on the same line."""
            sys.stdout.write('\033[1A')  # Move cursor up one line
            sys.stdout.write('\033[K')   # Clear the line
            sys.stdout.write(f"{percent}% Complete\n")
            sys.stdout.flush()
        
        # Load MuJoCo camera element
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        #camera_name = 'block_cam'
        camera_id = model.camera(camera_name).id

        # Render MJC images
        with mujoco.Renderer(model) as renderer:
            for number in range(len(self.trajectory)):

                data.mocap_pos[0] = np.array(self.trajectory[number])
                data.mocap_quat[0] = np.array(self.quat[number])
                mujoco.mj_forward(model, data)

                renderer.update_scene(data, camera=camera_id)
                image = renderer.render()
                media.write_image(f'images/pictcha{number}.png', image)

                percent = 100*number/len(self.trajectory)
                update_progress(percent)

    def mjc_viewer(self, model_path='../meshes/vid_cam/scene.xml', camera_id='track'):
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        model.opt.timestep = 0.02
        site_id = model.site("imu").id
        mocap_id = model.body("target").mocapid[0]

        step = 0
        # Construct Kp & Kd
        Kp = np.diag([55, 55, 55]) 
        Kd = np.diag([10, 10, 10]) #np.diag([10, 10, 30])     
        Ki = np.diag([2, 2, 2])  
        integral_error = np.zeros(3)        
        number = 0

        with mujoco.Renderer(model, width=1280, height=720) as renderer:
            with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
                self.drone_path = np.zeros((len(self.trajectory), 3))
                self.drone_euler = np.zeros((len(self.trajectory), 3))

                # Reset the simulation.
                mujoco.mj_resetDataKeyframe(model, data, 0)
                # Reset the free camera.
                mujoco.mjv_defaultFreeCamera(model, viewer.cam)
                # Enable site frame visualization.
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
                viewer.opt.sitegroup[4] = 1
                viewer.opt.geomgroup[4] = 1
                
                # Configure viewer camera to the specific camera by ID
                #viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                #viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)  # Use the specified camera

                mjv_camera = viewer.cam

                step = 0
                
                # update renderer to render depth
                renderer.enable_depth_rendering()
                # reset the scene
                renderer.update_scene(data, camera=camera_id)
                # depth is a float array, in meters.
                depth = renderer.render()
                # Shift nearest values to the origin.
                #depth -= depth.min()
                # Scale by 2 mean distances of near rays.
                #depth /= 2*depth[depth <= 1].mean()
                # Scale to [0, 255]
                pixels = 255*np.clip(depth, 0, 1)
                #media.show_image(pixels.astype(np.uint8))
                media.write_image(f'../run/{str(self.args["room"])}/{str(self.args["solver"])}/{str(self.args["case"])}/{str(self.args["noise"])}/depth_images/img_{number}.png', pixels.astype(np.uint8))
                renderer.disable_depth_rendering()

                renderer.update_scene(data, camera=camera_id)
                image = renderer.render()
                media.write_image(f'../run/{str(self.args["room"])}/{str(self.args["solver"])}/{str(self.args["case"])}/{str(self.args["noise"])}/images/img_{number}.png', image)
                
                while viewer.is_running():
                    step_start = time.time()

                    if step < len(self.trajectory):
                        # Update target position from trajectory
                        data.mocap_pos[mocap_id] = self.trajectory[step]
                        self.drone_path[step] = data.site_xpos[site_id]


                        # Calculate rotation matrix from azimuth and elevation
                        rotation = R.from_euler(
                            'zy', [mjv_camera.azimuth, mjv_camera.elevation], degrees=True
                        )

                        # Convert rotation matrix to Euler angles
                        euler_angles = rotation.as_euler('xyz', degrees=True)  # Roll, pitch, yaw
                        self.drone_euler[step] = euler_angles

                    # Calculate position error and velocity error
                    error_pos = data.mocap_pos[mocap_id] - data.site_xpos[site_id]
                    error_vel = data.qvel[0:3]
                    
                    # Update integral term for position error (anti-windup by clamping)
                    integral_error += error_pos * model.opt.timestep
                    integral_error = np.clip(integral_error, -10, 10)  # Prevent runaway integral

                    # PID Control
                    control = np.matmul(Kp, error_pos) - np.matmul(Kd, error_vel) + np.matmul(Ki, integral_error)
                    
                    # Set control
                    data.ctrl = control #np.array([-907.36684872, -846.47204769, 15.87966635])

                    # Step the simulation
                    mujoco.mj_step(model, data)

                    viewer.sync()
                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                    step += 1
                    number += 1

                    # update renderer to render depth
                    renderer.enable_depth_rendering()
                    # reset the scene
                    renderer.update_scene(data, camera=camera_id)
                    # depth is a float array, in meters.
                    depth = renderer.render()
                    # Shift nearest values to the origin.
                    depth -= depth.min()
                    # Scale by 2 mean distances of near rays.
                    depth /= 2*depth[depth <= 1].mean()
                    # Scale to [0, 255]
                    pixels = 255*np.clip(depth, 0, 1)
                    #media.show_image(pixels.astype(np.uint8))
                    media.write_image(f'../run/{str(self.args["room"])}/{str(self.args["solver"])}/{str(self.args["case"])}/{str(self.args["noise"])}/depth_images/img_{number}.png', pixels.astype(np.uint8))
                    renderer.disable_depth_rendering()

                    renderer.update_scene(data) #camera=camera_id
                    image = renderer.render()
                    media.write_image(f'../run/{str(self.args["room"])}/{str(self.args["solver"])}/{str(self.args["case"])}/{str(self.args["noise"])}/images/img_{number}.png', image)

                    if step >= len(self.trajectory):
                        break
                viewer.close()

    def process_imgs(self):
        self.output_file = 'ply_files/multi_runs/filtered_'+str(self.args['solver'])+'_'+str(self.args['utility'])+'_'+str(self.args['mesher'])+'_'+str(self.T)+'.obj'
        file_name = str(self.output_file)

        if self.args['mesher'] == 'opensfm':
            print('Meshing with OpenSfM')
            notif = image_to_cloud.main()
            print(notif)

            pcd = o3d.io.read_point_cloud('output/reconstruction.ply')
            # Visualize the processed point cloud
            o3d.visualization.draw_geometries([pcd],
                zoom=0.3412,
                front=[0.4257, -0.2125, -0.8795],
                lookat=[2.6172, 2.0475, 1.532],
                up=[-0.0694, -0.9768, 0.2024])

            opensfm.main('output/reconstruction.ply', file_name, distance_threshold=self.args['distance_threshold'], min_neighbors=self.args['min_neighbors'], parallel_threshold=0.85, threshold_percent=40)
        elif self.args['mesher'] == 'viewpoint_synthesis':
            """
            A bare-bones implementation of viewpoint synthesis to generate a mesh from a series of images.
            
            :param images: List of input images (numpy arrays)
            :return: vertices, faces (mesh representation)
            """
            image_folder = 'images'

            # Step 1: Read images from the folder
            images = []
            image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for image_file in image_files:
                img_path = os.path.join(image_folder, image_file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)

            if len(images) < 2:
                raise ValueError("At least two images are required for viewpoint synthesis.")

            # Step 2: Feature detection and matching
            sift = cv2.SIFT_create()
            bf = cv2.BFMatcher()
            keypoints = []
            descriptors = []

            for img in images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = sift.detectAndCompute(gray, None)
                keypoints.append(kp)
                descriptors.append(des)

            # Step 3: Find corresponding points across images
            matches = []
            for i in range(len(images) - 1):
                matches.append(bf.knnMatch(descriptors[i], descriptors[i+1], k=2))

            # Apply ratio test
            good_matches = []
            for m in matches:
                good = []
                for pair in m:
                    if len(pair) == 2:
                        if pair[0].distance < 0.75 * pair[1].distance:
                            good.append(pair[0])
                good_matches.append(good)

            # Step 4: Triangulation
            points_3d = []
            for i in range(len(images) - 1):
                pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches[i]]).reshape(-1, 2)
                pts2 = np.float32([keypoints[i+1][m.trainIdx].pt for m in good_matches[i]]).reshape(-1, 2)
                
                # Ensure we have enough points
                if pts1.shape[0] < 5 or pts2.shape[0] < 5:
                    print(f"Not enough matching points between images {i} and {i+1}. Skipping this pair.")
                    continue

                # Estimate camera matrices
                E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
                
                # Check if E is valid
                if E is None or E.shape != (3, 3):
                    print(f"Invalid essential matrix between images {i} and {i+1}. Skipping this pair.")
                    continue

                # Recover pose
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2)

                # Create projection matrices
                P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                P2 = np.hstack((R, t))

                # Triangulate points
                points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
                
                # Convert to 3D points
                points_3d_homogeneous = points_4d.T
                points_3d_homogeneous /= points_3d_homogeneous[:, 3:]
                points_3d.extend(points_3d_homogeneous[:, :3].tolist())

            # Step 5: Create mesh (simplified - just creating vertices and faces)
            vertices = np.array(points_3d)

            # Create faces (simplified - just connecting nearby points)
            faces = []
            for i in range(len(vertices) - 2):
                faces.append([i, i+1, i+2])
            
            opensfm.write_ply('ply_files/multi_runs/ptcld.ply', vertices)
            opensfm.write_obj(file_name, vertices, faces)

        ## INPUT ELSE STATEMENTS HERE TO ADD VIEWPOINT SYNTHESIS AND GAUSSIAN PROCESS -----------------------------------------------------------------------------------------------------------------------------

        print('Mesh Generated')

    def plot_system(self):
        mesh = self.mesh
        trajectory = self.trajectory
        P_XI = self.P_XI  # Assuming you have set the utility distribution as a class attribute
        title = '['+str(self.args['solver'])+'] Solution Using ['+str(self.args['utility'])+'] Meshed With ['+str(self.args['mesher'])+'].'
        axis_bound = 1.5*self.mesh_bound

        # Extract vertices and faces from the trimesh mesh
        vertices = mesh.vertices
        faces = mesh.faces

        traj_direction = self.direction

        # Conditionally apply the color map to the mesh based on self.colormap
        if self.args['colormap'] == 1:
            # Create mesh trace with color mapping
            mesh_trace = go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=P_XI,   # Map the utility distribution to the mesh
                colorscale='Plasma',  # Choose a color scale
                opacity=1,  # Increased opacity for a cleaner look
                flatshading=False,  # Use smooth shading for better 3D look
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.9,
                    fresnel=0.1,
                    roughness=0.3,
                    specular=0.5
                ),
                lightposition=dict(x=100, y=200, z=300),
                name="Mesh",
                showscale=True,  # Show color scale
                colorbar=dict(title="Utility", titleside="right")
            )
        else:
            # Create a normal mesh trace without color mapping
            mesh_trace = go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='dimgrey',  # Default color if no colormap
                opacity=0.85,     # Slightly transparent
                flatshading=False,  # Use smooth shading for better 3D look
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.9,
                    fresnel=0.1,
                    roughness=0.3,
                    specular=0.5
                ),
                lightposition=dict(x=100, y=200, z=300),
                name="Mesh",
                showscale=False  # No color scale
            )

        # Create trajectory trace (as a scatter plot)
        trajectory_trace = go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
            mode='lines',
            marker=dict(
                size=6, 
                color='red', 
                symbol='circle', 
                line=dict(color='black', width=1),
            ),
            line=dict(
                color='lightblue',
                width=8
            ),
            name="Trajectory"
        )

        # Add arrows pointing to specific vertices of interest
        arrow_positions = np.array(trajectory)
        arrow_vectors = np.array(traj_direction)

        arrows = []
        for pos, vec in zip(arrow_positions, arrow_vectors):
            arrows.append(go.Cone(
                x=[pos[0]], y=[pos[1]], z=[pos[2]], 
                u=[vec[0][0]], v=[vec[0][1]], w=[vec[0][2]], # note::::: ADD INDEXING [0] BEFORE EACH INDEX OF VEC IF USING NORMALS, REMOVE IF GRADIENT
                showscale=False,  # Disable scaling
                colorscale=[[0, 'red'], [1, 'red']],  # Softer arrow colors for professionalism
                opacity=0.8,      # Slight transparency for a more subtle effect
                sizemode='absolute',
                sizeref=0.009,     # Adjusted size of the arrows for the scale
                anchor="tail",    # Anchor the arrow at its tail
                name="Arrow"
            ))

        # Set up layout with fixed axis bounds and improved aesthetics
        layout = go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(
                    visible=False,
                    showgrid=False,
                    gridcolor='lightgray',
                    zeroline=False,
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False,  # No spikes for a cleaner plot
                    range=[-axis_bound, axis_bound]  # Set fixed axis bounds [0, 1]
                ),
                yaxis=dict(
                    visible=False,
                    showgrid=False,
                    gridcolor='lightgray',
                    zeroline=False,
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False,
                    range=[-axis_bound, axis_bound]  # Set fixed axis bounds [0, 1]
                ),
                zaxis=dict(
                    visible=False,
                    showgrid=False,
                    gridcolor='lightgray',
                    zeroline=False,
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False,
                    range=[-axis_bound, axis_bound]  # Set fixed axis bounds [0, 1]
                ),
                aspectmode='cube',  # Maintain equal scaling for all axes
            ),
            legend=dict(
                x=0.8, y=0.9, 
                bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent legend background
                bordercolor='black', 
                borderwidth=1
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor='white',  # Set paper background to white
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        )

        # Create the figure with the mesh, trajectory, and arrows
        fig = go.Figure(data=[mesh_trace, trajectory_trace] + arrows, layout=layout)
        
        # Show the plot
        fig.show()

    def clear_contents(self):
        # Delete all files in the 'images' folder
        images_folder = 'images'
        if os.path.exists(images_folder):
            for file_name in os.listdir(images_folder):
                file_path = os.path.join(images_folder, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Delete all folders in the 'output' folder except for certain files
        output_folder = 'output'
        if os.path.exists(output_folder):
            for item_name in os.listdir(output_folder):
                item_path = os.path.join(output_folder, item_name)
                shutil.rmtree(item_path)

    def run_sfm(self):
        print('Solver: {}'.format(self.args['solver']))
        print('Utility: {}'.format(self.args['utility']))
        print('Mesher: {}\n'.format(self.args['mesher']))
        print('Starting SFM...\n')

        self.solve_trajectory()
        self.interpolate_trajectory(self.trajectory, self.T)
        self.extract_directions()
        self.plot_system()
        self.direction_to_quaternion()

        print('\nBeginning MJC simulation...\n')
    
        self.mjc_sim()

        print('\nMJC simulation complete. Starting mesh generation...')

        self.process_imgs()
        self.clear_contents()

        print('SFM complete.\n')

            
if __name__ == '__main__':
    mdl_path = '/home/christian/Downloads/sfm_sandbox/sfm_runs/ply_files/emmd_before_tries/bunny_T1000_mV0001.obj' #'/home/christian/Downloads/obj_and_ply/bun_zipper.ply'
    args = {'mesh_path': mdl_path,
            'num_points': 2000,
            'push': 0.175, 
            'max_v': 0.0005, 
            'scaling': 0,
            'distance_threshold': 0.2,
            'min_neighbors': 3, 
            'tf': 10,
            'dt': 0.02,
            'colormap': 1,
            'solver': 'emmd', # Options: emmd, tsp, info_max
            'mesher': 'viewpoint_synthesis', # Options: opensfm, viewpoint_synthesis, gaussian
            'utility': 'heuristic'} # Options: uniform, heuristic, gaussian

    sfm = SFM_EMMD(args)

    """
    sfm.solve_trajectory()
    print(f'Trajectory shape: {sfm.trajectory.shape}')
    print('Extracting directions...')
    sfm.extract_directions()
    print('Plotting...')

    sfm.plot_system()
    """

    sfm.run_sfm()