import sys
sys.path.append('../')

import numpy as np
import cv2
import os

from sfm.sfm_run import SFM_EMMD

def images_to_video(image_folder, output_video, fps=30):
    # Get list of image files
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Ensure the images are sorted in order
    
    # Read the first image to get dimensions
    if not images:
        print("No images found in the directory.")
        return
    
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        video.write(img)

    video.release()
    print(f"Video saved as {output_video}")

def generate_random_points(n_points, min_amplitude=0.1, max_amplitude=1.0):
    """
    Generate random points and amplitudes.
    
    :param n_points: Number of points to generate
    :param min_amplitude: Minimum amplitude value (default: 0.3)
    :param max_amplitude: Maximum amplitude value (default: 1.0)
    :return: List of tuples (x, y, amplitude)
    """
    data = []
    for _ in range(n_points):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        amplitude = np.random.uniform(min_amplitude, max_amplitude)
        sigma = 0.20
        data.append((x, y, amplitude, sigma))
    return data

import numpy as np

def visitation_percentage(trajectory, radius, plane_size=(-1, 1), grid_resolution=100):
    """
    Calculate the visitation percentage of a trajectory on a 2D plane, accounting for a visitation radius.
    
    Parameters:
    - trajectory: numpy array of shape (n, 2), where each row represents (x, y) coordinates.
    - radius: float, the radius around each point within which cells are considered visited.
    - plane_size: tuple, the min and max bounds for both x and y axes, e.g., (-1, 1).
    - grid_resolution: int, the resolution of the grid for discretizing the plane.
    
    Returns:
    - float: A value between 0 and 1 representing the percentage of the plane that was visited.
    """
    # Create an empty grid for the plane
    grid = np.zeros((grid_resolution, grid_resolution), dtype=bool)
    
    # Define grid and cell properties
    min_bound, max_bound = plane_size
    grid_size = max_bound - min_bound
    cell_size = grid_size / grid_resolution
    radius_cells = int(np.ceil(radius / cell_size))
    
    for (x, y) in trajectory:
        # Convert (x, y) to grid indices
        center_row = int((y - min_bound) / cell_size)
        center_col = int((x - min_bound) / cell_size)
        
        # Loop over a square around the point, limited by radius in cells
        for i in range(-radius_cells, radius_cells + 1):
            for j in range(-radius_cells, radius_cells + 1):
                # Calculate the distance from the center point
                if i**2 + j**2 <= (radius / cell_size)**2:
                    row, col = center_row + i, center_col + j
                    # Mark cells within bounds as visited
                    if 0 <= row < grid_resolution and 0 <= col < grid_resolution:
                        grid[row, col] = True
    
    # Calculate the percentage of visited cells
    visited_cells = np.sum(grid)
    total_cells = grid_resolution * grid_resolution
    visitation_percent = visited_cells / total_cells
    visitation_percent = visitation_percent * 100
    
    return visitation_percent

def run_sim(args, case, noise, room):
    model_path = args['model_path']
    camera_name = args['camera_name']
    height = args['height']
    T = args['T']

    sfm = SFM_EMMD(args)
    sfm.solve_trajectory()
    sfm.extract_directions()
    sfm.quat = np.zeros((len(sfm.direction), 4))
    sfm.quat[:, 0] = 1
    #sfm.plot_system()
    sfm.trajectory = np.array([sfm.trajectory[:, 0], sfm.trajectory[:, 2], (sfm.trajectory[:, 1]+height)]).T

    if not os.path.exists(f'../run/{str(room)}/{str(args["solver"])}/{case}/{noise}'):
        os.makedirs(f'../run/{str(room)}/{str(args["solver"])}/{case}/{noise}')

    if not os.path.exists(f'../run/{str(room)}/{str(args["solver"])}/{case}/{noise}/images'):
        os.makedirs(f'../run/{str(room)}/{str(args["solver"])}/{case}/{noise}/images')

    if not os.path.exists(f'../run/{str(room)}/{str(args["solver"])}/{case}/{noise}/depth_images'):
        os.makedirs(f'../run/{str(room)}/{str(args["solver"])}/{case}/{noise}/depth_images')

    sfm.mjc_viewer(model_path, camera_name)

    images_to_video(f'../run/{str(room)}/{str(args["solver"])}/{case}/{noise}/images', 'output_video.mp4', fps=(1/dt))

    # Calculate the differences between consecutive points
    diffs = np.diff(sfm.drone_path[:, [0, 1]], axis=0)
    # Calculate the Euclidean distances between consecutive points
    distances = np.linalg.norm(diffs, axis=1)
    # Sum up all distances
    total_distance = np.sum(distances)

    print(f'Visitation (%) / 1 m Trajectory Distance: {visitation_percentage(sfm.drone_path[:, [0,1]], radius=0.04, grid_resolution=100)/total_distance}')

    output_args = {'trajectory': sfm.drone_path,
                   'directions': sfm.drone_euler,
                   'visitation_per_distance': (visitation_percentage(sfm.drone_path[:, [0,1]], radius=0.04, grid_resolution=100)/total_distance),
                   'distance': total_distance}

    #images_to_video(f'../run/{str(args["solver"])}/{case}/{noise}/depth_images', 'depth_video.mp4', fps=(1/dt))
    #
    #images_to_video(f'../run/{str(args["solver"])}/{case}/{noise}/images', 'output_video.mp4', fps=(1/dt))
    np.savez(f'../run/{str(room)}/{str(args["solver"])}/{case}/{noise}/output.npz', **output_args)

    return sfm.drone_path

if __name__ == '__main__':
    ## INPUTS
    tf = 30
    num_points = 250
    dt = 0.02
    push = 0
    scaling = 0
    h = 0.005
    max_v = 0.1
    height = 0.8

    camera_name = 'track' # ORIENTATION: +y = up
    T = int(tf/dt)
    
    sigma = 0.2
    
    no_noise = {
        'glass_click': [[-0.56, 0.4, .5102, sigma],
                        [-0.63, 0.33, 0.474, sigma],
                        [-0.49, 0.33, 0.4612, sigma],
                        [-0.6, 0.3, 0.4524, sigma],
                        [0.34, -0.03, 0.4143, sigma]],
        'knock': [[0.64, -0.03, 0.1439, sigma],
                  [0.7, -0.12, 0.1059, sigma],
                  [0.62, -0.16, 0.1045, sigma],
                  [0.32, -0.09, 0.1043, sigma],
                  [0.72, -0.09, 0.1034, sigma]],
        'frying_pan': [[0.14, -0.21, 0.1315, sigma],
                       [0.09, -0.11, 0.1297, sigma],
                       [-0.07, -0.07, 0.1279, sigma],
                       [0.53, -0.16, 0.1242, sigma],
                       [0.25, -0.11, -.1204, sigma]],
        'soccer_ball': [[0.51, 0.01, 0.1832, sigma],
                        [0.5, 0.02, 0.1832, sigma],
                        [0.72, -0.09, 0.1739, sigma],
                        [0.72, -0.08, 0.1732, sigma],
                        [0.62, -0.16, 0.1669, sigma]],
        'mug': [[0.28, -0.34, 0.1834, sigma],
                [-0.34, -0.8, 0.1766, sigma],
                [0.17, -0.3, 0.1681, sigma],
                [-0.3, -0.79, 0.1663, sigma],
                [-0.27, -0.77, 0.1638, sigma]],
        'saxophone': [[-0.59, -0.21, 0.5486, sigma],
                      [-0.57, -0.23, 0.3649, sigma],
                      [-0.15, 0.59, 0.1877, sigma],
                      [0.26, 0.01, 0.18, sigma],
                      [0.32, 0.45, 0.1425, sigma]],
        'alarm': [[0.53, -0.6, 0.332, sigma],
                  [0.5, -0.57, 0.5065, sigma],
                  [0.44, -0.51, 0.2759, sigma],
                  [0.43, -0.51, 0.2711, sigma],
                  [0.41, -0.61, 0.2534, sigma]],
        'bang': [[0.64, 0, 0.2655, sigma],
                 [0.56, -0.13, 0.2622, sigma],
                 [0.67, -0.02, 0.2574, sigma],
                 [0.56, -0.1, 0.2574, sigma],
                 [0.43, -0.12, 0.2557, sigma]],
        'beep': [[0.62, 0.52, 0.2209, sigma],
                 [0.16, -0.65, 0.1695, sigma],
                 [0.44, -0.51, 0.1393, sigma],
                 [0.37, 0.46, 0.1247, sigma],
                 [0.47, 0.61, 0.1184, sigma]]}

    light_noise = {
        'beep': [[0.62, 0.52, 0.2107, sigma],
                  [-0.63, 0.25, 0.1479, sigma],
                  [0.47, 0.61, 0.1433, sigma],
                  [-0.49, 0.33, 0.1242, sigma],
                  [-0.6, 0.28, 0.1237, sigma]],
        'bang': [[-0.27, -0.77, 0.1051, sigma],
                      [0.34, 0.45, 0.1039, sigma]],
        'knock': [[0.09, -0.26, 0.1319, sigma],
                   [0.06, -0.22, 0.13, sigma],
                   [0.54, -0.4, 0.1169, sigma],
                   [0.28, -0.34, 0.1053, sigma],
                   [0.17, -0.3, 0.1018, sigma]],
        'lamp': [[0.58, -0.71, 0.1804, sigma],
                 [0.34, -0.03, 0.1781, sigma],
                 [0.27, -0.03, 0.1769, sigma],
                 [-0.27, -0.77, 0.1765, sigma],
                 [0.45, -0.61, 0.1764, sigma]],
        'mug': [[0.28, -0.04, 0.2243, sigma],
                [0.27, -0.03, 0.2184, sigma],
                [0.15, -0.08, 0.2151, sigma],
                [0.33, -0.03, 0.2039, sigma],
                [0.34, -0.03, 0.2006, sigma]],
        'roomba': [[0.47, 0.61, 0.101, sigma]], 
        'saxophone': [[-0.59, -0.21, 0.5633, sigma],
                      [-0.57, -0.23, 0.397, sigma],
                      [0.49, -0.57, 0.1810, sigma],
                      [-0.49, 0.33, 0.163, sigma],
                      [-0.56, 0.4, 0.1599, sigma]],
        'soccer_ball': [[0.62, -0.16, 0.1873, sigma],
                        [0.56, -0.13, 0.1842, sigma],
                        [0.72, -0.09, 0.1841, sigma],
                        [0.72, -0.08, 0.1837, sigma],
                        [0.7, -0.12, 0.1777, sigma]],
        'warning_alarm': [[0.62, 0.52, 0.1873, sigma],
                          [0.32, 0.45, 0.1745, sigma],
                          [0.49, -0.57, 0.1739, sigma],
                          [0.53, -0.6, 0.1577, sigma],
                          [0.47, 0.61, 0.1548, sigma]],
        'glass_click': [[0.26, 0.01, 0.1623, sigma],
                        [0.09, -0.13, 0.1407, sigma],
                        [0.58, 0.04, 0.1373, sigma],
                        [0.39, -0.13, 0.136, sigma],
                        [0.14, -0.21, 0.1351, sigma]]}
    
    strong_noise = {
        'beep': [[0.62, 0.52, 0.4878, sigma],
                        [0.16, -0.65, 0.3786, sigma],
                        [0.47, 0.61, 0.3478, sigma],
                        [0.28, -0.34, 0.3478, sigma],
                        [0.34, 0.45, 0.2977, sigma]],
        'frying_pan': [[0.28, -0.04, 0.197, sigma],
                       [0.27, -0.03, 0.1878, sigma],
                       [0.34, -0.03, 0.1827, sigma],
                       [0.25, -0.11, 0.1737, sigma],
                       [0.09, -0.11, 0.166, sigma]],
        'knock': [[0.09, -0.26, 0.1772, sigma],
                   [0.06, -0.22, 0.1421, sigma],
                   [0.54, -0.4, 0.1155, sigma],
                   [0.17, -0.3, 0.1097, sigma],
                   [-0.34, -0.8, 0.1096, sigma]],
        'mug': [[0.46, -0.64, 0.1148, sigma],
                [0.45, -0.61, 0.1121, sigma],
                [0.41, -0.61, 0.1036, sigma]],
        'roomba': [[-0.57, -0.23, 0.2435, sigma],
                   [-0.59, -0.21, 0.2381, sigma],
                   [0.43, -0.53, 0.2234, sigma],
                   [-0.16, 0.42, 0.2193, sigma],
                   [0.43, -0.57, 0.2048, sigma]],
        'saxophone': [[0.64, -0.03, 0.2018, sigma],
                      [0.32, -0.09, 0.2003, sigma],
                      [0.32, -0.13, 0.1575, sigma],
                      [0.55, 0.04, 0.1476, sigma],
                      [0.56, -0.1, 0.1411, sigma]],
        'warning_alarm': [[-0.15, 0.59, 0.2337, sigma],
                          [0.32, 0.45, 0.182, sigma],
                          [-0.63, 0.33, 0.1794, sigma],
                          [0.37, 0.46, 0.1719, sigma],
                          [-0.6, 0.28, 0.1708, sigma]],
        'glass_click': [[0.64, -0.03, 0.1937, sigma],
                        [0.32, -0.09, 0.1751, sigma],
                        [0.72, -0.09, 0.1736, sigma],
                        [0.7, -0.12, 0.1697, sigma],
                        [0.51, 0.01, 0.1657, sigma]]}

    noises = ['glass_click', 'knock', 'frying_pan', 'soccer_ball', 'mug', 'saxophone', 'alarm', 'bang', 'beep', 'roomba']
    solvers = ['emmd', 'tsp', 'info_max']
    rooms = ['drone_room_1', 'drone_room_2', 'drone_room_3']

    for n in range(len(rooms)):
        for number in range(3):
            if number == 0:
                arguments = no_noise
                case = 'no_noise'
            elif number == 1:
                arguments = light_noise
                case = 'light_noise'
            elif number == 2:
                arguments = strong_noise
                case = 'strong_noise'
            for solver in solvers:
                mesh_path = f'../meshes/{rooms[n]}/assets/room.obj'
                model_path = f'../meshes/{rooms[n]}/scene.xml'

                for num in range(len(noises)):
                    noise = noises[num]
                    data = arguments.get(noise, None)

                    args = {
                        'T' : T,
                        'model_path': model_path,
                        'camera_name': camera_name,
                        'mesh_path' : mesh_path,
                        'num_points' : num_points,
                        'dt' : dt,
                        'tf' : tf,
                        'push' : push,
                        'scaling' : scaling, 
                        'h' : h,
                        'height' : height,
                        'max_v' : max_v,
                        'solver' : solver,
                        'mesher' : 'opensfm',
                        'utility' : 'distribution',
                        'colormap' : 1,
                        'data': data,
                        'noise': noise,
                        'case': case,
                        'room': rooms[n]}
                    
                    if data is None: 
                        print('No data found for this case. Using uniform distribution instead.')
                        args["utility"] = 'uniform'

                    print(f'\nRunning [{rooms[n]}] with[{solver}] solver on [{str(args["utility"])}] information for [{case}] case with [{noise}] noise...\n')
                    run_sim(args, case, noise, rooms[n])
            