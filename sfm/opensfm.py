import numpy as np
import pyvista as pv
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

def read_ply(file_name):
    with open(file_name, 'r') as f:
        header = []
        num_vertices = 0
        properties = []
        
        while True:
            line = f.readline().strip()
            if line == 'end_header':
                header.append(line)
                break
            header.append(line)
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                properties.append(line.split()[1:])

    data = np.loadtxt(file_name, skiprows=len(header), max_rows=num_vertices)
    return header, data, properties

def write_ply(file_name, data):
    xyz_data = data[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_data)
    o3d.io.write_point_cloud(file_name, point_cloud)

def write_obj(file_name, points, faces):
    with open(file_name, 'w') as f:
        for vertex in points:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def filter_axes(data, properties):
    color_indices = [i for i, prop in enumerate(properties) if prop[1] in ['red', 'green', 'blue']]
    print(f"Identified color indices: {color_indices}")

    if len(color_indices) != 3:
        raise ValueError(f"Expected 3 color channels, found {len(color_indices)}. Cannot proceed with filtering.")

    colors = data[:, color_indices].astype(int)
    mask = ~((colors == [255, 0, 0]).all(axis=1) | 
             (colors == [0, 255, 0]).all(axis=1) | 
             (colors == [0, 0, 255]).all(axis=1))
    filtered_data = data[mask]
    return filtered_data

def filter_sparse_points(data, distance_threshold, min_neighbors=3):
    xyz = data[:, :3]
    tree = cKDTree(xyz)
    neighbors = tree.query_ball_point(xyz, distance_threshold)
    mask = np.array([len(n) > min_neighbors for n in neighbors])
    filtered_data = data[mask]
    return filtered_data

def generate_faces_from_points(points, k_neighbors=35, angle_threshold=np.pi/6):
    # Step 1: Compute nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Step 2: Estimate local point density
    local_density = 1 / distances[:, -1]

    # Step 3: Compute normal vectors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    normals = np.asarray(pcd.normals)

    # Step 4: Generate faces
    faces = []
    for i in range(len(points)):
        for j in indices[i, 1:]:
            for k in indices[j, 1:]:
                if k in indices[i, 1:]:
                    # Check if the triangle is not too elongated
                    edge1 = points[j] - points[i]
                    edge2 = points[k] - points[i]
                    edge3 = points[k] - points[j]
                    if (np.linalg.norm(edge1) < 2 * local_density[i] and
                        np.linalg.norm(edge2) < 2 * local_density[i] and
                        np.linalg.norm(edge3) < 2 * local_density[j]):
                        
                        # Check if the normals are consistent
                        normal1 = normals[i]
                        normal2 = normals[j]
                        normal3 = normals[k]
                        if (np.dot(normal1, normal2) > np.cos(angle_threshold) and
                            np.dot(normal1, normal3) > np.cos(angle_threshold) and
                            np.dot(normal2, normal3) > np.cos(angle_threshold)):
                            
                            faces.append([i, j, k])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Apply Laplacian smoothing with more iterations and higher strength
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=50, lambda_filter=0.5)
    
    # Apply Taubin smoothing for better volume preservation
    mesh = mesh.filter_smooth_taubin(number_of_iterations=30, lambda_filter=0.5, mu=-0.53)

    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    return np.array(points),np.array(faces)

def generate_watertight_mesh(points, alpha=0.03, depth=8, scale=1, linear_fit=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    alpha_mesh.compute_vertex_normals()
    alpha_mesh = alpha_mesh.simplify_quadric_decimation(100000)

    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=linear_fit)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

    combined_mesh = alpha_mesh + poisson_mesh
    combined_mesh.remove_degenerate_triangles()
    combined_mesh.remove_duplicated_triangles()
    combined_mesh.remove_duplicated_vertices()
    combined_mesh.remove_non_manifold_edges()

    final_mesh = combined_mesh.simplify_quadric_decimation(100000)

    # Apply Laplacian smoothing with more iterations and higher strength
    final_mesh = final_mesh.filter_smooth_laplacian(number_of_iterations=50, lambda_filter=0.5)
    
    # Apply Taubin smoothing for better volume preservation
    final_mesh = final_mesh.filter_smooth_taubin(number_of_iterations=30, lambda_filter=0.5, mu=-0.53)

    return np.asarray(final_mesh.vertices), np.asarray(final_mesh.triangles)

def is_exploding(mesh, parallel_threshold=0.99, threshold_percent=50):
    # Compute normals for the cells (faces)
    mesh = mesh.compute_normals(cell_normals=True)
    face_normals = mesh.cell_normals

    # Normalize the normals
    normalized_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)

    # Count how many normals are close to parallel (dot product close to 1)
    num_normals = len(normalized_normals)
    parallel_count = 0
    
    for i in range(num_normals):
        # Dot product with other normals
        dot_products = np.dot(normalized_normals, normalized_normals[i])
        # Count normals that are "close" to parallel (dot product > parallel_threshold)
        #print(np.sum(dot_products > parallel_threshold))
        if np.sum(dot_products > parallel_threshold) > ((threshold_percent/100)*num_normals):
            parallel_count += 1

    print(f'# of parallels: {parallel_count}\n# of normals: {num_normals}')

    # Compute the percentage of parallel normals
    percent_parallel = parallel_count / num_normals * 100
    
    # Check if the percentage exceeds the user-defined threshold
    if percent_parallel > threshold_percent:
        return True
    return False

def filter_cloud(input_file, output_file, distance_threshold, min_neighbors=3, alpha=0.15):
    header, data, properties = read_ply(input_file)
    print("Filtering out axes points (red, green, blue)...")
    data_no_axes = filter_axes(data, properties)
    print("Filtering out sparse points...")
    data_filtered = filter_sparse_points(data_no_axes, distance_threshold, min_neighbors)
    print(f"Original data shape: {data.shape}")
    print(f"Filtered data shape after removing axes and sparse points: {data_filtered.shape}")
    return data_filtered

def plot_mesh(points, faces=0):
    if faces.all() == 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        pcd = o3d.geometry.TriangleMesh()
        pcd.vertices = o3d.utility.Vector3dVector(points)
        pcd.triangles = o3d.utility.Vector3iVector(faces)

    o3d.visualization.draw_geometries([pcd],
            zoom=0.3412,
            front=[0.4257, -0.2125, -0.8795],
            lookat=[2.6172, 2.0475, 1.532],
            up=[-0.0694, -0.9768, 0.2024])
    
def main(input_file = "output/reconstruction.ply", output_file = "output/output_filtered.obj", distance_threshold = 0.2, min_neighbors = 3, parallel_threshold = 0.85, threshold_percent = 30):   

    data_filtered = filter_cloud(input_file, output_file, distance_threshold, min_neighbors)[:, :3]

    if output_file.endswith(".ply"):
        write_ply(output_file, data_filtered)

        plot_mesh(data_filtered[:, :3])

    elif output_file.endswith(".obj"):
        print("Generating watertight mesh...")
        points, faces = generate_watertight_mesh(data_filtered)
        
        # Create PyVista mesh for explosion check
        pv_mesh = pv.PolyData(points)
        pv_mesh.faces = np.hstack([[3] + list(face) for face in faces])  # PyVista expects faces in this format

        if is_exploding(pv_mesh, parallel_threshold, threshold_percent):
            print("Mesh exploded into a plane, generating using points...")
            points, faces = generate_faces_from_points(data_filtered)
        else:
            print("Mesh is valid, keeping watertight mesh.")

        write_obj(output_file, points, faces)

        plot_mesh(points, faces)

    else:
        raise ValueError(f"Unsupported output format for file: {output_file}")

    print(f"Processed file saved as {output_file}")

if __name__ == "__main__":
    main()