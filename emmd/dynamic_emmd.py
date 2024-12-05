import sys 
sys.path.append('../')

import numpy as np
import jax.numpy as jnp
from jax import vmap
import trimesh as tm
from sklearn.neighbors import KDTree
import plotly.graph_objects as go
from emmd.solver_jaxopt import AugmentedLagrangeSolver

import pickle as pkl

class Dynamic_EMMD():
    def __init__(self, input_args):
        
        self.mesh = tm.load_mesh(input_args['mesh_path'])
        self.num_points = input_args.get('num_points', 3000)
        print(f'Number of points: {self.num_points}')
        self.dt = input_args.get('dt', 0.02)
        self.tf = input_args.get('tf', 30)
        self.info_dist = input_args.get('info_dist', lambda x : 1)
        self.scaling = input_args.get('scaling', 1)

        self.T = input_args.get('T', int(self.tf/self.dt))
        self.h = input_args.get('h', 0.001)

        if self.scaling == 1:
            self.scale_mesh()
        else:
            self.scaling_factor = 1

        self.points, self.faces = tm.sample.sample_surface(self.mesh, self.num_points)
        self.x_0 = np.array([0, 0, 0])#self.mesh.bounds[0]# - [0, 0, .90]
        self.x_f = np.array([0, 0, 0.25])#self.mesh.bounds[1]
        
        self.push_val = input_args.get('push', 0.03*self.scaling_factor)
        self.max_v = input_args.get('max_v', 1*self.scaling_factor)
        self.max_v = self.max_v * self.dt

        self.P_XI = input_args['P_XI'](self.points)
        self.P_XI = self.P_XI / jnp.sum(self.P_XI)
        self.X = jnp.linspace(self.x_0, self.x_f, num=self.T)

        self.args = {'h' : self.h, 'points' : self.points+self.push_val*self.mesh.face_normals[self.faces], 'P_XI' : self.P_XI, 'T' : self.T}
        self.params = {'X' : self.X,
                       'points': self.points}

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
    
    def RBF_kernel(self, x, xp, h=0.01):
        return jnp.exp(
            -jnp.sum((x-xp)**2)/h
        )

    def create_kernel_matrix(self, kernel_func, args=0):
        return vmap(vmap(kernel_func, in_axes=(0, None, None)), in_axes=(None, 0, None))
    
    def solve(self):
        KernelMatrix = self.create_kernel_matrix(self.RBF_kernel, args=self.args)
        
        # Define Loss and Constraints --------------------------------------------------------------------
        emmd_loss = lambda params, args: (
            np.sum(KernelMatrix(params['X'], params['X'], args['h'])) / (args['T']**2)
            - 2 * np.sum(args['P_XI'] @ KernelMatrix(params['X'], args['points'], args['h'])) / args['T']
            + 2 * jnp.mean(jnp.square(params['X'][1:] - params['X'][:-1]))
        )

        eq_constr = lambda params, args: jnp.array(0.)

        ineq_constr = lambda params, args: jnp.square(params['X'][1:] - params['X'][:-1]) - self.max_v**2

        # ------------------------------------------------------------------------------------------------

        solver = AugmentedLagrangeSolver(self.params, emmd_loss, eq_constr, ineq_constr, args = self.args)

        solver.solve(max_iter=1000,eps=1e-5)

        self.trajectory = solver.solution['X']

        return self.trajectory

    def plot_system(self, title="3D Mesh and Trajectory Visualization"):
        mesh = self.mesh
        trajectory = self.trajectory
        
        # Extract vertices and faces from the trimesh mesh
        vertices = mesh.vertices
        faces = mesh.faces

        # Build a KD-tree for fast nearest-neighbor search
        kdtree = KDTree(vertices)

        # Find the nearest mesh points for each trajectory point
        trajectory_indices = kdtree.query(self.trajectory)[1]

        # Get normals corresponding to the nearest mesh points
        mesh_normals = np.asarray(self.mesh.vertex_normals)
        
        # Ensure normals are correctly aligned with trajectory
        traj_direction = -mesh_normals[trajectory_indices]  # Inverse the normals for arrow direction
        traj_direction = traj_direction / np.linalg.norm(traj_direction)

        # Create mesh trace with improved aesthetics
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='dimgrey',  # Soft professional color
            opacity=0.85,       # Increased opacity for a cleaner look
            flatshading=False,  # Use smooth shading for better 3D look
            lighting=dict(
                ambient=0.5,   # Soft ambient lighting
                diffuse=0.9,   # Make mesh lighting more realistic
                fresnel=0.1,   # Slight fresnel effect for a more polished look
                roughness=0.3,
                specular=0.5
            ),
            lightposition=dict(x=100, y=200, z=300),  # Customize light source position
            name="Mesh",
            showscale=False    # No color scale to keep it clean
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
                u=[vec[0][0]], v=[vec[0][1]], w=[vec[0][2]],
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
                    title="X-Axis",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor='white',
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False,  # No spikes for a cleaner plot
                    range=[-1, 1]  # Set fixed axis bounds [0, 1]
                ),
                yaxis=dict(
                    title="Y-Axis",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor='white',
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False,
                    range=[-1, 1]  # Set fixed axis bounds [0, 1]
                ),
                zaxis=dict(
                    title="Z-Axis",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor='white',
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False,
                    range=[-1, 1]  # Set fixed axis bounds [0, 1]
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
        
    def save_data(self):
        self.out_args = {
        'mesh': self.mesh,
        'AXYZ': self.trajectory,
        'FOV_H': 50,
        'FOV_R': 3,
        'scaling_factor': self.scaling_factor
        }

        with open('pickle_files/args_emmd.pkl', 'wb') as f:
            pkl.dump(self.out_args, f)


if __name__ == '__main__':
    mdl_path = '/home/christian/Downloads/obj_and_ply/bridge.obj'#'/home/christian/Downloads/obj_and_ply/turbine.obj'
    args = {'mesh_path': mdl_path, 
            'push': 0.1, 
            'T': 120}

    emmd = Dynamic_EMMD(args)
    emmd.solve()
    emmd.plot_system()
    
    emmd.save_data()