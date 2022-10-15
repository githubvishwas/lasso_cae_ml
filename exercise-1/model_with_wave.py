from typing import List
import numpy as np
import plotly.graph_objects as go
from lasso.dyna import D3plot, ArrayType
import math
import os
import random

def create_tria_mesh(x_max: int, y_max: int) -> np.ndarray:
    '''
    creates a tria mesh for a 2D x*y rectangle 
    each entry represents one tria for a D3plot shell mesh

    Parameters
    ---------
    x_max: int
        how many x coords

    y_max: int
        how many y coords
    
    Returns
    -------
    tria_shell_mesh: np.ndarray
        numpy array of shape (2*(x_max-1)*(y_max-1), 4). Each entry looks like (p1, p2, p3, p3)
    '''

    # the mesh will be rectangular
    # each rect is split into a tria A and tria B
    trias_A = np.full(((x_max-1)*(y_max-1), 4), np.array([0, 1, x_max, x_max]))
    trias_B = np.full(((x_max-1)*(y_max-1), 4), np.array([1, x_max+1, x_max, x_max]))

    # currently all meshes are repetitions of same 2 trias
    mod = np.full((4, x_max-1), np.arange(x_max-1))

    # extend mesh over all trias
    for i in range(y_max-1):
        trias_A[(x_max-1)*i:(x_max-1)+(x_max-1)*i] += (mod + i*x_max).T
        trias_B[(x_max-1)*i:(x_max-1)+(x_max-1)*i] += (mod + i*x_max).T

    # stack both meshes
    tria_shell_mesh = np.stack([trias_A, trias_B]).reshape((2*(x_max-1)*(y_max-1), 4))

    return tria_shell_mesh

def create_plate(x_max: int, y_max: int) -> np.ndarray:
    '''
    creates a plate in x-y Plane containing x_max*y_max points
    returns array used for node_coordinates of shape (x_max*y_max, 3)

    Parameters
    ----------
    x_max: int
        how many x coords
    y_max: int
        how many y coords
    
    Returns
    -------
    node_coordinates: np.ndarray
        array of shape (x_max*y_max, 3)
    '''

    node_coordinates = np.zeros((x_max*y_max, 3))

    # fill in x_coords
    node_coordinates[:, 0] = np.stack([
        np.arange(x_max) for _ in range(y_max)
    ]).reshape(x_max*y_max)

    # fill in y_coords
    node_coordinates[:, 1] = np.stack([
        np.full((x_max), i) for i in range(y_max)
    ]).reshape(x_max*y_max)

    return node_coordinates

def create_wave(x_max: int, y_max: int, wave_pos: int, wave_thickness: float,
                wave_height: float, timesteps: int = 5) -> np.ndarray:
    '''
    creates node_displacement with x_max*y_max points, maximum height of z_max and
    provided timesteps of shape (timesteps, x_max*y_max, 3)

    Patameters
    ----------
    x_max: int
        how many x coords
    y_max: int
        how many y coords
    wav_pos: int
        location on x axis of wave top
    wave_thickness: float
        parameter influencing the width / thickness of wave
        may not turn negative!
    wave_height: float
        parameter influencing the height of the wave.
        Gets funky with values greater 4.0
    timesteps: int, default: 5
        number of timesteps

    Returns
    -------
    node_displacement: np.ndarray
        Array of shape (timesteps, x_max*y_max, 3)

    '''

    node_coordinates = create_plate(x_max, y_max)
    node_displacement = np.stack([node_coordinates for _ in range(timesteps)])

    wave_thickness = wave_thickness + (0.5 * timesteps)

    
    x_displ = np.zeros((x_max))
    for t in range(1, timesteps):
        z_displ = np.array([
            wave_height*math.exp(-1/(10*(wave_thickness - 0.5*t))*(x - wave_pos)**2) for x in range(x_max)
        ])
        node_displacement[t, :, 2] = np.stack([
            0.5*t*z_displ for _ in range(y_max)
        ]).reshape(x_max*y_max)
        for x in range(1, x_max):
            try:
                x_displ[x:] -= 1 - math.sqrt(1-(z_displ[x] - z_displ[x-1])**2)
            except ValueError:
                x_displ[x:] -= 1 # + math.sqrt(abs(1-(z_displ[x] - z_displ[x-1])**2))
        for y in range(y_max):
            node_displacement[t, y*x_max:(y+1)*x_max, 0] += 2 * x_displ


    return node_displacement


def plot_model_ploty(node_displacement: np.ndarray, shell_mesh: np.ndarray):
    '''
    creates a ploty plot of a provided model
    '''

    fig = go.Figure()
    
    for t in range(node_displacement.shape[0]):
        fig.add_trace(go.Mesh3d(
            x=node_displacement[t, :, 0],
            y=node_displacement[t, :, 1], 
            z=node_displacement[t, :, 2] + t, 
            i=shell_mesh[:, 0], 
            j=shell_mesh[:, 1], 
            k=shell_mesh[:, 2],
            text=np.arange(node_displacement.shape[1]),
            # alphahull=5,
            # opacity=0.4,
            # color="cyan"
        ))

    fig.update_layout(scene_aspectmode="data")
    fig.show()

def img_model_plotly(node_displacement: np.ndarray, shell_mesh: np.ndarray, save_path: str, img_nr: int):
    '''
    creates a plotly img of provided model last timestep
    '''

    fig_name = "{0}.png".format(img_nr)

    fig = go.Figure(go.Mesh3d(
        x=node_displacement[-1, :, 0],
        y=node_displacement[-1, :, 1],
        z=node_displacement[-1, :, 2],
        i=shell_mesh[:, 0],
        j=shell_mesh[:, 1],
        k=shell_mesh[:, 2]
    ))

    fig.add_trace(go.Scatter3d(
        x=[node_displacement[-1, :, 0].max()/2],
        y=[node_displacement[-1, :, 1].max()/2],
        z=[node_displacement[-1, :, 0].max()/4],
        opacity=0
    ))

    camera = dict(
        eye=dict(x=0, y=1, z=2.5)
    )

    fig.update_layout(
        scene=dict(
            xaxis_visible=False,        
            yaxis_visible=False,
            zaxis_visible=False),     
        scene_aspectmode="data",
        scene_camera=camera
    )
    fig.write_image(os.path.join(save_path, fig_name))

def create_D3plot(displacement: np.ndarray, shell_mesh: np.ndarray, save_path: str, sample_nr):
    ''' writes D3plot into target file '''
    
    plot = D3plot()
    plot.arrays[ArrayType.node_coordinates] = displacement[0]
    plot.arrays[ArrayType.node_displacement] = displacement
    plot.arrays[ArrayType.element_shell_node_indexes] = shell_mesh
    plot.arrays[ArrayType.element_shell_part_indexes] = np.full(
        (shell_mesh.shape[0]), 0)

    os.makedirs(save_path, exist_ok=True)
    plot.write_d3plot(os.path.join(save_path, "run{0}.d3plot".format(sample_nr)))

def latin_hypercube_sample(args: List[dict], samples: int, seed: str ="seed") -> List[list]:
    '''
    creates a latin hypercube samples

    Parameters
    ----------

    args: list
        each arg entry should be containing a dict with following keywords:
        start: float
            value of first sample
        stop: float
            value of last sample
    samples: int
        how many samples from start to stop
    
    seed: str, default: "seed"
        string to initialize random number generator


    Returns
    -------
    latin_samples: list of lists
        Returns a latin hypercube of samples, each sample containing unique sample values for all
        provided sample ranges in args
    '''

    random.seed(seed)

    samples = [
        [
            entry["start"] + i*((entry["stop"] - entry["start"]) / (samples-1)) for i in range(samples)
        ]
        for entry in args
    ]

    for lists in samples:
        random.shuffle(lists)

    return samples
        

def main():
    plot_dir = "plots"
    img_dir = "images"
    nr_samples = 100
    
    print("Please enter path to save directory:")
    path = input()
    if not os.path.isdir(path):
        print("Create new Folder at {0}? y/n".format(os.path.abspath(path)))
        reply = input()
        if(reply.lower() == "y"):
            os.makedirs(os.path.join(path, plot_dir))
            os.makedirs(os.path.join(path, img_dir))
        else:
            print("Aborted Script")
            return
    
    n_nodes_x = 100
    n_nodes_y = 25

    sample_index = [*range(nr_samples)]

    shell_mesh = create_tria_mesh(n_nodes_x, n_nodes_y)

    sample_hypercube = latin_hypercube_sample([
        dict(start=20, stop=80), dict(start=2, stop=10), dict(start=1.5, stop=3)],
        samples=nr_samples)

    for i in range(nr_samples):
        node_displacement = create_wave(n_nodes_x, n_nodes_y, sample_hypercube[0][i], sample_hypercube[1][i],
                                            sample_hypercube[2][i])
        create_D3plot(node_displacement, shell_mesh, os.path.join(path, plot_dir), sample_index[i])
        img_model_plotly(node_displacement, shell_mesh, os.path.join(path, img_dir), sample_index[i])
        
if __name__ == "__main__":
    main()

# plot_model_ploty(create_wave(x_max=50, y_max=20, wave_pos=10, 
#                             wave_thickness=3, wave_height=5),
#                 create_tria_mesh(50, 20))
# plot_model_ploty(create_wave(x_max=50, y_max=20, wave_pos=30, 
#                             wave_thickness=3, wave_height=2),
#                 create_tria_mesh(50, 20))
# plot_model_ploty(create_wave(x_max=50, y_max=20, wave_pos=20, 
#                             wave_thickness=3, wave_height=-2),
#                 create_tria_mesh(50, 20))

