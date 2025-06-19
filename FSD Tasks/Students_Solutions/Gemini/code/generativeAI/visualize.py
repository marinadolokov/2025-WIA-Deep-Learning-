
from manim import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


def extract_accelerations(row):
    return row["sensor_data"][:, 0:3]

def extract_rotational_velocities(row):
    return row["sensor_data"][:, 3:6]

def kmh_to_mps(kmh):
    return kmh / 3.6

def positional_verlet_integrate(a_n, dt, x_0=np.array([0, 0, 0]), v_start=np.array([0, 0, 0])):
    positions = [x_0, x_0 + v_start * dt]

    for acceleration in a_n:
        positions.append(acceleration * dt **2 - positions[-2] + 2 * positions[-1])

    return np.array(positions)

# Not used but equivalent logic in the visualize function.
def rotational_euler_integrate(v_n, dt, r_0=np.array([1, 0, 0])):
    rotations = [r_0]

    for velocity in v_n:
        rotations.append(rotations[-1] + velocity * dt)

    return rotations

def numeric_derivative(p_n, dt):
    v_n = []
    
    for prev, curr in zip(p_n, p_n[1:]):
        v_n.append((prev - curr) / dt)

    return np.array(v_n)


def visualize(data_point, dt=1/60, scale=2, save_path="car_trajectory.mp4"):
    """
    Visualize a single data point from the dataset.
    
    Parameters:
    - data_point: A row from the dataset containing accelerations and velocities.
    - dt: Time step for integration.
    - scale: Scale factor for visualization.
    - save_path: Path to save the video.
    """
    accelerations = extract_accelerations(data_point)
    rotational_velocities = extract_rotational_velocities(data_point)

    positions = positional_verlet_integrate(
        accelerations, dt,
        x_0=np.array([0, 0, 0]),
        v_start=np.array([kmh_to_mps(data_point["velocity"]), 0, 0])
    )[1:]

    path_points = [scale * np.array([x[0], x[2], 0]) for x in positions]

    class CarTrajectory(Scene):
        def construct(self):
            x0 = path_points[0][0]
            y0 = path_points[0][1]

            path = VMobject(color=BLUE)
            path.set_points_as_corners(path_points)
            self.add(path)

            grid = NumberPlane(
                x_range=[x0 - 20, x0 + 20, 0.5],
                y_range=[y0 - 20, y0 + 20, 0.5],
                background_line_style={
                    "stroke_color": LIGHT_GRAY,
                    "stroke_width": 1,
                    "stroke_opacity": 0.4
                },
                axis_config={
                "stroke_opacity": 0  # 
                }
                
            )
            grid.shift(path_points[0]) 
            self.add(grid)

            car = SVGMobject("car.svg")
            car.scale(0.3)
            car.move_to(path_points[0])
            self.add(car)

            def update_car(mob, alpha):
                i = min(int(alpha * (len(path_points) - 1)), len(path_points) - 2)
                pos = path_points[i]
                mob.move_to(pos)

                yaw = rotational_velocities[i][1] * dt 

                mob.rotate(math.radians(yaw), about_point=pos)
                            
                return mob

            # Animate in real time: duration = len * dt
            total_duration = len(path_points) * dt

            self.play(UpdateFromAlphaFunc(car, update_car), run_time=total_duration, rate_func=linear)
            self.wait()

            # Save the video
    with tempconfig({
        "quality": "low_quality",
        "preview": False,
        "output_file": save_path,
        "output_dir": "./media",
        "flush_cache" : True,
    }):
        scene = CarTrajectory()
        scene.render()

def plot_data(data: np.ndarray) -> None:   
    labels = ['ax', 'ay', 'az', 'rx', 'ry', 'rz']
    acc_min, acc_max = data[:, :3].min(), data[:, :3].max()
    rot_min, rot_max = data[:, 3:].min(), data[:, 3:].max()

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    for i, ax in enumerate(axs.flat):
        ax.plot(data[:, i], color='blue')
        ax.set_title(labels[i])
        if i < 3:
            ax.set_ylabel('m/s²')
            ax.set_ylim(acc_min, acc_max) 
        else:
            ax.set_ylabel('°/s')
            ax.set_ylim(rot_min, rot_max) 

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if i >= 3:
            ax.set_xlabel('Time Step')

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()