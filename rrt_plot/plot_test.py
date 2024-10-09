import argparse
import pandas as pd
import matplotlib.pyplot as plt
import util


parser = argparse.ArgumentParser(
                    prog='plot_test',
                    description='Plot logging from a test file')
parser.add_argument(
    '-n', '--name',
    required=True,
    help='The name of the test specified in the config file',
)
args = parser.parse_args()
config = util.get_config()
log_dir = util.copy_and_run(args.name, config)

def q1b():
    df = pd.read_csv(log_dir.joinpath('state.csv'))

    for _, group_df in df.groupby('episode'):
        plt.plot(group_df['x'], group_df['y'])
        
    plt.xlim([df['x'].min()-1, df['x'].max()+1])
    plt.ylim([df['y'].min()-1, df['y'].max()+1])

    plt.title("100 rollouts using random policy")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()
    
    
def q1c():
    state_df = pd.read_csv(log_dir.joinpath('state.csv'))
    cmd_df = pd.read_csv(log_dir.joinpath('cmd.csv'))
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].plot(state_df['x'], state_df['y'])
    axs[0].set_xlim([state_df['x'].min()-1, state_df['x'].max()+1])
    axs[0].set_ylim([state_df['y'].min()-1, state_df['y'].max()+1])
    axs[0].set_title('trajectory')
    
    axs[1].plot(cmd_df.index, cmd_df['a'], label='a')
    axs[1].plot(cmd_df.index, cmd_df['psi'], label='psi')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title('sampled sinusoidal command')
    
    plt.show()
    
def q2a():
    df = pd.read_csv(log_dir.joinpath('tree.csv'))
    
    for _, branch_df in df.groupby('branch'):
        plt.plot(branch_df['x'], branch_df['y'])
        branch_df = branch_df.reset_index(drop=True)
        assert branch_df.at[len(branch_df)-1, 'x'] == 0.0
        assert branch_df.at[len(branch_df)-1, 'y'] == 0.0
        
    plt.title("rrt tree with normalized weights after 1000 iterations")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.show()
    
def q2a_heatmap(iter=100):
    dfs = []
    for _ in range(iter):
        log_dir = util.copy_and_run(args.name, config)
        dfs.append(pd.read_csv(log_dir.joinpath('tree.csv')))

    nodes_x = []
    nodes_y = []
    for df in dfs:
        nodes_x.extend(list(df['x']))
        nodes_y.extend(list(df['y']))
        
    plt.scatter(nodes_x, nodes_y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    
    plt.title(f"all nodes from {iter} rrt runs under w3 weights")
    plt.show()
    
def q2c():
    df = pd.read_csv(log_dir.joinpath('tree.csv'))
    
    for _, branch_df in df.groupby('branch'):
        plt.plot(branch_df['x'], branch_df['y'])
        branch_df = branch_df.reset_index(drop=True)
        assert branch_df.at[len(branch_df)-1, 'x'] == 0.0
        assert branch_df.at[len(branch_df)-1, 'y'] == 0.0
        
    plt.scatter([3], [4], marker='x', c='r', s=36, label="goal")
        
    plt.title("rrt tree with goal biasing 0.5, k=3 after 1000 iterations")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.legend()
    plt.show()

def q3a():
    state_df = pd.read_csv(log_dir.joinpath('state.csv'))
    obstacle_df = pd.read_csv(log_dir.joinpath('obstacles.csv'))
    collision_df = pd.read_csv(log_dir.joinpath('collision_states.csv'))

    for i in range(state_df.at[len(state_df)-1, 'episode']):
        # Plot obstacles
        verts_l = util.extract_plottable_rectangles(obstacle_df, i)
        for vx, vy in verts_l:
            plt.plot(vx, vy, 'k-')

        # Plot collision state, if exists
        verts_l = util.extract_plottable_rectangles(collision_df, i)
        for vx, vy in verts_l:
            plt.plot(vx, vy, 'r-')

        ep_states = state_df[state_df['episode'] == i]
        if len(verts_l) > 0:
            plt.plot(ep_states['x'], ep_states['y'], 'r-')
        else:
            plt.plot(ep_states['x'], ep_states['y'], 'k-')

    plt.title("sinusoidal from origin, with an obstacle")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.legend()
    plt.show()

def q3b():
    df = pd.read_csv(log_dir.joinpath('tree.csv'))
    obstacle_df = pd.read_csv(log_dir.joinpath('obstacles.csv'))
    collision_df = pd.read_csv(log_dir.joinpath('collision_states.csv'))

    for _, branch_df in df.groupby('branch'):
        plt.plot(branch_df['x'], branch_df['y'], 'b-')
        branch_df = branch_df.reset_index(drop=True)
        assert branch_df.at[len(branch_df)-1, 'x'] == 0.0
        assert branch_df.at[len(branch_df)-1, 'y'] == 0.0

    verts_l = util.extract_plottable_rectangles(obstacle_df, 0)
    for vx, vy in verts_l:
        plt.plot(vx, vy, 'k-')

    verts_l = util.extract_plottable_rectangles(collision_df, 0)
    for vx, vy in verts_l:
        plt.plot(vx, vy, 'r-')

    plt.title("rrt with obstacles")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.legend()
    plt.show()
    
def plot_car():
    car_xs = [-0.135, 0.615, 0.615, -0.135]
    car_ys = [-0.3, -0.3, 0.3, 0.3]

    obs_xs = [0.95, 1.05, 1.05, 0.95]
    obs_ys = [-0.5, -0.5, 0.5, 0.5]

    fig, ax = plt.subplots()

    # Plot the car
    ax.plot(car_xs + [car_xs[0]], car_ys + [car_ys[0]], 'b-', label='Car')  # Close the rectangle by adding the first point at the end
    ax.fill(car_xs + [car_xs[0]], car_ys + [car_ys[0]], alpha=0.2, color='blue')

    # Plot the obstacle
    ax.plot(obs_xs + [obs_xs[0]], obs_ys + [obs_ys[0]], 'r-', label='Obstacle')
    ax.fill(obs_xs + [obs_xs[0]], obs_ys + [obs_ys[0]], alpha=0.2, color='red')

    # Add labels for car dimensions
    car_width = abs(car_xs[1] - car_xs[0])
    car_height = abs(car_ys[2] - car_ys[1])
    ax.text((car_xs[0] + car_xs[1]) / 2, car_ys[0] - 0.05, f'{car_width:.3f}m', ha='center')
    ax.text(car_xs[1] + 0.05, (car_ys[1] + car_ys[2]) / 2, f'{car_height:.3f}m', va='center')

    # Add labels for obstacle dimensions
    obs_width = abs(obs_xs[1] - obs_xs[0])
    obs_height = abs(obs_ys[2] - obs_ys[1])
    ax.text((obs_xs[0] + obs_xs[1]) / 2, obs_ys[0] - 0.05, f'{obs_width:.3f}m', ha='center')
    ax.text(obs_xs[1] + 0.05, (obs_ys[1] + obs_ys[2]) / 2, f'{obs_height:.3f}m', va='center')

    # Add legend
    ax.legend()

    # Set equal aspect ratio to avoid distortion
    ax.set_aspect('equal', adjustable='box')

    # Set grid and labels
    ax.grid(True)
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')

    plt.title("starting position of car, obstacle")
    plt.show()

if __name__ == '__main__':
    q3b()
