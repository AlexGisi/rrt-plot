import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import util


parser = argparse.ArgumentParser(
    prog="plot_test", description="Plot logging from a test file"
)
parser.add_argument(
    "-n",
    "--name",
    required=True,
    help="The name of the test specified in the config file",
)
args = parser.parse_args()
config = util.get_config()
log_dir = util.copy_and_run(args.name, config)


def q2a():
    df = pd.read_csv(log_dir.joinpath("tree.csv"))

    for _, branch_df in df.groupby("branch"):
        plt.plot(branch_df["x"], branch_df["y"])
        branch_df = branch_df.reset_index(drop=True)
        assert branch_df.at[len(branch_df) - 1, "x"] == 0.0
        assert branch_df.at[len(branch_df) - 1, "y"] == 0.0

    plt.title("rrt tree with normalized weights after 1000 iterations")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.show()


def multiple_tree_heatmap(iter=100):
    dfs = []
    for _ in range(iter):
        log_dir = util.copy_and_run(args.name, config)
        dfs.append(pd.read_csv(log_dir.joinpath("tree.csv")))

    nodes_x = []
    nodes_y = []
    for df in dfs:
        nodes_x.extend(list(df["x"]))
        nodes_y.extend(list(df["y"]))

    plt.scatter(nodes_x, nodes_y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])

    plt.title(f"all nodes from {iter} rrt runs under w3 weights")
    plt.show()


def q2c():
    df = pd.read_csv(log_dir.joinpath("tree.csv"))

    for _, branch_df in df.groupby("branch"):
        plt.plot(branch_df["x"], branch_df["y"])
        branch_df = branch_df.reset_index(drop=True)
        assert branch_df.at[len(branch_df) - 1, "x"] == 0.0
        assert branch_df.at[len(branch_df) - 1, "y"] == 0.0

    plt.scatter([3], [4], marker="x", c="r", s=36, label="goal")

    plt.title("rrt tree with goal biasing 0.5, k=3 after 1000 iterations")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.legend()
    plt.show()


def dubins_paths_test():
    state_df = pd.read_csv(log_dir.joinpath("state.csv"))

    try:
        obstacle_df = pd.read_csv(log_dir.joinpath("obstacles.csv"))
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("no obstacle data")
        obstacle_df = None

    try:
        collision_df = pd.read_csv(log_dir.joinpath("collision_states.csv"))
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("no collision data")
        collision_df = None

    plt.figure(figsize=(10, 6))
    for i in range(state_df.at[len(state_df) - 1, "episode"]):
        if obstacle_df is not None:
            verts_l = util.extract_plottable_rectangles(obstacle_df, i)
            for vx, vy in verts_l:
                plt.plot(vx, vy, "k-")

        episode_states = state_df[state_df["episode"] == i]

        if collision_df is not None:
            verts_l = util.extract_plottable_rectangles(collision_df, i)
        else:
            verts_l = []

        for vx, vy in verts_l:
            plt.plot(vx, vy, "r-")

        if len(verts_l) > 0:
            plt.plot(episode_states["x"], episode_states["y"], "--")
        else:
            plt.plot(episode_states["x"], episode_states["y"], "-")

    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", label="path"),
    ]

    if collision_df is not None:
        legend_elements.append(
            Line2D([0], [0], color="black", linestyle="--", label="collision path")
        )
        legend_elements.append(
            Patch(facecolor="white", edgecolor="red", label="collision state")
        )
    if obstacle_df is not None:
        legend_elements.append(
            Patch(facecolor="black", edgecolor="black", label="obstacle")
        )

    plt.legend(handles=legend_elements, loc="upper right")
    plt.title("paths from origin under uniformly sampled controls")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([state_df["x"].min() - 1, state_df["x"].max() + 1])
    plt.gca().set_aspect("equal")
    plt.show()


def plan_with_obstacles():
    start_goal_df = pd.read_csv(log_dir.joinpath("start_goal.csv"))
    start_tree_df = pd.read_csv(log_dir.joinpath("start_tree.csv"))
    goal_tree_df = pd.read_csv(log_dir.joinpath("goal_tree.csv"))
    
    start = (float(start_goal_df["x_start"].item()), float(start_goal_df["y_start"].item()))
    goal = (float(start_goal_df["x_goal"].item()), float(start_goal_df["y_goal"].item()))
    
    try:
        trajectory_df = pd.read_csv(log_dir.joinpath("trajectory.csv"))
    except (pd.errors.EmptyDataError):
        print("no trajectory")
        trajectory_df = None

    try:
        obstacle_df = pd.read_csv(log_dir.joinpath("obstacles.csv"))
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("no obstacles")
        obstacle_df = None

    plt.figure(figsize=(10, 6))
    
    # Start tree
    for _, branch_df in start_tree_df.groupby("branch"):
        plt.plot(branch_df["x"], branch_df["y"], "-")
        branch_df = branch_df.reset_index(drop=True)
        
        assert branch_df.at[len(branch_df) - 1, "x"] == start[0]
        assert branch_df.at[len(branch_df) - 1, "y"] == start[1]
    # Goal tree
    for _, branch_df in goal_tree_df.groupby("branch"):
        plt.plot(branch_df["x"], branch_df["y"], "--")
        branch_df = branch_df.reset_index(drop=True)
        assert branch_df.at[len(branch_df) - 1, "x"] == goal[0]
        assert branch_df.at[len(branch_df) - 1, "y"] == goal[1]
        
    # Planned trajectory tree
    if trajectory_df is not None:
        for _, branch_df in trajectory_df.groupby("branch"):
            plt.plot(branch_df["x"], branch_df["y"], "k-", linewidth=3.0)
            branch_df = branch_df.reset_index(drop=True)
            assert branch_df.at[len(branch_df) - 1, "x"].item() == start[0]
            assert branch_df.at[len(branch_df) - 1, "y"].item() == start[1]

    if obstacle_df is not None:
        verts_l = util.extract_plottable_rectangles(obstacle_df)
        for vx, vy in verts_l:
            plt.plot(vx, vy, "r-")

    plt.scatter(
        start_goal_df["x_start"],
        start_goal_df["y_start"],
        marker="x",
        c="g",
        s=36,
        label="start",
    )
    plt.scatter(
        start_goal_df["x_goal"],
        start_goal_df["y_goal"],
        marker="x",
        c="r",
        s=36,
        label="goal",
    )

    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", label="start tree"),
        Line2D([0], [0], color="black", linestyle="--", label="goal tree"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=3.0, alpha=1, label="trajectory"),
        Line2D(
            [0],
            [0],
            color="green",
            marker="x",
            linestyle="",
            markersize=8,
            label="start",
        ),
        Line2D(
            [0], [0], color="red", marker="x", linestyle="", markersize=8, label="goal"
        ),
    ]

    if obstacle_df is not None:
        legend_elements.append(
            Patch(facecolor="white", edgecolor="red", label="obstacle")
        )

    plt.title("rrt planning result")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.gca().set_aspect("equal")

    plt.legend(handles=legend_elements, loc="upper right")
    plt.show()


def interpolate_test():
    state_df = pd.read_csv(log_dir.joinpath("state.csv"))
    state_rectangles = util.extract_plottable_rectangles(state_df, ep=None)
    for i, (x, y) in enumerate(state_rectangles):
        if i == 0:
            plt.plot(x, y, c='g')
        elif i == len(state_rectangles)-1:
            plt.plot(x, y, c='r')
        else:
            plt.plot(x, y, c='k')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.gca().set_aspect('equal')
    plt.show()


name_to_plotter = {
    "dubins_paths_test": dubins_paths_test,
    "dubins_random_test": dubins_paths_test,
    "plan_with_obstacles_test": plan_with_obstacles,
    "plan_without_obstacles_test": plan_with_obstacles,
    "dubins_interpolate_test": interpolate_test,
    "dubins_distance_test": interpolate_test,
}

name_to_plotter[args.name]()
