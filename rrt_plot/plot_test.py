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
    
if __name__ == '__main__':
    q1c()