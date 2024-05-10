import matplotlib.pyplot as plt
import torch

def plot_compare_trajectories(traj_a, traj_b, int_dt, traj_a_name="Trajectory 1", traj_b_name="Trajectory 2"):
    traj_a.velocity =  torch.gradient(traj_a.position, dim=1)[0] / int_dt
    traj_a.acceleration =  torch.gradient(traj_a.velocity, dim=1)[0] / int_dt

    traj_b.velocity =  torch.gradient(traj_b.position, dim=1)[0] / int_dt
    traj_b.acceleration =  torch.gradient(traj_b.velocity, dim=1)[0] / int_dt

    position1_np = traj_a.position.cpu().numpy()[0,:,0]
    velocity1_np = traj_a.velocity.cpu().numpy()[0,:,0]
    acceleration1_np = traj_a.acceleration.cpu().numpy()[0,:,0]

    position2_np = traj_b.position.cpu().numpy()[0,:,0]
    velocity2_np = traj_b.velocity.cpu().numpy()[0,:,0]
    acceleration2_np = traj_b.acceleration.cpu().numpy()[0,:,0]

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot positions
    axs[0].plot(position1_np, label=f'{traj_a_name} Position')
    axs[0].plot(position2_np, label=f'{traj_b_name} Position')
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Position')
    axs[0].set_title('Comparison of Positions')
    axs[0].legend()
    axs[0].grid(True)

    # Plot velocities
    axs[1].plot(velocity1_np, label='Velocity')
    axs[1].plot(velocity2_np, label='Velocity')
    axs[1].set_xlabel('Time Steps')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('Comparison of Velocities')
    axs[1].legend()
    axs[1].grid(True)

    # Plot accelerations
    axs[2].plot(acceleration1_np, label='Acceleration')
    axs[2].plot(acceleration2_np, label='Acceleration')
    axs[2].set_xlabel('Time Steps')
    axs[2].set_ylabel('Acceleration')
    axs[2].set_title('Comparison of Accelerations')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()