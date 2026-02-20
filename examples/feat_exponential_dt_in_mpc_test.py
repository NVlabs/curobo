
from curobo.wrap.reacher.mpc import MpcSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml, get_world_configs_path

def check_timetraj_call():
    print("Checking where TimeTrajConfig is called...")
    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    world_cfg = load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    
    print("Calling MpcSolverConfig.load_from_robot_config with use_exp_dt=True...")
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        use_exp_dt=True,
    )

    print("Calling MpcSolverConfig.load_from_robot_config with use_exp_dt=False...")
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        use_exp_dt=False,
    )
    print("MpcSolverConfig loaded.")

if __name__ == "__main__":
    check_timetraj_call()
