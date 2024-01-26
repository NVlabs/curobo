source /isaac-sim/setup_python_env.sh

python() {
    /isaac-sim/python.sh "$@"
}

pip() {
    /isaac-sim/python.sh -m pip "$@"
}

