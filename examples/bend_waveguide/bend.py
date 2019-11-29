"""

To run an optimization:
$ python3 bend.py run save-folder

To view results:
$ python3 bend.py view save-folder

To see optimization status quickly:
$ python3 bend.py view_quick save-folder

To resume an optimization:
$ python3 bend.py resume save-folder

To generate a GDS file of the grating:
$ python3 bend.py gen_gds save-folder
"""
import os
import pickle
import shutil

import gdspy
import numpy as np
from typing import List, NamedTuple, Tuple

# `spins.invdes.problem_graph` contains the high-level spins code.
from spins.invdes import problem_graph
# Import module for handling processing optimization logs.
from spins.invdes.problem_graph import log_tools
# `spins.invdes.problem_graph.optplan` contains the optimization plan schema.
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace

# If `True`, also minimize the back-reflection.
MINIMIZE_BACKREFLECTION = False

# Yee cell grid spacing in nanometers.
GRID_SPACING = 40

def run_opt(save_folder: str) -> None:
    """Main optimization script.

    This function setups the optimization and executes it.

    Args:
        save_folder: Location to save the optimization data.
    """
    os.makedirs(save_folder)

    wg_thickness = 220

    sim_space = create_sim_space(
        "sim_fg.gds",
        "sim_bg.gds",
        wg_thickness=wg_thickness)
    obj, monitors = create_objective(
        sim_space, wg_thickness=wg_thickness)
    trans_list = create_transformations(
        obj, monitors, sim_space, 200, num_stages=5, min_feature=100)
    plan = optplan.OptimizationPlan(transformations=trans_list)

    # Save the optimization plan so we have an exact record of all the
    # parameters.
    with open(os.path.join(save_folder, "optplan.json"), "w") as fp:
        fp.write(optplan.dumps(plan))
    # Copy over the GDS files.
    shutil.copyfile("sim_fg.gds", os.path.join(save_folder, "sim_fg.gds"))
    shutil.copyfile("sim_bg.gds", os.path.join(save_folder, "sim_bg.gds"))

    # Execute the optimization and indicate that the current folder (".") is
    # the project folder. The project folder is the root folder for any
    # auxiliary files (e.g. GDS files).
    problem_graph.run_plan(plan, ".", save_folder=save_folder)


def create_sim_space(
        gds_fg_name: str,
        gds_bg_name: str,
        wg_thickness: float = 220,
        wg_length: float = 3000,
        wg_width: float = 200,
        buffer_len: float = 250,
        dx: int = 40,
        num_pmls: int = 10
) -> optplan.SimulationSpace:
    """Creates the simulation space.

    The simulation space contains information about the boundary conditions,
    gridding, and design region of the simulation.

    Args:
        gds_fg_name: Location to save foreground GDS.
        gds_bg_name: Location to save background GDS.
        wg_thickness: Thickness of the waveguide.
        dx: Grid spacing to use.
        num_pmls: Number of PML layers to use on each side.

    Returns:
        A `SimulationSpace` description.
    """

    sim_size = wg_length + buffer_len * 2

    waveguide_input = gdspy.Rectangle((-sim_size/2, -wg_width/2),
                                      (-wg_length/2, wg_width/2), 100)
    waveguide_output = gdspy.Rectangle((-wg_width/2, -sim_size/2),
                                      (wg_width/2, -wg_length/2), 100)
    design_region = gdspy.Rectangle((-wg_length/2, -wg_length/2),
                                    (wg_length/2, wg_length/2), 100)
    
    gds_fg = gdspy.Cell("FOREGROUND", exclude_from_current=True)
    gds_fg.add(waveguide_input)
    gds_fg.add(waveguide_output)
    gds_fg.add(design_region)

    gds_bg = gdspy.Cell("BACKGROUND", exclude_from_current=True)
    gds_bg.add(waveguide_input)
    gds_bg.add(waveguide_output)

    gdspy.write_gds(gds_fg_name, [gds_fg], unit=1e-9, precision=1e-9)
    gdspy.write_gds(gds_bg_name, [gds_bg], unit=1e-9, precision=1e-9)

    mat_oxide = optplan.Material(index=optplan.ComplexNumber(real=1.5))
    stack = [
        optplan.GdsMaterialStackLayer(
            foreground=mat_oxide,
            background=mat_oxide,
            gds_layer=[100, 0],
            extents=[-10000, -wg_thickness / 2],
        ),
        optplan.GdsMaterialStackLayer(
            foreground=optplan.Material(
                index=optplan.ComplexNumber(real=2.2)),
            background=mat_oxide,
            gds_layer=[100, 0],
            extents=[-wg_thickness / 2, wg_thickness / 2],
        ),
    ]

    mat_stack = optplan.GdsMaterialStack(
        # Any region of the simulation that is not specified is filled with
        # oxide.
        background=mat_oxide,
        stack=stack,
    )

    # Create a simulation space for both continuous and discrete optimization.
    simspace = optplan.SimulationSpace(
        name="simspace",
        mesh=optplan.UniformMesh(dx=dx),
        eps_fg=optplan.GdsEps(gds=gds_fg_name, mat_stack=mat_stack),
        eps_bg=optplan.GdsEps(gds=gds_bg_name, mat_stack=mat_stack),
        sim_region=optplan.Box3d(
            center=[0, 0, 0],
            extents=[wg_length*2, wg_length*2, dx],
        ),
        selection_matrix_type="direct_lattice",
        pml_thickness=[num_pmls, num_pmls, num_pmls, num_pmls, 0, 0],
    )

    return simspace


def create_objective(
        sim_space: optplan.SimulationSpace,
        wg_thickness: float = 220,
        wg_length: float = 3000,
        wg_width: float = 200,
        buffer_len: float = 250,
        dx: int = 40,
        num_pmls: int = 10
) -> Tuple[optplan.Function, List[optplan.Monitor]]:
    """Creates an objective function.

    The objective function is what is minimized during the optimization.

    Args:
        sim_space: The simulation space description.
        wg_thickness: Thickness of waveguide.

    Returns:
        A tuple `(obj, monitor_list)` where `obj` is an objectivce function that
        tries to maximize the coupling efficiency of the grating coupler and
        `monitor_list` is a list of monitors (values to keep track of during
        the optimization.
    """
    # Keep track of metrics and fields that we want to monitor.
    monitor_list = []

    wlen = 1550
    sim_size = wg_length + buffer_len * 2
    epsilon = optplan.Epsilon(
        simulation_space=sim_space,
        wavelength=wlen,
    )
    monitor_list.append(optplan.FieldMonitor(name="mon_eps", function=epsilon))

    # Add a Gaussian source that is angled at 10 degrees.
    sim = optplan.FdfdSimulation(
        source=optplan.PlaneWaveSource(
            polarization_angle=0,
            theta=np.deg2rad(-10),
            psi=np.pi / 2,
            center=[-sim_size/2, 0, 0],
            extents=[dx, 1500, 600],
            normal=[1, 0, 0],
            power=1,
            normalize_by_sim=True,
        ),
        solver="local_direct",
        wavelength=wlen,
        simulation_space=sim_space,
        epsilon=epsilon,
    )
    monitor_list.append(
        optplan.FieldMonitor(
            name="mon_field",
            function=sim,
            normal=[0, 0, 1],
            center=[0, 0, 0],
        ))

    wg_overlap = optplan.WaveguideModeOverlap(
        center=[0, -sim_size/2, 0],
        extents=[600, dx, 600],
        mode_num=0,
        normal=[0, -1, 0],
        power=1.0,
    )
    power = optplan.abs(optplan.Overlap(simulation=sim, overlap=wg_overlap))**2
    monitor_list.append(optplan.SimpleMonitor(name="mon_power", function=power))

    obj = 1 - power

    monitor_list.append(optplan.SimpleMonitor(name="objective", function=obj))
    return obj, monitor_list

def create_transformations(
        obj: optplan.Function,
        monitors: List[optplan.Monitor],
        sim_space: optplan.SimulationSpaceBase,
        cont_iters: int,
        num_stages: int = 3,
        min_feature: float = 100,
) -> List[optplan.Transformation]:
    """Creates a list of transformations for the device optimization.

    The transformations dictate the sequence of steps used to optimize the
    device. The optimization uses `num_stages` of continuous optimization. For
    each stage, the "discreteness" of the structure is increased (through
    controlling a parameter of a sigmoid function).

    Args:
        opt: The objective function to minimize.
        monitors: List of monitors to keep track of.
        sim_space: Simulation space ot use.
        cont_iters: Number of iterations to run in continuous optimization
            total acorss all stages.
        num_stages: Number of continuous stages to run. The more stages that
            are run, the more discrete the structure will become.
        min_feature: Minimum feature size in nanometers.

    Returns:
        A list of transformations.
    """
    # Setup empty transformation list.
    trans_list = []

    # First do continuous relaxation optimization.
    # This is done through cubic interpolation and then applying a sigmoid
    # function.
    param = optplan.CubicParametrization(
        # Specify the coarseness of the cubic interpolation points in terms
        # of number of Yee cells. Feature size is approximated by having
        # control points on the order of `min_feature / GRID_SPACING`.
        undersample=3.5 * min_feature / GRID_SPACING,
        simulation_space=sim_space,
        init_method=optplan.UniformInitializer(min_val=0, max_val=1),
    )

    iters = max(cont_iters // num_stages, 1)
    for stage in range(num_stages):
        trans_list.append(
            optplan.Transformation(
                name="opt_cont{}".format(stage),
                parametrization=param,
                transformation=optplan.ScipyOptimizerTransformation(
                    optimizer="L-BFGS-B",
                    objective=obj,
                    monitor_lists=optplan.ScipyOptimizerMonitorList(
                        callback_monitors=monitors,
                        start_monitors=monitors,
                        end_monitors=monitors),
                    optimization_options=optplan.ScipyOptimizerOptions(
                        maxiter=iters),
                ),
            ))

        if stage < num_stages - 1:
            # Make the structure more discrete.
            trans_list.append(
                optplan.Transformation(
                    name="sigmoid_change{}".format(stage),
                    parametrization=param,
                    # The larger the sigmoid strength value, the more "discrete"
                    # structure will be.
                    transformation=optplan.CubicParamSigmoidStrength(
                        value=4 * (stage + 1)),
                ))
    return trans_list


def view_opt(save_folder: str) -> None:
    """Shows the result of the optimization.

    This runs the auto-plotter to plot all the relevant data.
    See `examples/wdm2` IPython notebook for more details on how to process
    the optimization logs.

    Args:
        save_folder: Location where the log files are saved.
    """
    log_df = log_tools.create_log_data_frame(
        log_tools.load_all_logs(save_folder))
    monitor_descriptions = log_tools.load_from_yml(
        os.path.join(os.path.dirname(__file__), "monitor_spec.yml"))
    log_tools.plot_monitor_data(log_df, monitor_descriptions)


def view_opt_quick(save_folder: str) -> None:
    """Prints the current result of the optimization.

    Unlike `view_opt`, which plots fields and optimization trajectories,
    `view_opt_quick` prints out scalar monitors in the latest log file. This
    is useful for having a quick look into the state of the optimization.

    Args:
        save_folder: Location where the log files are saved.
    """
    with open(workspace.get_latest_log_file(save_folder), "rb") as fp:
        log_data = pickle.load(fp)
        for key, data in log_data["monitor_data"].items():
            if np.isscalar(data):
                print("{}: {}".format(key, data.squeeze()))


def resume_opt(save_folder: str) -> None:
    """Resumes a stopped optimization.

    This restarts an optimization that was stopped prematurely. Note that
    resuming an optimization will not lead the exact same results as if the
    optimization were finished the first time around.

    Args:
        save_folder: Location where log files are saved. It is assumed that
            the optimization plan is also saved there.
    """
    # Load the optimization plan.
    with open(os.path.join(save_folder, "optplan.json")) as fp:
        plan = optplan.loads(fp.read())

    # Run the plan with the `resume` flag to restart.
    problem_graph.run_plan(plan, ".", save_folder=save_folder, resume=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("run", "view", "view_quick", "resume", "gen_gds"),
        help="Must be either \"run\" to run an optimization, \"view\" to "
        "view the results, \"resume\" to resume an optimization, or "
        "\"gen_gds\" to generate the grating GDS file.")
    parser.add_argument(
        "save_folder", help="Folder containing optimization logs.")

    args = parser.parse_args()
    if args.action == "run":
        run_opt(args.save_folder)
    elif args.action == "view":
        view_opt(args.save_folder)
    elif args.action == "view_quick":
        view_opt_quick(args.save_folder)
    elif args.action == "resume":
        resume_opt(args.save_folder)
