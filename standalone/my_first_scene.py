

########
# LAUNCH THE SIMULATOR FIRST
########
import argparse
from omni.isaac.lab.app import AppLauncher

# create argparse
parser = argparse.ArgumentParser(description="Creating empty scene")

# Add Isaac Sim specific arguments to the parser
# This includes options like:
# --headless
# --multi-gpu
# --window-width/height
AppLauncher.add_app_launcher_args(parser)

# Parse the command line arguments
args_cli = parser.parse_args()

# Launch the omniverse application
# This starts up the actual simulator engine
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""
After this point, we can import Isaac Sim specific modules because the simulator is now running
"""
from omni.isaac.lab.sim import SimulationCfg, SimulationContext, DistantLightCfg, GroundPlaneCfg, ConeCfg
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from pxr import Gf, Sdf, Usd
import omni.isaac.lab.utils.math as math_utils
import torch


THOMAS_ASSETS:str = "/home/thomas/Assets"


def design_scene() -> tuple[dict[str, RigidObject] ,list[list[float]]]:
    """
    #scene_entities: dict[str, RigidObject] = {.
    Creates all the objects in our scene.
    This must be done before starting the simulation
    """
    # Create ground plane
    cfg_ground: GroundPlaneCfg = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Add distant light to illuminate the scene
    cfg_light_distant: DistantLightCfg = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75), # Light color in RGB
    )
    cfg_light_distant.func("/World/distantLight", cfg_light_distant, translation=(1, 0, 10))     #func: Callable = lights.spawn_light

    # Create a Xform (transform) prim to group all our objects
    prim_utils.create_prim("/World/Objects", "Xform")

    # Create two static red cones (Visual only, no physics)
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # Set the material to red using PreviewSurfaceCfg
    )
    # Place the cones at different positions
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(1.0, -1.0, 1.0))

    # Create a green cone with physics
    cfg_cone_rigid: ConeCfg = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        # Add physics properties
        rigid_props=sim_utils.RigidBodyPropertiesCfg(), # Makes it a rigid body
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(), # Enables collisions
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), # Make it Green
    )
    # Place and rotate the cone
    cfg_cone_rigid.func(
        "/World/Objects/Cone3_rigid",
        cfg_cone_rigid,
        translation=(0.0, 0.0, 1.0),
        orientation=(0.5, 0.0, 0.5, 0.0)
    )

    # Create a blue deformable cuboid (soft body physics)
    # Note: Deformable bodies only work with GPU simulation
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        # Adds deformable physics properties
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func(
        "/World/Objects/Cuboid_deformable",
        cfg_cuboid_deformable,
        translation=(0.15, 0.0, 2.0),
    )

    # Load a table from a USD file
    # USD files can contain complex meshes with material already defines
    cfg_table = sim_utils.UsdFileCfg(
        #usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        usd_path=f"{THOMAS_ASSETS}/rebuild_cart_pole/final.usd",

    )
    table: Usd.Prim = cfg_table.func(
        "/World/Objects/Table",
        cfg_table,
        translation=(0.0, 0.0, 1.05)
    )


    # Create multiple spawn location using Xform prims
    # Each location will have its own cone
    origins = [[0.25, 0.25, 0.0],
               [-0.25, 0.25, 0.0],
               [0.25, -0.25, 0.0],
               [-0.25, -0.25, 0.0]]

    for i, origin in enumerate(origins):
        # Create an Xform (transform) for each spawn point
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
    #######
    # Create the rigid object configuration
    ######
    # This defines a yellow cone with physics properties
    # RigidObject has additional functionality over simply ConeCfg().func()
    # - it tracks physics state
    # - provides methods to manipulate objects during simulation
    # - handles multiple instances (via regex)
    # - Manages internal buffers for physics data
    yellow_cones_cfg = RigidObjectCfg(
        # Use regex to spawn at all Origin locations
        prim_path="/World/Origin.*/Cone",
        # Define the cons physical and visual props
        spawn=sim_utils.ConeCfg(
            radius=0.30,
            height=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),
                metallic=3.0,
            ),
        ),
        # Initialize with default state
        init_state=RigidObjectCfg.InitialStateCfg()
    )
    # Create the rigid object instance
    yellow_cone_object = RigidObject(cfg=yellow_cones_cfg)

    # Return objects needed for simualtion
    scene_entities = {
        "yellow_cones":yellow_cone_object,
    }

    return scene_entities, origins




def main() -> None:
    """
    Main function that sets up and runs the simulation.
    """
    # create simulation configuration
    # dt=0.01 means physics will update every 0.01 sec (100Hz)
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)

    # Initialize the simulation context with our configuration
    # This creates the physics engine and sets up the simulation environment
    sim = SimulationContext(sim_cfg)

    # Position the camera in the 3D space
    # First array: Camera position (x,y,z)
    # Second array: Look-at point (center of scene)
    sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))

    # Create all our objects
    scene_entities, origins = design_scene()
    origins = torch.tensor(origins, device=sim.device)
    yellow_cone_object = scene_entities["yellow_cones"]

    # Initialize the physics engine and start the simulation
    # This is crucial - must be called before stepping the simulation
    # 1. Plays the timeline
    # 2. Initialize physics handles
    # 3. Sets up internal simulation rate
    sim.reset()

    print("[INFO]: Setup complete...")

    # Track time for periodic resets
    current_time = 0.0
    reset_interval = 2.0
    sim_dt = sim_cfg.dt

    # Main simulation loop
    # This runs continuously while the application window is open
    while simulation_app.is_running():
        # Check if it`s time to reset the cones
        if current_time >= reset_interval:
            current_time = 0.0

            # Reset cone position
            root_state = yellow_cone_object.data.default_root_state.clone()
            # Add base position from origins
            root_state[:, :3] += origins[0]
            # Add random height and position variations
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.2,
                h_range=(0.25, 0.5),
                size=yellow_cone_object.num_instances,
                device=yellow_cone_object.device,
            )
            # set velocity of first cone:
            root_state[0, 7:10] = torch.tensor([0.0, 0.0, 10.0], device=yellow_cone_object.device)

            # Apply new positions to simulation
            # Tells physics engine where objects should be
            yellow_cone_object.write_root_state_to_sim(root_state)
            # Reset internal buffers
            yellow_cone_object.reset()

        # Write any external forces/data to simulation
        yellow_cone_object.write_data_to_sim()

        # Performs one simulation step
        # This:
        # 1. Updates physics (collisions, forces, etc...)
        # 2. Updates render state
        # 3. Processes any external input
        sim.step()

        # Updates objects internal state from simulation
        yellow_cone_object.update(sim_dt)

        # Update time tracking
        current_time += sim_dt

if __name__ == "__main__":
    # Run the main simulation
    main()
    # Clean up and close the simulator
    # This ensures proper cleanup of Omniverse resources
    simulation_app.close()