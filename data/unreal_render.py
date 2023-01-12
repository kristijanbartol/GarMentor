# You have to enable the following modules in UE 4.27
# - Python Editor Script Plugin
# - Editor Scripting Utilities

import unreal
import mrq_stills
from unreal import EditorLevelLibrary as ell
from unreal import EditorAssetLibrary as eal
import numpy as np
import pickle

# Set GUIDs of the cameras that can be used for rendering
# In Unreal Editor, search for Actor Guid, right click the value and select copy
ARCHVIZ_CAMERA_GUIDS = [
    "354B0DBE46586EEE563B3AAC7B34626E", # Cam00: (X=374.888062,Y=-287.670349,Z=261.453613) (Pitch=-30.000000,Yaw=-227.000000,Roll=-0.000012) (SensorWidth=83.102051,SensorHeight=46.745972,SensorAspectRatio=1.777737) (MinFocalLength=4.000000,MaxFocalLength=1000.000000,MinFStop=1.200000,MaxFStop=22.000000,MinimumFocusDistance=15.000000,DiaphragmBladeCount=7)
    "DA1A88B846241D11B446DBAAE3231740", # Cam01: (X=394.000000,Y=180.000000,Z=258.000000) (Pitch=-34.999977,Yaw=220.000015,Roll=0.000009) (SensorWidth=78.464050,SensorHeight=44.541599,SensorAspectRatio=1.761590) (MinFocalLength=4.000000,MaxFocalLength=50.000000,MinFStop=1.200000,MaxFStop=22.000000,MinimumFocusDistance=15.000000,DiaphragmBladeCount=7)
    "DDDCB82241E91637B4716C877411834D", # Cam02: (X=-449.000000,Y=127.000000,Z=258.000000) (Pitch=-35.000000,Yaw=-45.000000,Roll=0.000006) (SensorWidth=86.579849,SensorHeight=49.041878,SensorAspectRatio=1.765427) (MinFocalLength=4.000000,MaxFocalLength=50.000000,MinFStop=1.200000,MaxFStop=22.000000,MinimumFocusDistance=15.000000,DiaphragmBladeCount=7)
    "87D45E2344503FA043DA69838F196FE0"  # Cam03: (X=-468.000000,Y=-263.000000,Z=262.000000) (Pitch=-34.999977,Yaw=46.000000,Roll=0.000009) (SensorWidth=82.845421,SensorHeight=46.300926,SensorAspectRatio=1.789282) (MinFocalLength=4.000000,MaxFocalLength=50.000000,MinFStop=1.200000,MaxFStop=22.000000,MinimumFocusDistance=15.000000,DiaphragmBladeCount=7)
]
BRUSHIFYGRASSLANDS_CAMERA_GUIDS = []

# Render script base directory
RENDER_SCRIPT_BASE_DIR = "/Game/Render_Script"

# Path to the engine directory that contains the scene configurations
SCENES_BASE_DIR = RENDER_SCRIPT_BASE_DIR + "/scenes"

# Path to the MoviePipelineMasterConfig file that should be used for rendering
PATH_MOVIEPIPELINEMASTERCONFIG = RENDER_SCRIPT_BASE_DIR + "/Utils/Still_HD_png"

# Directory where the Level Sequences will be created in
PATH_LEVEL_SEQUENCES = RENDER_SCRIPT_BASE_DIR + "/Level_Sequences"

# List to keep track of all actors that were spawned
ACTORS_SPAWNED = []

# List to keep track of which scene directories were not yet rendered 
UNRENDERED_SCENE_DIRECTORIES = []

# Whether the directory of a rendered scene should be deleted
DELETE_SCENE_DIRECTORIES_ONCE_RENDERED = False      # TODO for deployment, set this back to True

# Path to the pkl file that contains the camera and subject transformation information
PATH_TRANSFORMATION_INFORMATION = f"D:\\data\\garmentor\\agora\\scenes\\archviz\\transformation_info.pkl" 

# Delegate for rendering finished event
RENDER_FINISHED_DELEGATE = unreal.OnMoviePipelineExecutorFinished()

# Movie Pipeline Executor can be one of these:
# 1) https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/MoviePipelinePIEExecutor.html?highlight=moviepipelinelinearexecutorbase
# 2) https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/MoviePipelineNewProcessExecutor.html?highlight=moviepipelineexecutorbase
# Discouraged: 3) https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/MoviePipelineInProcessExecutor.html?highlight=moviepipelinelinearexecutorbase
MOVIE_PIPELINE_EXECUTOR = unreal.MoviePipelinePIEExecutor #NewProcessExecutor #PIEExecutor

def spawn_actor(obj, x, y, z, z_yaw):
    actor_location = unreal.Vector(x, y, z)
    actor_rotation = unreal.Rotator(0.0, 0.0, z_yaw)
    return ell.spawn_actor_from_object(obj, actor_location, actor_rotation)


def find_camera_to_use(scene_name):
    all_actors = ell.get_all_level_actors()
    if "brushifygrasslands" in scene_name:
        cam_idx = 0
        for actor in all_actors:
            if actor.get_editor_property("actor_guid").to_string() == BRUSHIFYGRASSLANDS_CAMERA_GUIDS[cam_idx]:
                return actor
    elif "archviz" in scene_name:
        cam_idx = int(scene_name.split('_')[-2].split('cam')[1])
        for actor in all_actors:
            if actor.get_editor_property("actor_guid").to_string() == ARCHVIZ_CAMERA_GUIDS[cam_idx]:
                return actor
    else:
        return None


def find_meshes_to_use(asset_path):
    assets = eal.list_assets(asset_path)
    meshes = [asset for asset in assets \
        if eal.find_asset_data(asset).asset_class == "StaticMesh"]
    return meshes


def delete_level_sequences(sequence_path):
    level_sequences = eal.list_assets(sequence_path)
    for sequence in level_sequences:
        eal.delete_asset(sequence) 


def on_rendering_finished(pipeline_executor, success):
    """Called when the render step for one scene has finished.
    Used to clean up the editor for the next scene to render
    """
    print("Deleting movie pipeline jobs...")
    render_queue = unreal.get_editor_subsystem(
        unreal.MoviePipelineQueueSubsystem
    ).get_queue()
    render_jobs = render_queue.get_jobs()
    for job in render_jobs:
        render_queue.delete_job(job)
    print("Deleting level sequences...")
    delete_level_sequences(PATH_LEVEL_SEQUENCES)
    print("Destroying spawned actors...")
    while len(ACTORS_SPAWNED) > 0:
        actor = ACTORS_SPAWNED.pop()
        ell.destroy_actor(actor)
    ell.save_current_level()
    # Delete directory containing the static mesh assets that were rendered
    if len(UNRENDERED_SCENE_DIRECTORIES) == 0:
        unreal.log_warning("Scene directory of the current scene does not exist, stopping render process!")
        return
    dir_to_delete = UNRENDERED_SCENE_DIRECTORIES.pop()
    if DELETE_SCENE_DIRECTORIES_ONCE_RENDERED:
        eal.delete_directory(dir_to_delete)
    # Start rendering of next batch of meshes
    main()


def render_scene(scene_name, camera_transformation):
    movie_pipeline_queue_subsystem = unreal.get_editor_subsystem(
        unreal.MoviePipelineQueueSubsystem
    )
    queue = movie_pipeline_queue_subsystem.get_queue()
    camera = find_camera_to_use(scene_name)
    if camera_transformation:
        camera.set_actor_location_and_rotation(
            unreal.Vector(camera_transformation['x'], camera_transformation['y'], camera_transformation['z']),
            unreal.Rotator(0.0, 0.0, camera_transformation['yaw'])
        )
    render_config = unreal.load_asset(PATH_MOVIEPIPELINEMASTERCONFIG)
    editor_world_name = unreal.SoftObjectPath(
        ell.get_editor_world().get_path_name()
    )
    job_name = f"{scene_name}"
    sequence = mrq_stills.create_sequence_from_selection(
        asset_name=f"{job_name}",
        length_frames=1,
        package_path=PATH_LEVEL_SEQUENCES,
        target_camera=camera
    )
    movie_pipeline_executor_job = queue.allocate_new_job(
        unreal.MoviePipelineExecutorJob
    )
    movie_pipeline_executor_job.set_editor_property(
        "sequence",
        unreal.SoftObjectPath(path_string=sequence.get_path_name())
    )
    movie_pipeline_executor_job.set_editor_property("job_name", job_name)
    movie_pipeline_executor_job.set_configuration(render_config)
    movie_pipeline_executor_job.set_editor_property(
        "map",
        editor_world_name
    )
    movie_pipeline_executor_job.set_editor_property(
        "author",
        "Garmentor ClothAgora Rendering Script"
    )


    movie_pipeline_executor = movie_pipeline_queue_subsystem.render_queue_with_executor(MOVIE_PIPELINE_EXECUTOR)
    if movie_pipeline_executor:
        movie_pipeline_executor.set_editor_property(
            "on_executor_finished_delegate",
            RENDER_FINISHED_DELEGATE
        )
    else:
        unreal.log_error("Executor could not be created!")


def setup_and_check_required_directories():
    if not eal.does_asset_exist(PATH_MOVIEPIPELINEMASTERCONFIG):
        raise ValueError(f"The Movie Pipeline Master Config asset does not "
        f"exist at the specified location: {PATH_MOVIEPIPELINEMASTERCONFIG}")
    if not eal.does_directory_exist(PATH_LEVEL_SEQUENCES):
        eal.make_directory(PATH_LEVEL_SEQUENCES)
    else:
        delete_level_sequences(PATH_LEVEL_SEQUENCES)
    eal.save_directory(SCENES_BASE_DIR)
    return


def find_subdirectories(base_dir):
    """Returns a list containing all direct subdirectories of the provided directory"""
    assets = eal.list_assets(base_dir, False, True)
    subdirs = [asset for asset in assets if eal.does_directory_exist(asset)]
    return subdirs


def main():
    if len(UNRENDERED_SCENE_DIRECTORIES) == 0:
        print("Finished rendering of all batches!")
        return
    scene_dir = UNRENDERED_SCENE_DIRECTORIES[-1]    # Important that we take the last element
    scene_name = scene_dir.split('/')[-2]
    meshes_to_render = find_meshes_to_use(scene_dir)    
    # Load meshes and place them inside the scene
    for mesh in meshes_to_render:
        loaded_mesh = eal.load_asset(mesh)
        subject_name = loaded_mesh.get_name().split('_')[0]
        trans = TRANS_INFO[scene_name][subject_name]
        actor = spawn_actor(
            loaded_mesh,
            trans['x'],
            trans['y'],
            trans['z'],
            trans['yaw']
        )
        ACTORS_SPAWNED.append(actor)
    ell.save_current_level()
    # Render the scene
    render_scene(
        scene_name,
        None if "archviz" in scene_name else TRANS_INFO['camera']
    )
    # clean up is handled in function "on_rendering_finished" which is called
    # once the rendering has finished


if __name__ == '__main__':
    setup_and_check_required_directories()
    RENDER_FINISHED_DELEGATE.add_callable(on_rendering_finished)
    UNRENDERED_SCENE_DIRECTORIES = find_subdirectories(SCENES_BASE_DIR)
    TRANS_INFO = None
    # Load transformation information for the scene
    with open(PATH_TRANSFORMATION_INFORMATION, 'rb') as file:
        TRANS_INFO = pickle.load(file)
    main()