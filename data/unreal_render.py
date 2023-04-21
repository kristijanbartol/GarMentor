# You have to enable the following modules in UE 4.27
# - Python Editor Script Plugin
# - Editor Scripting Utilities

import unreal
import mrq_stills
from unreal import EditorLevelLibrary as ell
from unreal import EditorAssetLibrary as eal
import numpy as np
import pickle

#######################################
### Variables to be set by the User ###
#######################################

# NOTE: Consult https://github.com/pixelite1201/agora_evaluation/blob/ed739e0f5496dedb4a2f45f9a45aac4c479adea8/agora_evaluation/projection.py#L72
# to see which camera parameters to set inside of Unreal

# Set GUIDs of the cameras that can be used for rendering
# In Unreal Editor, search for Actor Guid, right click the value and select copy
CAMERA_GUIDS = {
    "archviz": [
        # Use camera information from here: https://github.com/pixelite1201/agora_evaluation/blob/master/agora_evaluation/projection.py#L93
        "37CE6B424488FEB72F25DC93E1FBBF7F",
        "FA919BBE41C94D2787FF1BBDDE19ECE5",
        "6CAF59A948A8C79C6B63098B871C666B",
        "2C5CE14B43680D766091B08B37987B7F"
    ],
    'brushifygrasslands': [],
    'brushifyforest': ["2FDA37744EA9D4C3FD8FB9B3F87D42D3"],
    'construction': ["9B7BD9C74BE978CA433EE099F354A995"],
    'flowers': ["16F3FC3B434A42D5813BD585E01FB5A9"]
}

# Camera height offsets according to https://github.com/pixelite1201/agora_evaluation/blob/ed739e0f5496dedb4a2f45f9a45aac4c479adea8/agora_evaluation/projection.py#L64
# Note that ground_plane is in meters and in OpenCV coordinate frame
# (so negative y parameter corresponds to height, see https://github.com/pixelite1201/agora_evaluation/blob/ed739e0f5496dedb4a2f45f9a45aac4c479adea8/agora_evaluation/projection.py#L41)
# a value of y=-1.7 would correspond to a height offset of +170
CAMERA_HEIGHT_OFFSETS = {
    'archviz': 0,
    'flowers': 170,
    'construction': 170,
    'brushifygrasslands': 170,
    'brushifyforest': 170
}

# Render script base directory
RENDER_SCRIPT_BASE_DIR = "/Game/Render_Script"

# Path to the engine directory that contains the scene configurations
SCENE_CONFIGS_BASE_DIR = RENDER_SCRIPT_BASE_DIR + "/Scene_Configurations"

# The scene for which renders should be created, make sure to also load the corresponding scene inside of the engine!
SCENE_NAME = None # e.g. "brushifyforest", "construction", "flowers", "archviz"

# Path to the MoviePipelineMasterConfig file that should be used for rendering
PATH_MOVIE_PIPELINE_MASTER_CONFIG = RENDER_SCRIPT_BASE_DIR + "/Movie_Pipeline_Configs/Still_Ultra_png"

# Directory where the Level Sequences will be created in
PATH_LEVEL_SEQUENCES = RENDER_SCRIPT_BASE_DIR + "/Level_Sequences"

# Whether the directory of a rendered scene should be deleted
DELETE_SCENE_DIRECTORIES_ONCE_RENDERED = False      # As we potentially have to do multiple renders of the same scene 
                                                    # (and reimporting all scene configurations is tedious), we disable this for now

# Path to the pkl file that contains the camera and subject transformation information
PATH_TRANSFORMATION_INFORMATION = f"D:\\data\\garmentor\\agora\\scenes\\{SCENE_NAME}\\transformation_info.pkl"

# World Outline folder where newly spawned bodies should be played to. Eases clean-up if the rendering is stopped unexpectedly
SPAWNED_ACTORS_FOLDER = 'Spawned_Bodies'

# Delegate for rendering finished event
RENDER_FINISHED_DELEGATE = unreal.OnMoviePipelineExecutorFinished()

# Movie Pipeline Executor can be one of these:
# 1) https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/MoviePipelinePIEExecutor.html?highlight=moviepipelinelinearexecutorbase
# 2) https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/MoviePipelineNewProcessExecutor.html?highlight=moviepipelineexecutorbase
# Discouraged: 3) https://docs.unrealengine.com/4.27/en-US/PythonAPI/class/MoviePipelineInProcessExecutor.html?highlight=moviepipelinelinearexecutorbase
MOVIE_PIPELINE_EXECUTOR = unreal.MoviePipelinePIEExecutor #NewProcessExecutor #PIEExecutor

# How many scenes should be rendered in one go
# Necessary since VRAM usage is increasing with each scene
# Specify -1 to render all scenes at once
RENDER_BATCH_SIZE = 50

# How many batches (of size specified above) have already been processed.
# For each step, this number has to be manually increased by one
BATCHES_ALREADY_PROCESSED = 0

##############################################
### End of variables to be set by the user ###
##############################################

# Path to the directory that contains the scene configurations for the current scene
SCENE_CONFIGS_DIR = SCENE_CONFIGS_BASE_DIR + "/" + SCENE_NAME

# List to keep track of which scene directories were not yet rendered
__UNRENDERED_SCENE_DIRECTORIES = []

# List to keep track of all actors that were spawned
__ACTORS_SPAWNED = []

# For internal use, keeps track of the total amount of scenes found
__NUMBER_SCENES_FOUND = -1

def spawn_actor(obj, x, y, z, z_yaw):
    actor_location = unreal.Vector(x, y, z)
    actor_rotation = unreal.Rotator(0.0, 0.0, z_yaw)
    actor = ell.spawn_actor_from_object(obj, actor_location, actor_rotation)
    actor.set_folder_path(SPAWNED_ACTORS_FOLDER)
    return actor


def find_camera_to_use(scene_config_name):
    all_actors = ell.get_all_level_actors()
    if SCENE_NAME == "archviz":
        # archviz has four static cameras
        cam_idx = int(scene_config_name.split('_')[-2].split('cam')[1])
    else:
        # Remaining scenes have only one dynamic camera
        cam_idx = 0
    for actor in all_actors:
        if actor.get_editor_property("actor_guid").to_string() == CAMERA_GUIDS[SCENE_NAME][cam_idx]:
            return actor
    raise ValueError(f"Could not find camera for scene {SCENE_NAME} with index {cam_idx} and specified GUID {CAMERA_GUIDS[SCENE_NAME][cam_idx]}")


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
    while len(__ACTORS_SPAWNED) > 0:
        actor = __ACTORS_SPAWNED.pop()
        ell.destroy_actor(actor)
    ell.save_current_level()
    # Delete directory containing the static mesh assets that were rendered
    if len(__UNRENDERED_SCENE_DIRECTORIES) == 0:
        unreal.log_warning("Scene directory of the current scene does not exist, stopping render process!")
        return
    dir_to_delete = __UNRENDERED_SCENE_DIRECTORIES.pop(0)
    if DELETE_SCENE_DIRECTORIES_ONCE_RENDERED:
        eal.delete_directory(dir_to_delete)
    # Start rendering of next batch of meshes
    main()


def render_scene(scene_config_name, camera_transformation):
    movie_pipeline_queue_subsystem = unreal.get_editor_subsystem(
        unreal.MoviePipelineQueueSubsystem
    )
    queue = movie_pipeline_queue_subsystem.get_queue()
    camera = find_camera_to_use(scene_config_name)
    if camera_transformation:
        camera.set_actor_location_and_rotation(
            unreal.Vector(
                camera_transformation['x'],
                camera_transformation['y'],
                camera_transformation['z'] + CAMERA_HEIGHT_OFFSETS[SCENE_NAME]
            ),
            unreal.Rotator(0.0, 0.0, camera_transformation['yaw']),
            False,
            True
        )
    render_config = unreal.load_asset(PATH_MOVIE_PIPELINE_MASTER_CONFIG)
    editor_world_name = unreal.SoftObjectPath(
        ell.get_editor_world().get_path_name()
    )
    job_name = f"{scene_config_name}"
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
        "Garmentor ClothAGORA Rendering Script"
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
    if not eal.does_asset_exist(PATH_MOVIE_PIPELINE_MASTER_CONFIG):
        raise ValueError(f"The Movie Pipeline Master Config asset does not "
        f"exist at the specified location: {PATH_MOVIE_PIPELINE_MASTER_CONFIG}")
    if not eal.does_directory_exist(PATH_LEVEL_SEQUENCES):
        eal.make_directory(PATH_LEVEL_SEQUENCES)
    else:
        delete_level_sequences(PATH_LEVEL_SEQUENCES)
    eal.save_directory(SCENE_CONFIGS_DIR)
    return


def find_subdirectories(base_dir):
    """Returns a list containing all direct subdirectories of the provided directory"""
    assets = eal.list_assets(base_dir, False, True)
    subdirs = [asset for asset in assets if eal.does_directory_exist(asset)]
    return subdirs


def get_batched_subdirectories(subdirectories, batch_size, batch_number):
    """For the given batch size, returns the provided batch"""
    if batch_size <= 0:
        return subdirectories
    return subdirectories[
        batch_number * batch_size : (batch_number+1) * batch_size
    ]


def main():
    if len(__UNRENDERED_SCENE_DIRECTORIES) == 0:
        if (BATCHES_ALREADY_PROCESSED + 1) * RENDER_BATCH_SIZE >= __NUMBER_SCENES_FOUND:
            unreal.log_warning("Finished rendering of all batches!")
        else:
            unreal.log_warning("Finished rendering of all scenes in the provided batch!")
            unreal.log_warning(
                "Set value of BATCHES_ALREADY_PROCESSED to "
                f"{BATCHES_ALREADY_PROCESSED + 1} in order to continue with "
                "the next batch."
            )
        return
    scene_dir = __UNRENDERED_SCENE_DIRECTORIES[0]
    scene_config_name = scene_dir.split('/')[-2]
    meshes_to_render = find_meshes_to_use(scene_dir)    
    # Load meshes and place them inside the scene
    for mesh in meshes_to_render:
        loaded_mesh = eal.load_asset(mesh)
        subject_name = loaded_mesh.get_name().split('_')[0]
        trans = TRANS_INFO[scene_config_name][subject_name]
        actor = spawn_actor(
            loaded_mesh,
            trans['x'],
            trans['y'],
            trans['z'],
            trans['yaw']
        )
        __ACTORS_SPAWNED.append(actor)
    ell.save_current_level()
    # Render the scene
    render_scene(
        scene_config_name,
        None if SCENE_NAME == "archviz" else TRANS_INFO[scene_config_name]['camera']
    )
    # clean up is handled in function "on_rendering_finished" which is called
    # once the rendering has finished


if __name__ == '__main__':
    setup_and_check_required_directories()
    RENDER_FINISHED_DELEGATE.add_callable(on_rendering_finished)
    __UNRENDERED_SCENE_DIRECTORIES = find_subdirectories(SCENE_CONFIGS_DIR)
    __NUMBER_SCENES_FOUND = len(__UNRENDERED_SCENE_DIRECTORIES)
    if __NUMBER_SCENES_FOUND <= 0:
        print(f"Could not find scenes to render in {SCENE_CONFIGS_DIR}")
    else:
        __UNRENDERED_SCENE_DIRECTORIES = get_batched_subdirectories(
            __UNRENDERED_SCENE_DIRECTORIES,
            RENDER_BATCH_SIZE,
            BATCHES_ALREADY_PROCESSED
        )
        TRANS_INFO = None
        # Load transformation information for the scene
        with open(PATH_TRANSFORMATION_INFORMATION, 'rb') as file:
            TRANS_INFO = pickle.load(file)
        main()
