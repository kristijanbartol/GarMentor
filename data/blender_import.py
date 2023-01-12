import os
from os import path as osp
import bpy
from addon_utils import check

# Location of the downloaded send2ue plugin (version 2.0.2 is required)
ue_addon_location = "C:\\Users\\Julien\\Downloads\\send2ue_2.0.2.zip"

# Directory that contains all scene configurations that should be transferred
# to Unreal Engine
scene_location = "D:\\data\\garmentor\\agora\\scenes\\archviz"
#obj_file_location = "C:\\Users\\Julien\\Downloads\\smpl_gt\\trainset_renderpeople_p27_adults_body"

# How many scene configurations should be transferred to Unreal Engine at once.
# This will also determine how many scenes will be rendered in Unreal Engine
# in a single step.
# Provide -1 to transfer all available configurations at once
batch_size = 40 # -1

# How many batches (of the size specified above) have already been transferred.
# For each step, this number has to be manually increased by one.
batches_already_processed = 1 # 0

# Setup send2ue plugin

# Check if send2ue plugin is enabled
is_enabled, is_loaded = check('send2ue')

if not is_enabled and not is_loaded:
    # Install Send2UE addon
    bpy.ops.preferences.addon_install(
        overwrite=False,
        target='DEFAULT',
        filepath=ue_addon_location,
        filter_folder=True,
        filter_python=False,
        filter_glob="*.py;*.zip"
    )
    bpy.ops.preferences.addon_enable(module="send2ue")

export_collection = bpy.context.scene.collection.children.get("Export")
# send2ue plugin is now enabled and usable

scene_configurations = [el for el in sorted(os.listdir(scene_location)) \
    if osp.isdir(osp.join(scene_location, el))]

if batch_size <= 0:
    batch_size = len(scene_configurations)

for scene_idx in range(
    batches_already_processed * batch_size,
    (batches_already_processed + 1) * batch_size
):
    # Deselect all previously selected objects
    bpy.ops.object.select_all(action='DESELECT')

    scene_name = scene_configurations[scene_idx]
    obj_files = [el for el in os.listdir(osp.join(scene_location, scene_name))\
        if os.path.splitext(el)[1] == '.obj']
    number_subjects = len(obj_files)

    # Where to send the mesh to inside of UE
    bpy.data.scenes['Scene'].send2ue.unreal_mesh_folder_path = \
        f"/Game/Render_Script/scenes/{scene_name}"

    for obj_file in obj_files:
        imported_obj_file = bpy.ops.import_scene.obj(
            filepath=osp.join(scene_location, scene_name, obj_file),
            use_split_objects=False
        )
        # blender automatically selects imported objects
        selected_objects = bpy.context.selected_objects

        # set pivot point to z-value of lowest vertex
        # found_body = False
        # for mesh_object in selected_objects:
        #     if not 'body' in mesh_object.name:
        #         continue
        #     bpy.context.view_layer.objects.active = mesh_object
        #     bpy.ops.object.mode_set(mode="OBJECT")
        #     verts = mesh_object.data.vertices
        #     wmx = mesh_object.matrix_world
        #     z_coords = [(wmx @ v.co).z for v in verts]
        #     min_z = min(z_coords)
        #     bpy.context.scene.cursor.location = (0.0, 0.0, min_z)
        #     found_body = True
        # if not found_body:
        #     raise ValueError(
        #         "The script was unable to set the pivot point, most likely "
        #         "because no body mesh was found."
        #     )

        for mesh_object in selected_objects:
            # set pivot point to z-value of lowest vertex
            bpy.context.view_layer.objects.active = mesh_object
            bpy.ops.object.mode_set(mode="OBJECT")
            verts = mesh_object.data.vertices
            wmx = mesh_object.matrix_world
            z_coords = [(wmx @ v.co).z for v in verts]
            min_z = min(z_coords)
            bpy.context.scene.cursor.location = (0.0, 0.0, min_z)
            # move such that feet are at
            # world origin (which is used as pivot point)
            bpy.context.view_layer.objects.active = mesh_object
            bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
            mesh_object.location.z = 0.0
            # In case a mesh has multiple materials assigned
            for material_slot in mesh_object.material_slots:
                # Deactivate the mesh material's specular image texture
                mesh_material = material_slot.material
                if not mesh_material.use_nodes:
                    mesh_material.specular_intensity = 0
                else:
                    for n in mesh_material.node_tree.nodes:
                        if n.type == 'BSDF_PRINCIPLED':                        
                            for link in n.inputs["Specular"].links:
                                mesh_material.node_tree.links.remove(link)
                            n.inputs["Specular"].default_value = 0
            # Move object from default 'Collection' collection to the 'Export'
            # collection for the send2ue plugin
            for coll in mesh_object.users_collection:
                # Unlink mesh from all collections. By default, this should
                # only be the 'Collection' collection
                coll.objects.unlink(mesh_object)
            # Link to 'Export' collection
            export_collection.objects.link(mesh_object)
    
    # Send2UE
    bpy.ops.wm.send2ue()

    # Delete all meshes in the 'Export' collection for clean-up
    for mesh in export_collection.objects:
        bpy.data.objects.remove(mesh, do_unlink=True)