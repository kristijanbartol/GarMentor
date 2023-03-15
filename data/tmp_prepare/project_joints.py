# NOTE (kbartol): Adapted from MPI (original copyright and license below):

#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#------------------------------------------------------------------------------
import argparse
import logging
import os
import pandas
from glob import glob
from tqdm import tqdm

from get_joints_verts_from_dataframe import add_joints_verts_in_dataframe


def run_projection(
        model_folder: str = 'data/model/smpl',
        debug_path: str = '',
        num_betas: int = 10,
        img_height: int = 720,
        img_width: int = 1280,
        img_folder: str = '',
        load_precomputed: str = '',
        model_type: str = 'SMPL',
        kid_template_path: str = 'data/',
        gt_model_path: str = 'data/GT_fits',
        debug: bool = False
    ):
    """Function to run the evaluation."""

    # Parse arguments to mimic the command line arguments.
    args = argparse.Namespace()
    args.modelFolder = model_folder
    args.debug_path = debug_path
    args.numBetas = num_betas
    args.imgHeight = img_height
    args.imgWidth = img_width
    args.imgFolder = img_folder
    args.loadPrecomputed = load_precomputed
    args.modeltype = model_type
    args.kid_template_path = kid_template_path
    args.gt_model_path = gt_model_path
    args.debug = debug

    # Because AGORA pose params are in vector format.
    args.pose2rot = True 

    print('heree i am')

    all_df = glob(os.path.join(args.loadPrecomputed, '*.pkl'))
    for df_iter, df_path in tqdm(enumerate(all_df)):
        logging.info(
            'Processing {}th dataframe'.format(
                str(df_iter)))
        df = pandas.read_pickle(df_path)
        # Check if gt joints and verts are stored in dataframe. If not
        # generate them ####
        if 'gt_joints_2d' not in df or 'gt_joints_3d' not in df:
            logging.info('Generating Ground truth joints')
            store_joints = True
            df = add_joints_verts_in_dataframe(args, df, store_joints)
            # write new dataframe with joints and verts stored
            df.to_pickle(df_path.replace('.pkl', '_withjv.pkl'))
        else:
            raise KeyError('Joints already stored in the dataframe. Please remove it before processing')
