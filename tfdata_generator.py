
#tf.data generator 
#for the CPH/Naive dataset using RABIES-preprocessed data.
#functions to build tf.data.Datasets for training and evaluation.


from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import tensorflow as tf 
import nibabel as nib
import numpy as np
import os
import cv2
import skimage
import scipy
from tensorflow.keras.models import Model
from ipywidgets import IntSlider, interact
from matplotlib import animation, rc
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from scipy import ndimage
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from collections import Counter
import random


# paths for raw and RABIES data
RAW_ROOT    = r"F:/rawdata/" #chante to your rawdata path
RABIES_ROOT = r"C:/Users/gdaalumno/Desktop/rabies"  #change to your rabies path

'''
Class to build lists of subjects/sessions and derive
corresponding file paths and labels for the CPH/Naive dataset.

'''
class FILES_and_LABELS():
    def __init__(self, subjects, sessions, MRI_type, functional_type):
        
        self.sessions = sessions
        self.sub = subjects
        
        ID_subjects = []
        for sub in subjects:
            if 0 < int(sub) < 182:
                subj_ID = 'sub-' + str.zfill(str(sub), 3)
                ID_subjects.append(subj_ID)
        self.subjects = ID_subjects

        ID_sessions = []
        for ses in sessions:
            if 0 < int(ses) <= 3:
                ses_ID = 'ses-' + str.zfill(str(ses), 2)
                ID_sessions.append(ses_ID)
        self.sess = ID_sessions
        
        self.MRI_type = MRI_type
        self.functional_type = functional_type
    

    def get_label(self, sess):
        if sess == 'ses-01':
            label = 0
        elif sess == 'ses-02':
            label = 1
        elif sess == 'ses-03':
            label = 2   
        return label
        
    def get_ID_filenames(self):
        #Funcional_type -> rest o dist
        #FMRI_type -> func o anat
        files = []
        label_files = []
        
        rootpath = RAW_ROOT


        for subj in self.subjects:
            for sess in self.sess:
                if self.MRI_type == 'anat':
                    file = subj + '/' + sess + '/anat/' + subj + '_' + sess +  '_T1w.nii'
                elif self.MRI_type == 'func':
                    file = subj + '/' + sess + '/func/' + subj + '_' + sess + '_task-' + self.functional_type + '_bold.nii'
                if os.path.exists(rootpath + file):
                    files.append(file)
                    label = self.get_label(sess)
                    label_files.append(label)
    
        return files, label_files
    
    def get_mask_and_bold(self):  #Build full paths to RABIES common-space BOLD images and masks.
        # files : list of [image_path, mask_path]
        # Funcional_type -> rest o dist
        # FMRI_type -> func o anat
        files = []

        def batch_str(sub_id_int):
            return "001" if sub_id_int <= 68 else "002"

        for i in self.sessions:  # i = sesion (int)
            for j in self.sub:   # j = subject (int)
                b = batch_str(j)

                # Common base for filenames in the RABIES output
                common = (
                    f"_scan_info_subject_id{str(j).zfill(3)}"
                    f".session{str(i).zfill(2)}"
                    f"_split_name_sub-{str(j).zfill(3)}"
                    f"_ses-{str(i).zfill(2)}"
                    f"_desc-o_T2w/_run_None/"
                    f"sub-{str(j).zfill(3)}_ses-{str(i).zfill(2)}"
                )

                if self.functional_type == "rest":
                    mask = (
                        RABIES_ROOT
                        + f"preprocess_batch-{b}_rest/"
                        "bold_datasink/commonspace_mask/"
                        + common
                        + "_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                    )
                    image = (
                        RABIES_ROOT
                        + f"preprocess_batch-{b}_rest/"
                        "bold_datasink/commonspace_bold/"
                        + common
                        + "_task-rest_desc-oa_bold_autobox_combined.nii.gz"
                    )
                elif self.functional_type == "dist":
                    mask = (
                        RABIES_ROOT
                        + f"preprocess_batch-{b}_rest/"
                        "bold_datasink/commonspace_mask/"
                        + common
                        + "_task-rest_desc-oa_bold_autobox_EPI_brain_mask.nii.gz"
                    )
                    image = (
                        RABIES_ROOT
                        + f"preprocess_batch-{b}/"
                        "commonspace_bold/"
                        + common
                        + "_task-dist_desc-oa_bold_autobox_combined.nii.gz"
                    )
                else:
                    continue

                if os.path.exists(image) and os.path.exists(mask):
                    files.append([image, mask])

        return files  

# Subject groups (same logic as in the my_data_generator.py )   
_CPHfemale = tf.constant(
    ['sub-049','sub-050','sub-051','sub-052','sub-065','sub-066',
     'sub-077','sub-078','sub-079','sub-080','sub-081','sub-082','sub-083'],
    dtype=tf.string)
_NAIVEfemale = tf.constant(['sub-019','sub-020','sub-067','sub-068'], dtype=tf.string)

_CPHmale = tf.constant(
    ['sub-057','sub-059','sub-060','sub-073','sub-074',
     'sub-093','sub-094','sub-095','sub-096','sub-098','sub-099','sub-100'],
    dtype=tf.string)
_NAIVEmale = tf.constant(['sub-024','sub-028','sub-075','sub-076'], dtype=tf.string)

# Extrae 'sub-###' y 'ses-##' del path (todo en TF)
def _parse_ids_from_path(path: tf.Tensor):
    sub = tf.strings.regex_replace(
        tf.strings.regex_replace(path, r'.*(sub-\d{3}).*', r'\1'),
        r'^([^s].*)$', 'unknown')
    ses = tf.strings.regex_replace(
        tf.strings.regex_replace(path, r'.*(ses-\d{2}).*', r'\1'),
        r'^([^s].*)$', 'unknown')
    return sub, ses

def _belongs_to(list_str: tf.Tensor, elem: tf.Tensor):
    return tf.reduce_any(tf.equal(list_str, elem))

# task_name options:
#  'bl_vs_w7' - 0: ses-01 (BL), 1: ses-03 (W7)
#  'bl_vs_w1' - 0: (ses-01), 1 = W1 (ses-02)
#  'sessions_3way' - 0: (ses-01), 1: (ses-02), 2: (ses-03)


#Assign an integer label based on subject ID, session and task name.
def _label_from_path_tf(path: tf.Tensor, task_name: tf.Tensor) -> tf.Tensor:
    _, ses = _parse_ids_from_path(path)

    def label_bl_vs_w7():
        # 1 for week 7 (ses-03), 0 otherwise (BL or anything else)
        return tf.where(
            tf.equal(ses, 'ses-03'),
            tf.constant(1, tf.int32),
            tf.constant(0, tf.int32),
        )

    def label_bl_vs_w1():
        # 1 for week 1 (ses-02), 0 otherwise (BL or anything else)
        return tf.where(
            tf.equal(ses, 'ses-02'),
            tf.constant(1, tf.int32),
            tf.constant(0, tf.int32),
        )

    def label_sessions_3way():
        # 0 = BL (ses-01), 1 = W1 (ses-02), 2 = W7 (ses-03), -1 otherwise
        return tf.case(
            [
                (tf.equal(ses, 'ses-01'), lambda: tf.constant(0, tf.int32)),
                (tf.equal(ses, 'ses-02'), lambda: tf.constant(1, tf.int32)),
                (tf.equal(ses, 'ses-03'), lambda: tf.constant(2, tf.int32)),
            ],
            default=lambda: tf.constant(-1, tf.int32),
        )

    # Choose which labelling rule to apply depending on task_name
    return tf.case(
        [
            (tf.equal(task_name, 'bl_vs_w7'),      label_bl_vs_w7),
            (tf.equal(task_name, 'bl_vs_w1'),      label_bl_vs_w1),
            (tf.equal(task_name, 'sessions_3way'), label_sessions_3way),
        ],
        # If task_name is something weird, default to bl_vs_w7
        default=label_bl_vs_w7,
    )

# Numpy-based loading (called from tf.data via tf.numpy_function)
def _load_session_epoch_py(img_path_b, msk_path_b, epoch_np, seed_np,
                           z0, y0, x0, z1, y1, x1, take_vols):
    rng   = np.random.RandomState(int(seed_np) + int(epoch_np))
    img_p = img_path_b.decode("utf-8")
    msk_p = msk_path_b.decode("utf-8")

    img   = nib.load(img_p) # 4D
    mask  = nib.load(msk_p) # 3D
    data4d = img.dataobj
    mask3d = mask.dataobj

    T = int(img.shape[3])
    start = 19 #skip initial volumes
    usable_idx = np.arange(start, T, dtype=int)
    take = min(len(usable_idx), int(take_vols))
    if take <= 0:
        return np.zeros((0, 42, 65, 29), dtype=np.float32)

    idx = rng.choice(usable_idx, size=take, replace=False)
    z0, y0, x0, z1, y1, x1 = int(z0), int(y0), int(x0), int(z1), int(y1), int(x1)

    # crfop the mask only once
    m = np.asarray(mask3d, dtype=np.float32)[z0:z1, y0:y1, x0:x1]
    
    m[m < 0.5] = 0.0
    m[m >= 0.5] = 1.0
    if m.sum() == 0:
        m[:] = 1.0  # fallback if mask is empty (should not happen)

    out = np.empty((take, 42, 65, 29), dtype=np.float32)
    for k, t in enumerate(idx):
        vol = np.asarray(data4d[..., t], dtype=np.float32)[z0:z1, y0:y1, x0:x1]
        vol_masked = vol * m
        mu, sigma = vol_masked.mean(), vol_masked.std() + 1e-8
        out[k] = (vol_masked - mu) / sigma
    return out

def load_session_tf_epoch(pair, epoch_tf, seed_tf, crop_idx6, take_vols:int, task_name: tf.Tensor):
    # pair= (img_path, mask_path)
    img_path, msk_path = pair[0], pair[1]
    z0,y0,x0,z1,y1,x1 = [tf.cast(c, tf.int32) for c in crop_idx6]
    vols = tf.numpy_function(
        _load_session_epoch_py,
        [img_path, msk_path,
         tf.cast(epoch_tf, tf.int32), tf.cast(seed_tf, tf.int32),
         z0,y0,x0,z1,y1,x1, tf.cast(take_vols, tf.int32)],
        tf.float32
    )
    vols.set_shape([None, 42, 65, 29])
   # label derived from the image path and task name
    y_scalar = _label_from_path_tf(img_path, task_name)
    T = tf.shape(vols)[0]
    labels = tf.fill([T], y_scalar)
    return vols, labels

# python loader for a full session
def _load_session_all_time_py(img_path_b, msk_path_b, z0, y0, x0, z1, y1, x1):
    img_p = img_path_b.decode("utf-8")
    msk_p = msk_path_b.decode("utf-8")
    img   = nib.load(img_p)
    mask  = nib.load(msk_p)
    data4d = img.dataobj
    mask3d = mask.dataobj

    T = int(img.shape[3])
    start = 19
    idx = np.arange(start, T, dtype=int)
    z0, y0, x0, z1, y1, x1 = int(z0), int(y0), int(x0), int(z1), int(y1), int(x1)

    m = np.asarray(mask3d, dtype=np.float32)[z0:z1, y0:y1, x0:x1]
    m[m < 0.5] = 0.0
    m[m >= 0.5] = 1.0
    if m.sum() == 0:
        m[:] = 1.0

    out = np.empty((len(idx), 42, 65, 29), dtype=np.float32)
    for k, t in enumerate(idx):
        vol = np.asarray(data4d[..., t], dtype=np.float32)[z0:z1, y0:y1, x0:x1]
        vol_masked = vol * m
        mu, sigma = vol_masked.mean(), vol_masked.std() + 1e-8
        out[k] = (vol_masked - mu) / sigma
    return out

def load_session_tf_all(pair, crop_idx6, task_name: tf.Tensor):
    img_path, msk_path = pair[0], pair[1]
    z0,y0,x0,z1,y1,x1 = [tf.cast(c, tf.int32) for c in crop_idx6]
    vols = tf.numpy_function(
        _load_session_all_time_py,
        [img_path, msk_path, z0,y0,x0,z1,y1,x1],
        tf.float32
    )
    vols.set_shape([None, 42, 65, 29])
    y_scalar = _label_from_path_tf(img_path, task_name)
    T = tf.shape(vols)[0]
    labels = tf.fill([T], y_scalar)
    return vols, labels

def _add_channel(vol, y):  # Add a singleton channel dimension to the volume: (42, 65, 29) to (42, 65, 29, 1)
    return tf.expand_dims(vol, -1), y  # (42,65,29,1), ()

# build tf.data Datasets for training and full runs
# build a tf.data.Dataset for a single training epoch.
def make_epoch_ds(pairs, training: bool, epoch: int, par: int, prefetch_buf: int,
                  seed: int, subbatch: int, vols_per_session_epoch: int,
                  crop_idx6, task_name: str):
    
    #  ds: tf.data.Dataset (batch_volumes, batch_labels) 
    #  batch_volumes has shape (subbatch, 42, 65, 29, 1).
    task_tf = tf.convert_to_tensor(task_name, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(pairs)  # each item: (img_path, mask_path)
    if training:
        ds = ds.shuffle(buffer_size=min(64, len(pairs)), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda pair: load_session_tf_epoch(
            pair,
            tf.constant(epoch, tf.int32),
            tf.constant(seed, tf.int32),
            crop_idx6,
            tf.constant(vols_per_session_epoch, tf.int32),
            task_tf
        ),
        num_parallel_calls=par,
        deterministic=not training
    )
    #convert (session, T_in_session) into a flat stream of volumes
    ds = ds.unbatch()
    ds = ds.map(_add_channel, num_parallel_calls=par, deterministic=not training)
    ds = ds.batch(subbatch, drop_remainder=True)
    ds = ds.prefetch(prefetch_buf)
    return ds

# build a tf.data.Dataset with all available volumes from the given sessions.

def make_full_ds(pairs, subbatch: int, crop_idx6, task_name: str,
                 par: int = 1, prefetch_buf: int = 1):
    task_tf = tf.convert_to_tensor(task_name, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(pairs)
    ds = ds.map(lambda pair: load_session_tf_all(pair, crop_idx6, task_tf),
                num_parallel_calls=par, deterministic=True)
    ds = ds.unbatch()
    ds = ds.map(_add_channel, num_parallel_calls=par, deterministic=True)
    ds = ds.batch(subbatch, drop_remainder=False)
    ds = ds.prefetch(prefetch_buf)
    return ds
