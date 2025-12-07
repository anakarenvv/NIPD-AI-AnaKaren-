
#Original keras data generator 

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

        #rootpath = "C:/Users/"+os.getlogin()+"/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/rawdata/"
        #label_table = pd.read_table(rootpath + '/participants.tsv', encoding='utf-8')
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


    
class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, df,
                 batch_size,
                 subbatch_size,
                 input_size=(100, 128, 128, 22),
                 shuffle=True,
                 format = "rgb",
                 classes = None,
                 num_class = None,
                 vols = 600,
                 augmentation = False,
                 functional_type = "rest"):
        #df es una lista con los paths de las sesiones despues del "rawdata/"
        
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.path = RAW_ROOT

        #if os.getlogin() == "damia":
        #    self.path = "E:/rawdata/"
        #else:
        #    self.path =  "C:/Users/"+os.getlogin()+"/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/rawdata/"
        self.n = len(self.df)
        self.format = format
        self.vols = vols
        self.num_class = num_class
        self.classes = classes
        self.augment = augmentation
        self.subbatch_size = subbatch_size
        self.contador = 0
        self.X = []
        self.y = []
        self.sess_per_batch = []
        self.functional_type = functional_type
        
        if ((self.batch_size*self.vols)%self.subbatch_size) != 0:
            print("El tamaño del batch, subbatch y vols no permiten particiones exactas\n Escogerlos tal que batch_size*vols%subbatch_size = 0")
        
      
    def flip_augmentation(self, random_num, vols_batch):
        if random_num == 1:
            vols_batch = np.flip(vols_batch,axis=(3))
            
        if random_num == 2:
            vols_batch = np.flip(vols_batch,axis=(2))
            
        if random_num == 3:
            vols_batch = np.flip(vols_batch,axis=(1))
            
        if random_num == 4:
            vols_batch = np.flip(vols_batch,axis=(1,2))
            
        if random_num == 5:
            vols_batch = np.flip(vols_batch,axis=(1,3))
        
        if random_num == 6:
            vols_batch = np.flip(vols_batch,axis=(2,3))
        
        if random_num == 7:
            vols_batch = np.flip(vols_batch,axis=(1,2,3))
            
        return vols_batch
    
    def on_epoch_end(self):
        #self.contador = 0
        
        CPHfemale = ['sub-049','sub-050','sub-051','sub-052','sub-065','sub-066','sub-077','sub-078','sub-079','sub-080','sub-081',
               'sub-082','sub-083']
        NAIVEfemale = ['sub-019','sub-020','sub-067','sub-068']
        
        CPHmale = ['sub-057','sub-059','sub-060','sub-073','sub-074','sub-093','sub-094','sub-095','sub-096','sub-098','sub-099',
                   'sub-100']
        NAIVEmale = ['sub-024','sub-028','sub-075','sub-076']
        
        
        def interleave_diff_len_lists(list1, list2):
            # Create an empty list called 'result' to store the interleaved elements.
            result = []
            
            # Get the lengths of the input lists and store them in 'l1', 'l2', 'l3', and 'l4'.
            l1 = len(list1)
            l2 = len(list2)
            
            # Iterate from 0 to the maximum length among the input lists using 'max' function.
            for i in range(max(l1, l2)):
                # Check if 'i' is less than 'l1' and add the element from 'list1' to 'result'.
                if i < l1:
                    result.append(list1[i])
                # Check if 'i' is less than 'l2' and add the element from 'list2' to 'result'.
                if i < l2:
                    result.append(list2[i])
            # Return the 'result' list containing interleaved elements from the input lists.
            return result
        
        if self.shuffle:
            c0=[]
            c1=[]
            
            if self.format == "just_brain" or self.format=="just_brain_M2D" or self.format=="just_brain_vol":
                if self.classes == "CPHvsNAIVEfemale":
                    for i in range(len(self.df)):
                        for j in CPHfemale:
                            if j in self.df[i][0]:
                                if 'ses-02' in self.df[i][0] or 'ses-03' in self.df[i][0]:
                                    c1.append(self.df[i])
                                else:
                                    c0.append(self.df[i])
                        
                        for k in NAIVEfemale:
                            if k in self.df[i][0]:
                                c0.append(self.df[i])
                                
                if self.classes == "CPHvsNAIVEmale":
                    for i in range(len(self.df)):
                        for j in CPHmale:
                            if j in self.df[i][0]:
                                if 'ses-02' in self.df[i][0] or 'ses-03' in self.df[i][0]: 
                                    c1.append(self.df[i])
                                else:
                                    c0.append(self.df[i])
                        
                        for k in NAIVEmale:
                            if k in self.df[i][0]:
                                c0.append(self.df[i])
                                
                if self.classes == "sex":
                    for i in range(len(self.df)):
                        for j in CPHfemale:
                            if j in self.df[i][0]:
                                c1.append(self.df[i])
                        for k in CPHmale:
                            if k in self.df[i][0]:
                                c0.append(self.df[i])
                                
                if self.classes == "W1vsW7":
                    for i in range(len(self.df)):
                        if 'ses-02' in self.df[i][0]:
                            c1.append(self.df[i])
                        elif 'ses-03' in self.df[i][0]:
                            c0.append(self.df[i])
                            
                
                
            indices0 = np.arange(len(c0),dtype=int)
            indices1 = np.arange(len(c1),dtype=int)
            
            np.random.shuffle(indices0)
            np.random.shuffle(indices1)
            
            self.df = interleave_diff_len_lists(np.array(c0)[indices0].tolist(),np.array(c1)[indices1].tolist())
            #print(self.df)
                
    
    def __get_input(self, file):
        if self.functional_type == "dist":
            if self.format == "just_brain":
                #self.vols = 135
                
                img = nib.load(file[0])
                mask = nib.load(file[1])
                
                data = img.dataobj
                data = np.transpose(data, (3,0,1,2))
                
                #extract dist volumes
                task_time = []
                for i in range(15):
                    task_time.append(45)
                    task_time.append(15)
                schedule = []
                t = 0
                for i in range(len(task_time)): 
                    t += task_time[i]
                    schedule.append(t)
                final = [] #final dist vol per cycle
                initial=[] #initial dist vol per cycle
                for i in range(len(schedule)):
                    if i%2==0:
                        #print(i)
                        initial.append(int(np.ceil(schedule[i]/1.51069)+1))
                    else:
                        final.append(int(schedule[i]/1.51069))
                
                vol = []
                for i in range(len(initial)):
                    vol.extend(data[initial[i]:final[i]+1])
                   
                vol = np.array(vol)
                a = np.linspace(0, 134, num=self.vols, endpoint=True, dtype=int)
                vol = vol[a]
                
                #print("shape of vol", np.shape(vol))
                #data = img.get_fdata()
                maskdata = mask.dataobj
                
                #print("shape mask", np.shape(maskdata))
                
                image_arr = maskdata*np.array(vol)
                image_arr = image_arr[:,3:45,4:69,7:36]
                
                #Z-scoring
                #image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
                
                return np.array(image_arr)
            
            elif self.format == "just_brain_M2D" or self.format == "just_brain_vol":
                #self.vols = 135
                
                img = nib.load(file[0])
                mask = nib.load(file[1])
                
                data = img.get_fdata()
                maskdata = mask.get_fdata()
                
                data = np.transpose(data, (3,0,1,2))
                
                #extract dist volumes
                task_time = []
                for i in range(15):
                    task_time.append(45)
                    task_time.append(15)
                schedule = []
                t = 0
                for i in range(len(task_time)): 
                    t += task_time[i]
                    schedule.append(t)
                final = [] #final dist vol per cycle
                initial=[] #initial dist vol per cycle
                for i in range(len(schedule)):
                    if i%2==0:
                        #print(i)
                        initial.append(int(np.ceil(schedule[i]/1.51069)+1))
                    else:
                        final.append(int(schedule[i]/1.51069))
                
                vol = []
                for i in range(len(initial)):
                    vol.extend(data[initial[i]:final[i]+1])
                    
                vol = np.array(vol)
                a = np.linspace(0, 134, num=self.vols, endpoint=True, dtype=int)
                vol = vol[a]
                
                image_arr = maskdata*np.array(vol)
                #VGG16 needs at least 32x32 images. that´s why the crop when ussing vgg16 2md is bigger
                image_arr = image_arr[:,3:45,4:69,6:38]
                
                #Z-scoring
                #image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
                #histogram equalization
                """
                hist = scipy.ndimage.histogram(image_arr, min = 0,print
                                 max = 255,
                                 bins =256)
                cdf = hist.cumsum()/hist.sum()
                image_arr = cdf[np.array(image_arr,dtype="uint8")] * 255
                """
                
                return np.array(image_arr)
                
        if self.functional_type == "rest":
            
            if self.format != "just_brain" and self.format != "just_brain_M2D" and self.format != "just_brain_vol":
                #temp = nib.load("E:/rawdata/"+file)
                temp = nib.load(self.path+file)
                image_arr = []
                #image_arr = temp.dataobj[:,:,:,:self.vols]
                for k in range(20,int(600/self.vols)*self.vols,int(580/self.vols)):
                    image_arr.append(temp.dataobj[:,:,:,k])
    
            if self.format == "rgb":
                image_arr = tf.transpose(image_arr, (0,3,1,2))
                image_arr = np.reshape(image_arr, (-1,128,128,1))
                #Z-scoring
                image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
                X_rgb = []
                for i in range(len(image_arr)):
                    X_rgb.append(cv2.cvtColor(image_arr[i].astype(np.uint8),cv2.COLOR_GRAY2RGB))
    
                X_rgb = np.array(X_rgb).reshape([-1, 128, 128, 3, 1])
                return np.array(X_rgb)
    
            elif self.format == "vol":
                #image_arr = tf.transpose(image_arr, (3,0,1,2))
                image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
                return np.array(image_arr)
                
    
            elif self.format == "grayscale":
                #image_arr = tf.transpose(image_arr, (3,2,0,1))
                image_arr = tf.transpose(image_arr, (0,3,1,2))
                image_arr = np.reshape(image_arr, (-1,128,128,1))
                return np.array(image_arr)
            
            elif self.format == "just_brain":
                img = nib.load(file[0])
                mask = nib.load(file[1])
                
                data = img.dataobj
                data = np.transpose(data, (3,0,1,2))
                
                a = np.linspace(19, 600, num=self.vols, endpoint=True, dtype=int)
                data = data[a]
                
                #data = img.get_fdata()
                maskdata = mask.dataobj
                
                image_arr = maskdata*data
                image_arr = image_arr[:,3:45,4:69,7:36]
                
                #Z-scoring
                #image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
                """
                #histogram equalization
                hist = scipy.ndimage.histogram(image_arr, min = 0,
                                 max = 255,
                                 bins =256)
                cdf = hist.cumsum()/hist.sum()
                image_arr = cdf[np.array(image_arr,dtype="uint8")] * 255
                """
                return np.array(image_arr)
            
            elif self.format == "M2D":
                #Z-scoring
                image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
                #histogram equalization
                hist = scipy.ndimage.histogram(image_arr, min = 0,
                                 max = 255,
                                 bins =256)
                cdf = hist.cumsum()/hist.sum()
                image_arr = cdf[np.array(image_arr,dtype="uint8")] * 255
                
                return np.array(image_arr)
            
            elif self.format == "M2D_VGG16":
                #Z-scoring
                image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
                image_arr = skimage.transform.resize(image_arr, (self.vols,128, 128, 32))
                
                return np.array(image_arr)
            
            elif self.format == "just_brain_M2D" or self.format == "just_brain_vol":
                
                img = nib.load(file[0])
                mask = nib.load(file[1])
                
                data = img.get_fdata()
                maskdata = mask.get_fdata()
                
                data = np.transpose(data, (3,0,1,2))
                a = np.linspace(19, 600, num=self.vols, endpoint=True, dtype=int)
                data = data[a]
                
                image_arr = maskdata*data
                #VGG16 needs at least 32x32 images. that´s why the crop when ussing vgg16 2md is bigger
                image_arr = image_arr[:,3:45,4:69,6:38]
                
                #Z-scoring
                #image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
                #histogram equalization
                """
                hist = scipy.ndimage.histogram(image_arr, min = 0,
                                 max = 255,
                                 bins =256)
                cdf = hist.cumsum()/hist.sum()
                image_arr = cdf[np.array(image_arr,dtype="uint8")] * 255
                """
                
                return np.array(image_arr)
            
            elif self.format == "just_brain_flatten":
                img = nib.load(file[0])
                mask = nib.load(file[1])
                
                data = img.get_fdata()
                maskdata = mask.get_fdata()
                
                data = np.transpose(data, (3,0,1,2))
                a = np.linspace(19, 600, num=self.vols, endpoint=True, dtype=int)
                data = data[a]
                
                image_arr = maskdata*data
                image_arr = image_arr[:,3:45,4:69,7:36]
                
                #Z-scoring
                image_arr = (image_arr - np.mean(image_arr))/np.std(image_arr)
                #histogram equalization
                hist = scipy.ndimage.histogram(image_arr, min = 0,
                                 max = 255,
                                 bins =256)
                cdf = hist.cumsum()/hist.sum()
                image_arr = cdf[np.array(image_arr,dtype="uint8")] * 255
                
                image_arr.flatten()
                
                return np.array(image_arr)
        
    def __get_output(self, file):
        male = ['sub-057',
                 'sub-059',
                 'sub-060',
                 'sub-073',
                 'sub-074',
                 'sub-093',
                 'sub-094',
                 'sub-095',
                 'sub-096',
                 #'sub-097',
                 'sub-098',
                 'sub-099',
                 'sub-100']
        female = ['sub-049',
                 'sub-050',
                 'sub-051',
                 'sub-052',
                 'sub-065',
                 'sub-066',
                 'sub-077',
                 'sub-078',
                 'sub-079',
                 'sub-080',
                 'sub-081',
                 'sub-082',
                 'sub-083',
                 #'sub-084',
                 ]
        CPHfemale = ['sub-049','sub-050','sub-051','sub-052','sub-065','sub-066','sub-077','sub-078','sub-079','sub-080','sub-081',
               'sub-082','sub-083']
        NAIVEfemale = ['sub-019','sub-020','sub-067','sub-068']
        
        CPHmale = ['sub-057','sub-059','sub-060','sub-073','sub-074','sub-093','sub-094','sub-095','sub-096','sub-098','sub-099',
                   'sub-100']
        NAIVEmale = ['sub-024','sub-028','sub-075','sub-076']
        
        label = []
        if self.classes == "CPHvsNAIVEfemale":
            if self.format == "just_brain":
                
                for j in NAIVEfemale:
                    if j in file[0]:
                        label.append(0)
                        
                for i in CPHfemale:
                    if i in file[0]:
                        if 'ses-02' in file[0] or 'ses-03' in file[0]:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    print("El siguiente sujeto genera conflicto con las etiquetas\n",file[0])
                
                
            elif self.format == "just_brain_M2D" or self.format == "just_brain_vol":
                for j in NAIVEfemale:
                    if j in file[0]:
                        label.append(0)
                        
                for i in CPHfemale:
                    if i in file[0]:
                        if 'ses-02' in file[0] or 'ses-03' in file[0]:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    print("El siguiente sujeto genera conflicto con las etiquetas\n",file[0])
            else:    
                for j in NAIVEfemale:
                    if j in file[0]:
                        label.append(0)
                        
                for i in CPHfemale:
                    if i in file[0]:
                        if 'ses-02' in file[0] or 'ses-03' in file[0]:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    print("El siguiente sujeto genera conflicto con las etiquetas\n",file[0])
                    
        """Para el siguiente caso, las etiquetas estan invertidas (CPH = 1, Naive = 0)"""
        if self.classes == "CPHvsNAIVEmale":
            if self.format == "just_brain":
                for j in NAIVEmale:
                    if j in file[0]:
                        label.append(0)
                for i in CPHmale:
                    if i in file[0]:
                        if 'ses-02' in file[0] or 'ses-03' in file[0]:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    print("El siguiente sujeto genera conflicto con las etiquetas\n",file[0])
                
            elif self.format == "just_brain_M2D" or self.format == "just_brain_vol":
                for j in NAIVEmale:
                    if j in file[0]:
                        label.append(0)
                for i in CPHmale:
                    if i in file[0]:
                        if 'ses-02' in file[0] or 'ses-03' in file[0]:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    print("El siguiente sujeto genera conflicto con las etiquetas\n",file[0])
                
            else:    
                for j in NAIVEmale:
                    if j in file:
                        label.append(0)
                for i in CPHmale:
                    if i in file:
                        if 'ses-02' in file or 'ses-03' in file:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    print("El siguiente sujeto genera conflicto con las etiquetas\n",file)
                
                    
        if self.classes == "sessions":
            if self.format == "just_brain":
                if 'ses-01' in file[0]:
                    label.append(0)
                elif 'ses-02' in file[0]:
                    label.append(1)
                elif 'ses-03' in file[0]:
                    label.append(2)
                    
            elif self.format == "just_brain_M2D" or self.format == "just_brain_vol":
                if 'ses-01' in file[0]:
                    label.append(0)
                elif 'ses-02' in file[0]:
                    label.append(1)
                elif 'ses-03' in file[0]:
                    label.append(2)
            else:    
                if 'ses-01' in file:
                    label.append(0)
                elif 'ses-02' in file:
                    label.append(1)
                elif 'ses-03' in file:
                    label.append(2)
                else:
                    label.append("algo salio mal")
                
        if self.classes == "sex":
            if self.format == "just_brain":
                for i in male:
                    if i in file[0]:
                        label.append(0)
                for j in female:
                    if j in file[0]:
                        label.append(1)
                        
            elif self.format == "just_brain_M2D" or self.format == "just_brain_vol":
                for i in male:
                    if i in file[0]:
                        label.append(0)
                for j in female:
                    if j in file[0]:
                        label.append(1)
                 
            else:
                for i in male:
                    if i in file:
                        label.append(0)
                for j in female:
                    if j in file:
                        label.append(1)
        
        if self.classes == "W1vsW7":
            if self.format == "just_brain":
                if 'ses-02' in file[0]:
                    label.append(0)
                elif 'ses-03' in file[0]:
                    label.append(1)
                    
        if self.format == "vol":
            label = label*self.vols

        
        elif self.format == "rgb":
            label = label*self.vols*22
        
        elif self.format == "grayscale":
            label = label*self.vols*22
        
        elif self.format == "just_brain":
            label = label*self.vols
        
        elif self.format == "M2D":
            label = label*self.vols
        
        elif self.format == "M2D_VGG16":
            label = label*self.vols
        
        elif self.format == "just_brain_M2D" or self.format == "just_brain_vol":
            label = label*self.vols
        return tf.keras.utils.to_categorical(label, num_classes=self.num_class)
        #return label
        

    def __get_data(self, batches):
        # Generates data containing batch_size samples
        path_batch = batches

        y_batch = np.asarray([self.__get_output(y) for y in path_batch])
        y_batch = np.reshape(y_batch, (-1,self.num_class))
        #y_batch = np.reshape(y_batch, (-1))
        
        """eliminar siguientes 2 linea"""
        file = np.asarray([[f]*self.vols for f in path_batch])
        file = np.reshape(file, (-1))
        
        
        indices = np.arange(len(y_batch),dtype=int)
        np.random.shuffle(indices)
        
        if self.shuffle == True:
            y_batch = np.array(y_batch)[indices]
            
        """eliminar siguiente linea"""
        file = np.array(file)[indices]
        
        
        if self.format == "rgb": 
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,3))
            
            X_batch = np.array(X_batch)[indices]
            
            return X_batch, y_batch
            
        elif self.format == "vol":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,22))
            
            X_batch = np.array(X_batch)[indices]
    
            return X_batch, y_batch
            
        elif self.format == "grayscale":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,1))
            
            X_batch = np.array(X_batch)[indices]
            
            return X_batch, y_batch
            
        elif self.format == "just_brain":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,42, 65, 29))
            
            if self.shuffle == True:
                X_batch = np.array(X_batch)[indices]
            
            return X_batch, y_batch
            
        elif self.format == "M2D":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,22))
            
            X_batch = np.array(X_batch)[indices]
            
            image_arr_left = np.transpose(X_batch, (0, 2, 3, 1))
            image_arr_front = np.transpose(X_batch, (0, 1, 3, 2))
            
            return [X_batch,image_arr_front,image_arr_left], y_batch
        
        elif self.format == "M2D_VGG16":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,128,128,32))
            
            X_batch = np.array(X_batch)[indices]
            
            image_arr_left = np.transpose(X_batch, (0, 2, 3, 1))
            image_arr_front = np.transpose(X_batch, (0, 1, 3, 2))
            
            return [X_batch,image_arr_front,image_arr_left], y_batch
        
        elif self.format == "just_brain_M2D":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
             
            X_batch = np.reshape(X_batch, (-1,42, 65, 32))
            
            if self.shuffle == True:
                X_batch = np.array(X_batch)[indices]
            
            image_arr_left = np.transpose(X_batch, (0, 2, 3, 1))
            image_arr_front = np.transpose(X_batch, (0, 1, 3, 2))
            
            
            return [X_batch,image_arr_front,image_arr_left], y_batch
        
        elif self.format == "just_brain_vol":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (-1,42, 65, 32))
            
            X_batch = np.array(X_batch)[indices]
            
            return X_batch, y_batch
        
        elif self.format == "just_brain_flatten":
            X_batch = np.asarray([self.__get_input(x) for x in path_batch])
            X_batch = np.reshape(X_batch, (len(path_batch),-1))
            
            X_batch = np.array(X_batch)[indices]
            
            return X_batch, y_batch
    
        
        
        #return np.array(X_batch, dtype='uint8'), y_batch
        

        
        
    """   
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        
        X, y= self.__get_data(batches)   
        
        #indices = np.arange(len(y),dtype=int)
        #np.random.shuffle(indices)
        if self.augment == True:
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                ])
            X = data_augmentation(X)
        
        return X, y""" 
    
    def __getitem__(self,subbatch_index):   #Prueba 
        """Escoger un subbatch tal que vols%subbatch_size = 0"""
        batch_index = int(subbatch_index/(self.batch_size*(self.vols/self.subbatch_size)))
        batch = self.df[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        
        if batch != self.sess_per_batch:
            self.sess_per_batch = self.df[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            self.X, self.y= self.__get_data(self.sess_per_batch)
            
            if self.format == "just_brain_M2D" or self.format == "just_brain_vol":
                #Z-scoring
                mean_x = np.mean(self.X[0])
                std_x = np.std(self.X[0])
                
                self.X[0] = (self.X[0] - mean_x)/std_x
                self.X[1] = (self.X[1] - mean_x)/std_x
                self.X[2] = (self.X[2] - mean_x)/std_x
            else:
                self.X = (self.X - np.mean(self.X))/np.std(self.X)

            
        #l = slices del batch completo
        l = int(subbatch_index-(batch_index*(((self.batch_size*self.vols))/self.subbatch_size)))
        
        if self.format == "just_brain_M2D" or self.format == "M2D" or self.format == "M2D_VGG16":
            up = self.X[0][l*self.subbatch_size:(l+1)*self.subbatch_size]
            front = self.X[1][l*self.subbatch_size:(l+1)*self.subbatch_size]
            left = self.X[2][l*self.subbatch_size:(l+1)*self.subbatch_size]
            
            
            if self.augment == True:
                if self.functional_type == "dist":
                    up_aug = []
                    front_aug = []
                    left_aug = []
                    
                    y_aug= []
                    
                    aug = [1,2,3,4,5,6,7]
                    
                    for i in range(4):
                        random_num = np.random.choice(aug)
                        up_aug.extend(self.flip_augmentation(random_num, up))
                        front_aug.extend(self.flip_augmentation(random_num, front))
                        left_aug.extend(self.flip_augmentation(random_num, left))
                        
                        #up_aug = np.array(up_aug)
                        #front_aug = np.array(front_aug)
                        #left_aug = np.array(left_aug)
                        
                        y_aug.extend(self.y[l*self.subbatch_size:(l+1)*self.subbatch_size])
                        
                    return [up_aug,front_aug,left_aug], y_aug
                        
                elif self.functional_type == "rest":
                    data_augmentation = tf.keras.Sequential([
                        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                        #tf.keras.layers.RandomRotation(0.2),
                        ])
                    up = data_augmentation(up)
                    front = data_augmentation(front)
                    left = data_augmentation(left)
            
                    return [up,front,left], self.y[l*self.subbatch_size:(l+1)*self.subbatch_size]
                
            elif self.augment == False:
                return [up,front,left], self.y[l*self.subbatch_size:(l+1)*self.subbatch_size]
            
        else:
            vols_batch = self.X[l*self.subbatch_size:(l+1)*self.subbatch_size]
            
            if self.augment == True:
                aug = [1,2,3,4,5,6,7]
                if self.functional_type == "rest":
                    random_num = np.random.choice(aug)
                    vols_batch = self.flip_augmentation(random_num, vols_batch)
                    
                    return vols_batch, self.y[l*self.subbatch_size:(l+1)*self.subbatch_size]
 
                elif self.functional_type == "dist":
                    aug = [1,2,3,4,5,6,7]
                    aug_x = []
                    y_aug = []
                    for i in range(4):
                        random_num = np.random.choice(aug)
                        aug_x.extend(self.flip_augmentation(random_num, vols_batch))
                        y_aug.extend(self.y[l*self.subbatch_size:(l+1)*self.subbatch_size])
                        
                    return np.array(aug_x), np.array(y_aug) 
                
            elif self.augment == False:    
                return vols_batch, self.y[l*self.subbatch_size:(l+1)*self.subbatch_size]
            
    def __len__(self):    #PRUEBA
        return (self.n*self.vols) // self.subbatch_size
    """
    def __len__(self):
        return self.n // self.batch_size
    """
    
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate class activation heatmap"""
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        #print("label predicted: ",pred_index)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel (equivalent to global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    # We multiply each channel in the feature map array
    # by 'how important this channel is' with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # Notice that we clip the heatmap values, which is equivalent to applying ReLU
    heatmap = tf.math.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_resized_heatmap(heatmap, shape):
    """Resize heatmap to shape"""
    # Rescale heatmap to a range 0-255
    upscaled_heatmap = np.uint8(255 * heatmap)

    upscaled_heatmap = zoom(
        upscaled_heatmap,
        (
            shape[0] / upscaled_heatmap.shape[0],
            shape[1] / upscaled_heatmap.shape[1],
            shape[2] / upscaled_heatmap.shape[2],
        ),
    )

    return upscaled_heatmap

def get_bounding_boxes(heatmap, threshold=0.15, otsu=False):
    """Get bounding boxes from heatmap"""
    p_heatmap = np.copy(heatmap)

    if otsu:
        # Otsu's thresholding method to find the bounding boxes
        threshold, p_heatmap = cv2.threshold(
            heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        # Using a fixed threshold
        p_heatmap[p_heatmap < threshold * 255] = 0
        p_heatmap[p_heatmap >= threshold * 255] = 1

    # find the contours in the thresholded heatmap
    contours = cv2.findContours(p_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # get the bounding boxes from the contours
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append([x, y, x + w, y + h])

    return bboxes


def get_bbox_patches(bboxes, color='r', linewidth=2):
    """Get patches for bounding boxes"""
    patches = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        patches.append(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor=color,
                facecolor='none',
                linewidth=linewidth,
            )
        )
    return patches

def _draw_line(ax, coords, clr='g'):
    line = Path(coords, [Path.MOVETO, Path.LINETO])
    pp = PathPatch(line, linewidth=3, edgecolor=clr, facecolor='none')
    ax.add_patch(pp)


def _set_axes_labels(ax, axes_x, axes_y):
    ax.set_xlabel(axes_x)
    ax.set_ylabel(axes_y)
    ax.set_aspect('equal', 'box')


def _draw_bboxes(ax, heatmap):
    bboxes = get_bounding_boxes(heatmap, otsu=True)
    patches = get_bbox_patches(bboxes)
    for patch in patches:
        ax.add_patch(patch)





def show_volume(vol, z, y, x, heatmap=None, alpha=0.3, fig_size=(6, 6)):
    _rec_prop = dict(linewidth=5, facecolor='none')
    """Show a slice of a volume with optional heatmap"""
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=fig_size)
    v_z, v_y, v_x = vol.shape

    img0 = axarr[0, 0].imshow(vol[z, :, :], cmap='bone')
    if heatmap is not None:
        axarr[0, 0].imshow(
            heatmap[z, :, :], cmap='jet', alpha=alpha, extent=img0.get_extent()
        )
        _draw_bboxes(axarr[0, 0], heatmap[z, :, :])

    axarr[0, 0].add_patch(Rectangle((-1, -1), v_x, v_y, edgecolor='r', **_rec_prop))
    _draw_line(axarr[0, 0], [(x, 0), (x, v_y)], 'g')
    _draw_line(axarr[0, 0], [(0, y), (v_x, y)], 'b')
    _set_axes_labels(axarr[0, 0], 'X', 'Y')

    img1 = axarr[0, 1].imshow(vol[:, :, x].T, cmap='bone')
    if heatmap is not None:
        axarr[0, 1].imshow(
            heatmap[:, :, x].T, cmap='jet', alpha=alpha, extent=img1.get_extent()
        )
        _draw_bboxes(axarr[0, 1], heatmap[:, :, x].T)

    axarr[0, 1].add_patch(Rectangle((-1, -1), v_z, v_y, edgecolor='g', **_rec_prop))
    _draw_line(axarr[0, 1], [(z, 0), (z, v_y)], 'r')
    _draw_line(axarr[0, 1], [(0, y), (v_x, y)], "b")
    _set_axes_labels(axarr[0, 1], 'Z', 'Y')

    img2 = axarr[1, 0].imshow(vol[:, y, :], cmap='bone')
    if heatmap is not None:
        axarr[1, 0].imshow(
            heatmap[:, y, :], cmap='jet', alpha=alpha, extent=img2.get_extent()
        )
        _draw_bboxes(axarr[1, 0], heatmap[:, y, :])

    axarr[1, 0].add_patch(Rectangle((-1, -1), v_x, v_z, edgecolor='b', **_rec_prop))
    _draw_line(axarr[1, 0], [(0, z), (v_x, z)], 'r')
    _draw_line(axarr[1, 0], [(x, 0), (x, v_y)], 'g')
    _set_axes_labels(axarr[1, 0], 'X', 'Z')
    axarr[1, 1].set_axis_off()
    fig.tight_layout()


def interactive_show(volume, heatmap=None):
    """Show a volume interactively"""
    # transpose volume from (x, y, z) to (z, y, x)
    volume = np.transpose(volume, (2, 0, 1))
    if heatmap is not None:
        heatmap = np.transpose(heatmap, (2, 0, 1))
    vol_shape = volume.shape

    interact(
        lambda x, y, z: plt.show(show_volume(volume, z, y, x, heatmap)),
        z=IntSlider(min=0, max=vol_shape[0] - 1, step=1, value=int(vol_shape[0] / 2)),
        y=IntSlider(min=0, max=vol_shape[1] - 1, step=1, value=int(vol_shape[1] / 2)),
        x=IntSlider(min=0, max=vol_shape[2] - 1, step=1, value=int(vol_shape[2] / 2)),
    )


def create_animation(array, case, heatmap=None, alpha=0.3):
    """Create an animation of a volume"""
    array = np.rot90(np.transpose(array, (1, 0, 2)),axes=(1,2))
    if heatmap is not None:
        heatmap = np.rot90(np.transpose(heatmap, (1, 0, 2)),axes=(1,2))
    fig = plt.figure(figsize=(4, 4))
    images = []
    for idx, image in enumerate(array):
        # plot image without notifying animation
        image_plot = plt.imshow(image, animated=True, cmap='bone')
        aux = [image_plot]
        if heatmap is not None:
            image_plot2 = plt.imshow(
                heatmap[idx], animated=True, cmap='jet', alpha=alpha,vmin=0,vmax=255, extent=image_plot.get_extent())
            aux.append(image_plot2)

            # add bounding boxes to the heatmap image as animated patches
            bboxes = get_bounding_boxes(heatmap[idx])
            patches = get_bbox_patches(bboxes)
            aux.extend(image_plot2.axes.add_patch(patch) for patch in patches)
        images.append(aux)

    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.title(f'Patient ID: {case}', fontsize=16)
    ani = animation.ArtistAnimation(
        fig, images, interval=5000//len(array), blit=False, repeat_delay=1000)
    plt.close()
    return ani

def mislabeled_subj(y_test, preds, subjects, vols_per_sess_ts):
    index_mislabeled = []
    for i in range(len(preds)):
        if preds[i] != y_test[i]:
            index_mislabeled.append(i) 
            
    mislabeled_in = np.ceil((np.array(index_mislabeled)+1)/vols_per_sess_ts)-1

    mis_subj = []
    for i in mislabeled_in:
        mis_subj.append(subjects[int(i)][0])

    contador = Counter(mis_subj)

    return contador

def index_for_gradcam(label, y_test, preds):
    for i in range(len(y_test)):
        if y_test[i] == label:
            if y_test[i] == preds[i]:
                return i

    
def fuse_layers(layers, model, x_vols, index_subj, emphasize=False):
    '''
    Fuses grad-cam heatmaps from a list of model layers into a single heatmap
    and superimposes the heatmap onto an image.

    Args:
      layers: list of strings
      model: tf model
      img: (img_width x img_height x 3) numpy array


    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
      '''
    #index_subj = index_for_gradcam(class_,y_test,preds)
    
    cams = []
    for layer in layers:
        cam = make_gradcam_heatmap(np.expand_dims(x_vols[index_subj], axis=0), model, layer)
        
        cam = get_resized_heatmap(cam, np.shape(x_vols[index_subj]))
        #print(cam.max(),cam.min())
        cams.append(cam)
        #print(np.shape(cam))

    fused = np.mean(cams, axis=0)
    fused = np.uint8(fused)
    #print(np.shape(fused))
    #superimposed = create_animation(x_vols[index_subj], 'All layers GradCam', heatmap=fused)

    return fused

def grad_cam_per_frames(vol,gradcam, threshold):
    fig = plt.figure(figsize=(13, 8))
    for i in range(len(vol[0,:,0])):
        fig.add_subplot(9, 8, i+1)
        # show the upsampled image
        plt.imshow(cv2.resize(np.rot90(np.array(vol)[:,i,:]),dsize=(3*(np.array(vol).shape[0]),3*(np.array(vol).shape[2]))), alpha=0.8, cmap='bone')
    
        # over the cam output
        plt.imshow(cv2.resize(np.rot90(gradcam[:,i,:]*np.ma.masked_greater(gradcam[:,i,:],threshold).mask),dsize=(3*(np.array(vol).shape[0]),3*(np.array(vol).shape[2]))),vmin = 0, vmax=255, alpha=0.4,cmap='jet')
        
        plt.axis('off')
        # display the image
    plt.show()
    
    return fig

"""conflicto
def accuracy_loss_rate(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras
"""

'''
class CombineCallback(tf.keras.callbacks.Callback):

    def __init__(self, **kargs):
        super(CombineCallback, self).__init__(**kargs)

    def on_epoch_end(self, epoch, logs={}): 
        logs['combine_metric'] = logs['val_Accuracy']/logs['val_loss']
'''

class CombineCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()  # sin kwargs

    def on_epoch_end(self, epoch, logs=None):
        import numpy as np
        logs = logs or {}
        acc = logs.get('val_Accuracy') or logs.get('val_accuracy')
        loss = logs.get('val_loss')

        if (acc is not None) and (loss is not None) and np.isfinite(loss) and (loss != 0):
            logs['combine_metric'] = float(acc) / float(loss)
        elif acc is not None:
            logs['combine_metric'] = float(acc)
        elif loss is not None:
            logs['combine_metric'] = -float(loss)
        else:
            logs['combine_metric'] = 0.0


"""FUNCIONES PARA PYTORCH"""

class CustomBatchSampler(Sampler):
    r"""Yield a mini-batch of indices. The sampler will drop the last batch of
            an image size bin if it is not equal to ``batch_size``

    Args:
        examples (dict): List from dataset class.
        batch_size (int): Size of mini-batch.
    """
    def __interleave_diff_len_lists(self,list1, list2):
            # Create an empty list called 'result' to store the interleaved elements.
            result = []
            
            # Get the lengths of the input lists and store them in 'l1', 'l2', 'l3', and 'l4'.
            l1 = len(list1)
            l2 = len(list2)
            
            # Iterate from 0 to the maximum length among the input lists using 'max' function.
            for i in range(max(l1, l2)):
                # Check if 'i' is less than 'l1' and add the element from 'list1' to 'result'.
                if i < l1:
                    result.append(list1[i])
                # Check if 'i' is less than 'l2' and add the element from 'list2' to 'result'.
                if i < l2:
                    result.append(list2[i])
            # Return the 'result' list containing interleaved elements from the input lists.
            return result
        
    def __init__(self, examples, batch_size, clases):
        
        self.CPHfemale = ['sub-049','sub-050','sub-051','sub-052','sub-065','sub-066','sub-077','sub-078','sub-079','sub-080','sub-081',
       'sub-082','sub-083']
        self.NAIVEfemale = ['sub-019','sub-020','sub-067','sub-068']
        
        self.CPHmale = ['sub-057','sub-059','sub-060','sub-073','sub-074','sub-093','sub-094','sub-095','sub-096','sub-098','sub-099',
                   'sub-100']
        self.NAIVEmale = ['sub-024','sub-028','sub-075','sub-076']
        
        self.batch_size = batch_size

        self.classes = clases

        self.df = examples
    def __iter__(self):
        batch = []

        c0=[]
        c1=[]
        
        if self.classes == "CPHvsNAIVEfemale":
            for i in range(len(self.df)):
                for j in self.CPHfemale:
                    if j in self.df[i][0]:
                        if 'ses-02' in self.df[i][0] or 'ses-03' in self.df[i][0]:
                            c1.append(self.df[i])
                        else:
                            c0.append(self.df[i])
                
                for k in self.NAIVEfemale:
                    if k in self.df[i][0]:
                        c0.append(self.df[i])
                        
        if self.classes == "CPHvsNAIVEmale":
            for i in range(len(self.df)):
                for j in self.CPHmale:
                    if j in self.df[i][0]:
                        if 'ses-02' in self.df[i][0] or 'ses-03' in self.df[i][0]: 
                            c1.append(self.df[i])
                        else:
                            c0.append(self.df[i])
                
                for k in self.NAIVEmale:
                    if k in self.df[i][0]:
                        c0.append(self.df[i])
                        
        if self.classes == "sex":
            for i in range(len(self.df)):
                for j in self.CPHfemale:
                    if j in self.df[i][0]:
                        c1.append(self.df[i])
                for k in self.CPHmale:
                    if k in self.df[i][0]:
                        c0.append(self.df[i])
                        
        if self.classes == "W1vsW7":
            for i in range(len(self.df)):
                if 'ses-02' in self.df[i][0]:
                    c1.append(self.df[i])
                elif 'ses-03' in self.df[i][0]:
                    c0.append(self.df[i])
                    
        
            
        indices0 = np.arange(len(c0),dtype=int)
        indices1 = np.arange(len(c1),dtype=int)
        
        np.random.shuffle(indices0)
        np.random.shuffle(indices1)
        
        shuffled = self.__interleave_diff_len_lists(np.array(c0)[indices0].tolist(), np.array(c1)[indices1].tolist())

        #print(np.array(shuffled)[:,0])

        indices = [self.df.index(element) for element in shuffled]

        #print(indices)

        batch = np.reshape(indices, (-1, self.batch_size))

        for i in range(len(batch)):
            yield batch[i]

    def __len__(self):
        return int(len(self.df)/self.batch_size)
    

class fMRIDataset_torch(Dataset):
    def __init__(self, data, vols, functional_type = "rest", format = "just_brain", clases = "CPHvsNAIVEfemale"):
        """
        Args:
            data: A dictionary or list containing all fMRI volumes for each session.
            labels: A list of labels corresponding to each session.
        """
        self.data = data
        self.functional_type = functional_type
        self.format = format
        self.vols = vols
        self.classes = clases

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #Volumes
        if self.functional_type == "rest":
            if self.format == "just_brain":
                img = nib.load(self.data[idx][0])
                mask = nib.load(self.data[idx][1])
                
                data = img.dataobj
                data = np.transpose(data, (3,0,1,2))
                
                a = np.linspace(19, 600, num=self.vols, endpoint=True, dtype=int)
                data = data[a]
                
                maskdata = mask.dataobj
                
                image_arr = maskdata*data
                image_arr = image_arr[:,3:45,4:69,7:36]
        #Labels
        male = ['sub-057','sub-059','sub-060','sub-073','sub-074','sub-093','sub-094','sub-095','sub-096',
                 #'sub-097',
                 'sub-098','sub-099','sub-100']
        female = ['sub-049','sub-050','sub-051','sub-052','sub-065','sub-066','sub-077','sub-078','sub-079','sub-080','sub-081','sub-082','sub-083',
                 #'sub-084',
                 ]
        CPHfemale = ['sub-049','sub-050','sub-051','sub-052','sub-065','sub-066','sub-077','sub-078','sub-079','sub-080','sub-081',
               'sub-082','sub-083']
        NAIVEfemale = ['sub-019','sub-020','sub-067','sub-068']
        
        CPHmale = ['sub-057','sub-059','sub-060','sub-073','sub-074','sub-093','sub-094','sub-095','sub-096','sub-098','sub-099',
                   'sub-100']
        NAIVEmale = ['sub-024','sub-028','sub-075','sub-076']
        
        label = []
        if self.classes == "CPHvsNAIVEfemale":
            if self.format == "just_brain":
                
                for j in NAIVEfemale:
                    if j in self.data[idx][0]:
                        label.append(0)
                        
                for i in CPHfemale:
                    if i in self.data[idx][0]:
                        if 'ses-02' in self.data[idx][0] or 'ses-03' in self.data[idx][0]:
                            label.append(1)
                        else:
                            label.append(0)
                if len(label) == 0:
                    raise ValueError(f"Subject generates conflict with labels: {self.data[idx][0]}")
                    
        label = label*self.vols
        
        # Retrieve all volumes for a specific session and its label
        return torch.tensor(image_arr, dtype=torch.float32), label

def collate_fn_torch(batch):
    """
    #Args:
    #    batch: A list of tuples, where each tuple contains (session_data, session_label).
    #Returns:
    #    sub_batches: A list of sub-batches of data.
    #    sub_labels: A list of labels corresponding to the sub-batches.
    """
    all_volumes = []
    all_labels = []

    for session_data, session_label in batch:
        all_volumes.extend(session_data)  # Combine volumes from the two sessions
        all_labels.extend(session_label)  # Collect session labels
        
    # Convert to tensor
    all_volumes_tensor = torch.stack(all_volumes)  # Shape: [total_volumes, D, H, W]
    all_labels_tensor = torch.tensor(all_labels)  # Shape: [total_volumes]


    # Shuffle volumes and labels
    perm = torch.randperm(len(all_volumes_tensor))  # Generate random permutation indices
    all_volumes_tensor = all_volumes_tensor[perm]  # Apply permutation to volumes
    all_labels_tensor = all_labels_tensor[perm]  # Apply permutation to labels
    
    # Perform Z-scoring
    mean = all_volumes_tensor.mean()
    std = all_volumes_tensor.std()
    all_volumes_tensor = (all_volumes_tensor - mean) / (std + 1e-8)

    # Create sub-batches of 30 volumes
    sub_batches = [all_volumes_tensor[i:i + 45] for i in range(0, len(all_volumes_tensor), 45)]
    sub_batches = [sub_batch.unsqueeze(1) for sub_batch in sub_batches]  # Add channel dimension

    # Replicate labels across all sub-batches
    sub_labels = [all_labels_tensor[i:i + 45] for i in range(0, len(all_labels_tensor), 45)]


    return sub_batches, sub_labels

