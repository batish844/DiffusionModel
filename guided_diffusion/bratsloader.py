import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        patient_ids_to_load = None
        if directory.lower().endswith('.txt') and os.path.isfile(directory):
            txt_path = os.path.expanduser(directory)
            with open(txt_path, 'r') as f:
                patient_ids_to_load = {line.strip() for line in f if line.strip()}
            search_directory = os.path.dirname(txt_path)
        else:
            search_directory = os.path.expanduser(directory)
        
        super().__init__()
        self.directory = search_directory
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                # --- Start of Changes ---
                # If a list of IDs was loaded, filter the folders by name
                if patient_ids_to_load is not None:
                    folder_name = os.path.basename(root)
                    if folder_name not in patient_ids_to_load:
                        continue # Skip this folder if its name is not in the list
                # --- End of Changes ---

                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    # --- Start of Changes ---
                    # Changed to handle .nii extension by splitting it off
                    seqtype = f.split('_')[3].split('.')[0]
                    # --- End of Changes ---
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):

        # --- Start of Changes ---
        patient_ids_to_load = None
        # Check if 'directory' is a path to a .txt file
        if directory.lower().endswith('.txt') and os.path.isfile(directory):
            txt_path = os.path.expanduser(directory)
            with open(txt_path, 'r') as f:
                # Load patient IDs into a set for efficient lookup
                patient_ids_to_load = {line.strip() for line in f if line.strip()}
            # The new directory to search is the one containing the .txt file
            search_directory = os.path.dirname(txt_path)
        else:
            # Otherwise, the directory is the search directory itself
            search_directory = os.path.expanduser(directory)
        # --- End of Changes ---

        super().__init__()
        self.directory = search_directory
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                # --- Start of Changes ---
                # If a list of IDs was loaded, filter the folders by name
                if patient_ids_to_load is not None:
                    folder_name = os.path.basename(root)
                    if folder_name not in patient_ids_to_load:
                        continue # Skip this folder if its name is not in the list
                # --- End of Changes ---
                
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
    
    def __len__(self):
        return len(self.database) * 155

    def __getitem__(self, x):
        out = []
        n = x // 155
        slice = x % 155
        filedict = self.database[n]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            o = torch.tensor(nib_img.get_fdata())[:,:,slice]
            # if seqtype != 'seg':
            #     o = o / o.max()
            out.append(o)
        out = torch.stack(out)
        if self.test_flag:
            image=out
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            # label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path