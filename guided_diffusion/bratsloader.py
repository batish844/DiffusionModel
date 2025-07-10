import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, transform, test_flag=False):
        # --- Hard-coded paths added ---
        DATA_ROOT = ( 
                       "MICCAI_BraTS2020_TrainingData"
        )
        IDS_TXT = "test.txt"

        with open(IDS_TXT, "r") as f:
            patient_ids_to_load = {line.strip() for line in f if line.strip()}
        
        search_directory = DATA_ROOT
        # --- End of added section ---

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
                # If a list of IDs was loaded, filter the folders by name
                if patient_ids_to_load is not None:
                    folder_name = os.path.basename(root)
                    if folder_name not in patient_ids_to_load:
                        continue # Skip this folder if its name is not in the list

                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    # strip off “.nii” or “.nii.gz” reliably
                    basename = os.path.splitext(f)[0]
                    # grab whatever comes after the last underscore
                    seqtype  = basename.rsplit('_', 1)[-1]
                    # only keep the modalities you expect
                    if seqtype not in self.seqtypes_set:
                        continue
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
    def __init__(self, transform, test_flag=False):
        # --- Hard-coded paths added ---
        DATA_ROOT = ( 
                       "MICCAI_BraTS2020_TrainingData"
        )
        IDS_TXT = "test.txt"

        with open(IDS_TXT, "r") as f:
            patient_ids_to_load = {line.strip() for line in f if line.strip()}
        
        search_directory = DATA_ROOT
        # --- End of added section ---

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
                # If a list of IDs was loaded, filter the folders by name
                if patient_ids_to_load is not None:
                    folder_name = os.path.basename(root)
                    if folder_name not in patient_ids_to_load:
                        continue # Skip this folder if its name is not in the list
                
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                # strip off “.nii” or “.nii.gz” reliably
                    basename = os.path.splitext(f)[0]
                    # grab whatever comes after the last underscore
                    seqtype  = basename.rsplit('_', 1)[-1]
                    # only keep the modalities you expect
                    if seqtype not in self.seqtypes_set:
                        continue
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