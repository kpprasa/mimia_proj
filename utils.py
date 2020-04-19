''' utils.py
    
    Contains many helper functions. Some are associated with the LNDb Dataset, while others
    are written for the neural network itself. 
    ______________________________________________________
    by Kiran Prasad <kiranpra@cs.cmu.edu>
    16-725 Methods in Medical Image Analysis Final Project
    ======================================================
'''
import shutil
import PIL
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import csv
import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def writeCsv(csfname,rows):
    # write csv from list of lists
    with open(csfname, 'w', newline='') as csvf:
        filewriter = csv.writer(csvf)
        filewriter.writerows(rows)
        
def readMhd(filename):
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage) #3D image
    spacing = itkimage.GetSpacing() #voxelsize
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    return scan,spacing,origin,transfmat

def writeMhd(filename,scan,spacing,origin,transfmat):
    # write mhd/raw image
    itkim = sitk.GetImageFromArray(scan, isVector=False) #3D image
    itkim.SetSpacing(spacing) #voxelsize
    itkim.SetOrigin(origin) #world coordinates of origin
    itkim.SetDirection(transfmat) #3D rotation matrix
    sitk.WriteImage(itkim, filename, False)    

def getImgWorldTransfMats(spacing,transfmat):
    # calc image to world to image transformation matrixes
    transfmat = np.array([transfmat[0:3],transfmat[3:6],transfmat[6:9]])
    for d in range(3):
        transfmat[0:3,d] = transfmat[0:3,d]*spacing[d]
    transfmat_toworld = transfmat #image to world coordinates conversion matrix
    transfmat_toimg = np.linalg.inv(transfmat) #world to image coordinates conversion matrix
    
    return transfmat_toimg,transfmat_toworld

def convertToImgCoord(xyz,origin,transfmat_toimg):
    # convert world to image coordinates
    xyz = xyz - origin
    xyz = np.round(np.matmul(transfmat_toimg,xyz))    
    return xyz
    
def convertToWorldCoord(xyz,origin,transfmat_toworld):
    # convert image to world coordinates
    xyz = np.matmul(transfmat_toworld,xyz)
    xyz = xyz + origin
    return xyz

def extractCube(scan, spacing, xyz, cube_size=80, cube_size_mm=51):
    ''' Extract cube of cube_size^3 voxels and world dimensions of cube_size_mm^3 mm 
        from scan at image coordinates xyz


        scan = image
        spacing = list or tuple of spacing along each axis 
        xyz = center of the cube (image coordinates)
    '''

    xyz = np.array([xyz[i] for i in [2,1,0]],np.int) # itk convention
    spacing = np.array([spacing[i] for i in [2,1,0]]) # itk convention
    scan_halfcube_size = np.array(cube_size_mm/spacing/2,np.int)
    if np.any(xyz<scan_halfcube_size) or np.any(xyz+scan_halfcube_size>scan.shape): # check if padding is necessary
        maxsize = max(scan_halfcube_size)
        scan = np.pad(scan,((maxsize,maxsize,)),'constant',constant_values=0)
        xyz = xyz+maxsize
    
    scancube = scan[xyz[0]-scan_halfcube_size[0]:xyz[0]+scan_halfcube_size[0], # extract cube from scan at xyz
                    xyz[1]-scan_halfcube_size[1]:xyz[1]+scan_halfcube_size[1],
                    xyz[2]-scan_halfcube_size[2]:xyz[2]+scan_halfcube_size[2]]

    sh = scancube.shape
    # resample for cube_size
    scancube = zoom(
        scancube, (cube_size/sh[0], cube_size/sh[1], cube_size/sh[2]), order=2)

    return scancube


############### BEGIN NN HELPERS ##################
def save_checkpoint(state_dict, save_directory, save_filename, is_best):
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    full_save_path = os.path.join(save_directory, save_filename)

    torch.save(state_dict, full_save_path)

    if is_best:
        shutil.copyfile(
            full_save_path, os.path.join(
                save_directory, "best_" + save_filename),
        )


def load_checkpoint(model, optimizer, save_directory, load_filename, device):
    checkpoint = torch.load(os.path.join(save_directory, load_filename))
    model.load_state_dict(checkpoint['model_state'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # Make sure optimizer is on same device as model
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    return model, optimizer, checkpoint["accuracy"], checkpoint["epoch"]


def mean_normalize(data):
    """Normalize data by mean"""
    # mean should be over time and broadcast
    data= data - torch.mean(data, dim=0)
    return data

def rescale(data):
    """scales data between [-1,1]"""
    m = torch.max(torch.abs(data))
    return data / m 

def get_dataloaders(batch_size, fold=0, DEVICE='cuda', datapath):
    """
    Used to get dataloaders which is used for standard training. 
    """

    # Load data
    val_data = np.load(os.path.join(datapath,'val_data_except_{}.npy'.format(fold)), allow_pickle=True)
    train_data = np.load(
        os.path.join(datapath, 'train_data_except_{}.npy'.format(
            fold)), allow_pickle=True)
    val_labels = np.load(os.path.join(datapath, 'val_labels_except_{}.npy'.format(fold)),
                        allow_pickle=True)  # N x S (variable)
    train_labels = np.load(
        os.path.join(datapath, 'train_labels_except_{}.npy'.format(
            fold)), allow_pickle=True)


    # probably want to mean normalize over time
    val_labels = torch.tensor(val_labels)
    val_data = torch.tensor(val_data)
    val_data = mean_normalize(val_data)
    val_data = rescale(val_data)

    train_labels = [torch.Tensor(y) for y in train_labels]
    train_data = [torch.Tensor(x) for x in train_data]
    train_data = mean_normalize(train_data)
    train_data = rescale(train_data)


    class LNDbDataset(Dataset):
        def __init__(self, examples, labels):
            self.examples=examples
            self.phonemes=labels
        def __getitem__(self, i):
            ''' returns the example and label of the corresponding index'''
            return self.examples[i, :,:,:], self.phonemes[i,:,:,:]
        def __len__(self):
            return self.labels.shape[0]
    
    train_dataset = LNDbDataset(train_data, train_labels)
    val_dataset = LNDbDataset(val_data, val_labels)
    
    train_loader=DataLoader(
        train_dataset,
        shuffle = True,
        #drop_last=True,
        batch_size = batch_size,
        pin_memory = True,   # Copy data to CUDA pinned memory
        # so that they can be transferred to the GPU very fast
        # Number of worker processes for loading data.
        num_workers = 8 if DEVICE == 'cuda' else os.cpu_count()//2
    )
    val_loader=DataLoader(
        val_dataset,  
        batch_size = batch_size,      # Batch size
        shuffle = False,      # Ok to shuffle after creation
        pin_memory = True,   # Copy data to CUDA pinned memory
        # so that they can be transferred to the GPU very fast
        # Number of worker processes for loading data.
        num_workers = 8 if DEVICE == 'cuda' else os.cpu_count()//2
    )

    return train_loader, val_loader

################ END NN HELPERS ###################
if __name__ == "__main__":
    #Extract and display cube for example nodule
    lnd = 1
    rad = 1
    finding = 1
    # Read scan
    # ../drive/My Drive/MIMIA/ for colab
    [scan,spacing,origin,transfmat] =  readMhd("data/LNDb-0001.mhd")
    print(spacing,origin,transfmat)
    # Read segmentation mask
    [mask,spacing,origin,transfmat] =  readMhd('masks/LNDb-{:04}_rad{}.mhd'.format(lnd,rad))
    print(spacing,origin,transfmat)
    # Read nodules csv
    csvlines = readCsv('trainNodules.csv')
    header = csvlines[0]
    nodules = csvlines[1:]
    for n in nodules:
        if int(n[header.index('LNDbID')])==lnd and int(n[header.index('RadID')])==rad and int(n[header.index('FindingID')])==finding:
            ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
            break
    
    # Convert coordinates to image
    transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
    ctr = convertToImgCoord(ctr,origin,transfmat_toimg)
    
    # Display nodule scan/mask slice
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(scan[int(ctr[2])])
    axs[1].imshow(mask[int(ctr[2])])
    plt.show()
    
    # Extract cube around nodule
    scan_cube = extractCube(scan,spacing,ctr)
    mask[mask!=finding] = 0
    mask[mask>0] = 1
    mask_cube = extractCube(mask,spacing,ctr)
    
    # Display mid slices from resampled scan/mask
    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:])
    axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])
    axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:])
    axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])
    axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)])
    axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])    
    plt.show()
    
    
