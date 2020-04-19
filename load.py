''' load.py
    
    Takes images from specified file and generates examples in
    the form of # of examples = # of findings. The examples are 
    generated as cubes around nodule-like occurrences that are 
    specified in the input file. Each image has the potential to
    generate more than 1 example if there is more than 1 finding. 

    For each image the loader will load the LNDbID, coordinates, 
    and texture. The file type will be converted to a numpy
    array. There are some requisite tranformations to obtain the 
    image coordinates necessary to extract the cube examples.

    Current behavior is to store generated examples in memory. 

    ################## ASSUMPTIONS ################
    # -train and validation csvs are in the same working directory
    # -images are in a subfolder from pwd called "data"
    ###############################################
    ______________________________________________________
    by Kiran Prasad <kiranpra@cs.cmu.edu>
    16-725 Methods in Medical Image Analysis Final Project
    ======================================================
'''
import os
import pickle
import numpy as np
from utils import readCsv, readMhd, extractCube, convertToImgCoord, getImgWorldTransfMats


def load_image(img_name, centers):
    ''' loads a single image and returns a list of subimages (cubes around nodule-like
            findings)  

        img_name: valid path to an image from current working directory
        centers: a list of the centroids of nodule-like occurrences in img
                in world coordinates
    '''
    scan, spacing, origin, transfmat = readMhd(img_name)
    transfmat_toimg, _ = getImgWorldTransfMats(spacing, transfmat)  # just need image
    centers = [convertToImgCoord(c, origin, transfmat_toimg) for c in centers]
    
    subimages = [extractCube(scan, spacing, c) for c in centers]
    return subimages
    

    
def threshold_texture(texture):
    ''' Buckets the radiologist averaged texture into 
    0 for nonNodules (Text == 0)
    1 for less than 2.33 (0 < Text <= 2.33)
    2 for between 2.33 and 3.66 (2.33 < Text <= 3.66)
    3 for greater than 3.66 (3.66 < Text)
    '''
    if texture == 0:
        return 0
    elif texture <= 2.33:
        return 1
    elif texture <= 3.66:
        return 2
    else:
        return 3


def process_line(header, line, consolidate_centers, consolidate_labels):
    ''' Processes 1 line of a csv input file
    to extract the id, coordinates, and texture

    adds to the consolidation tables which are a 
    dictionaries with {lndbid: [centers] } or {lndbid: [labels] }
    respectively
    where the labels are thresholded 
    '''
    ID = line[header.index('LNDbID')]
    if ID in consolidate_centers:
        if ID not in consolidate_labels:
            raise NameError('ID is in centers but not labels')
        consolidate_centers[ID].append(
            np.array(
                [float(line[header.index('x')]), float(line[header.index('y')]), float(line[header.index('z')])]
            )
        )
        consolidate_labels[ID].append(
            threshold_texture(float(line[header.index('Text')]))
        )
    else:
        if ID in consolidate_labels:
            raise NameError('ID is in labels but not centers')
        consolidate_centers[ID] = [np.array([float(line[header.index('x')]), float(line[header.index('y')]), float(line[header.index('z')])])]
        consolidate_labels[ID] = [
            threshold_texture(float(line[header.index('Text')]))
        ]

def generate_data(data_file, fold, data_path='../data/', is_train=True):

    
    lines = readCsv(data_file)
    header = lines[0]
    lines = lines[1:]
    consolidate_centers = {}
    consolidate_labels = {}

    # loop through lines, collate labels, centers by images
    for line in lines:
        process_line(header, line, consolidate_centers, consolidate_labels)

    # load and save examples (subimages) and labels (textures)
    examples = []
    labels = []
    for Id, centers in consolidate_centers.items():
        img_name = 'LNDb-{:04}.mhd'.format(int(Id))
        examples += load_image(os.path.join(data_path,img_name), centers)
        labels += consolidate_labels[Id]

    examples = np.array(examples)
    labels = np.array(labels)

    # now the labels and examples should be in order
    # so we save as a pickle file (serialized file) to be loaded by ML model
    data = ('train' if is_train else 'val') + \
        '_data_except_{}.npy'.format(fold)
    labels = ('train' if is_train else 'val') + \
        '_labels_except_{}.npy'.format(fold)

    DATAFILE = open(data, mode='wb')
    pickle.dump(examples, DATAFILE)
    DATAFILE.close()

    LABELFILE = open(labels, mode='wb')
    pickle.dump(labels, LABELFILE)
    LABELFILE.close()

if __name__ == "__main__":
    fold = input("Which fold would you like to load for validation?")
    train_file = '../folds_exclude_{}Nodules_gt.csv'.format(fold)
    val_file = '../val_fold_{}Nodules_gt.csv'.format(fold)
    generate_data(train_file, fold)
    generate_data(val_file, fold, is_train=False)
    

    

    
    
     
