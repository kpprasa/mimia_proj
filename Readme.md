# ADVERSARIAL ROBUSTNESS IN MEDICAL IMAGES# 
## SHORT DESCRIPTION 
This project aimed to make a first novel foray into the exciting space of adversarial
robustness as it pertains to medical image analysis. This project uses the LNDb dataset 
from the CHUSJ in Porto, Portugal. It is comprised of 294 (expanded to 312) CT scans of the lungs. More 
information on this dataset can be found here: https://lndb.grand-challenge.org. 
Please refer to the final presentation for more information. 

I tried to do the image visualization, but ran into significant issues with the custom loss function. Essentially, this would require a rewrite of the robustness repository to work with the code I have written 
and it likely an additional two weeks work in itself. Should I find some time, I may try to implement this later. A realization I had, is that for this to be clinically relevant, it has to be very fast to use for doctors and this likely requires some multimodal NLP on top of meaningful representations. This is an area I may look further into. 

The overall pipeline looks like: 
readNoduleList (if needed)-> getFolds (if needed)-> load -> driver

## TO RUN
### Confirmed Environment: 
    - python 3.8.2
    - Ubuntu Bionic Beaver
    - advertorch 0.2.2
    - pytorch 1.4.0
    - newest tensorboard
    - GPU with at least 16 GB 

Install advertorch if you don't currently have it:
`pip install advertorch` 
Install SITK if you don't currently have it:
`pip install SimpleITK`

You should have access to the data as a google drive link that importantly should have:
- A data folder with 312 mhd/raw image pairs

** This repo should have: **
1. driver.py: main driver for ML portion
2. train.py: training files from driver
3. getFolds.py: way to determine train/val folds
4. load.py: custom loader to create data for ML portion
5. utils.py: a file with misc. utility functions
6. trainFolds.csv: used by getFolds
7. TrainNodules.csv: this has the radiologist data, but is unconsolidated

---------------------------------------------------------------------------------
1. Is there a file named `trainNodules_gt.csv`? If so, continue onto 2. Otherwise generate this file by running ```$ python readNoduleList.py```. This file is the ground truth consolidated across radiologists and is necessary to continue. 

2. Take a look and see if there is are files with the following structure ```folds_exclude_0Nodules_gt.csv```  and ```val_fold_0Nodules_gt.csv``` where the number can be between 0 and 3. If not run getFolds.py ```$python getFolds.py``` 4 times and input numbers 0-3 each time. Then run it 4 more times with the same inputs with the following flag: ```$python getFolds.py --val False```. This is to create the train CSVs. 

3. Run load.py using ```$python load.py``` and enter a number between 0 and 3 when prompted for a fold (I usually used 0). *NOTE: this file assumes that there is a folder in the present working directory (pwd) that called data that actually has the data.* You should now have pickle files (.npy) for train data and labels. 

4. Now the driver can be run (hopefully). Look at the driver.py folder for the numerous arguments. However, if you chose fold 0 earlier, you should be able to run it simply by ```$python driver.py --batch_size 16```. You may have to specify the datapath via the `--datapath` argument. This datapath MUST have both the npy files generated in 3 (which in turn are likely required to be in the same folder as the csv files in 2). 

5. Done! Feel free to follow the progress with tensorboard. The accuracy should get somewhere in the vicinity of 40-60. If you run into memory issues with cuda, reduce the batch size further. 
Please let me know if you have any questions. Interpretability is a massive task to take on, but with time and work, it may be solvable by bringing together different disciplines within ML. 
