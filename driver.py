'''
    driver.py
    Main driver for Final Project. Assumes that the data and labels have been generated
    via get_Folds.py and load.py. Trains a neural network (in model.py) via Projected
    Gradient Descent to learn texture classification and visualize the learned representations.

    ################## Invariants #################
    # train and val files are of the form 
    # "[train/val]_[data/labels]_except_{fold}"
    # e.g. train_data_except_0
    ###############################################

    ______________________________________________________
    by Kiran Prasad <kiranpra@cs.cmu.edu>
    16-725 Methods in Medical Image Analysis Final Project
    ======================================================
'''
import argparse
import torch
from torch import optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time

from model import generate_model

from utils import *
from train import *


parser = argparse.ArgumentParser(
    description="Adversarial Training for LNDb Dataset; MIMIA 2020 Final project by Kiran Prasad"
)

####### DATALOADER SETTINGS
parser.add_argument(
    "--datapath",
    type=str,
    default="..",
    help="path where train and validation data can be found",
)
parser.add_argument(
    "--fold",
    type=str,
    default='0',
    help="Fold number",
)


####### TRAINING SETTINGS
parser.add_argument(
    "--batch_size", type=int, default=64, help="training batch size",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.02,
    help="Initial learning rate - 0.02 default",
)
parser.add_argument(
    "--num_epochs", type=int, default=50, help="Number of training epochs",
)

############ ADVERSARIAL SETTINGS
parser.add_argument(
    "--train_adv_epsilon",
    type=float,
    default=8 / 255,
    help="Radius of sphere for PGD adversary used for training",
)
parser.add_argument(
    "--train_adv_iterations",
    type=int,
    default=7,
    help="Number of iterations for PGD adversary used for training",
)
parser.add_argument(
    "--eval_adv_epsilon",
    type=float,
    default=8 / 255,
    help="Radius of sphere for PGD adversary used for eval",
)
parser.add_argument(
    "--eval_adv_iterations",
    type=int,
    default=7,
    help="Number of iterations for PGD adversary used for eval",
)


######## SAVING/LOADING MODEL SETTINGS
parser.add_argument(
    "--do_saving",
    type=int,
    default=0,
    help="0 to skip saving, 1 to save model being trained",
)
parser.add_argument(
    "--save_directory",
    type=str,
    default="saves",
    help="Directory to a saved model file",
)
parser.add_argument(
    "--save_filename",
    type=str,
    default="model_checkpoint",
    help="Directory to a saved model file",
)
parser.add_argument(
    "--do_training",
    type=int,
    default=1,
    help="0 to skip to eval, 1 to do training",
)
parser.add_argument(
    "--resume_training",
    type=int,
    default=0,
    help="0 to train from scratch, 1 to load from checkpoint and continue training",
)
parser.add_argument(
    "--load_filename",
    type=str,
    default="best_model_checkpoint.pt",
    help="Path for loading a saved model file",
)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Setup
    train_loader, val_loader = get_dataloaders(
        args.batch_size, args.fold, DEVICE)
    
    adv_step = 0.25 * args.train_adv_epsilon
    train_adversary_settings = (
        args.train_adv_epsilon, args.train_adv_iterations, adv_step)

    eval_adversary_settings = (args.eval_adv_epsilon, args.eval_adv_iterations)

    # using ResNet50
    model = generate_model(50) 
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), args.lr,
                          momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # flag, directory, filename, starting epoch, best accuracy so far
    save_settings = (args.do_saving, args.save_directory,
                     args.save_filename, 0, 0.0)

    # Train
    if args.do_training == 1:
        # Log training metrics in tensorboard
        # run cmd in separate terminal: tensorboard --logdir runs
        # view on localhost:6006
        tensorboard_writer = SummaryWriter()

        # Load checkpoint
        if args.resume_training == 1:
            model, optimizer, accuracy, epoch = load_checkpoint(
                model, optimizer, args.save_directory, args.load_filename, DEVICE)
            baseline_loss, baseline_acc, adv_l, adv_a = validate(eval_adversary_settings,
                model, criterion, val_loader, DEVICE)
            save_settings = (args.do_saving, args.save_directory,
                             args.save_filename, epoch + 1, accuracy)
            print("*" * 100)
            print("Resuming training from:", args.load_filename,
                  "\tepoch:", epoch, "\taccuracy:", accuracy)
            print("-" * 100)
        else:
            print("*" * 100)
            print("Training new model.")
            print("-" * 100)

            
            train_model(
                train_adversary_settings,
                eval_adversary_settings,
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                args.num_epochs,
                tensorboard_writer,
                DEVICE,
                save_settings
            )

    # Eval only
    else:
        model, _, accuracy, epoch = load_checkpoint(
            model, optimizer, args.save_directory, args.load_filename, DEVICE)
        print("*" * 100)
        print("Evaluating model from:", args.load_filename,
              "\tepoch:", epoch, "\taccuracy:", accuracy)
        print("-" * 100)

    start = time.time()
    

    # Evaluate accuracy
    start = time.time()
    normal_loss, normal_acc, adv_loss, adv_acc = validate(
        eval_adversary_settings, model, criterion, val_loader, DEVICE)
    print("Normal accuracy:", normal_acc,
          "\tNormal loss:", normal_loss)
    print("Adversarial accuracy:", adv_acc,
          "\tAdversarial loss:", adv_loss)
    print("Validation time:", time.time() - start)
    print("-" * 100)

