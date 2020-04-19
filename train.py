'''
    train.py
    File that defines all training related functions that are called in driver.py
    
    ______________________________________________________
    by Kiran Prasad <kiranpra@cs.cmu.edu>
    16-725 Methods in Medical Image Analysis Final Project
    ======================================================
'''
import torch
from advertorch.attacks import L2PGDAttack
from advertorch.utils import predict_from_logits
from utils import save_checkpoint


def train_loop_adversarial(
    adversary, model, optimizer, criterion, train_loader, device
):
    """
    Runs a single epoch of training using PGD
    """
    model.train()
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0

    for examples, labels in train_loader:
        optimizer.zero_grad()

        examples, labels = examples.to(device), labels.to(device)

        # Generate adversarial training examples
        # The model should not accumulate gradients in this step
        # Set to eval mode and toggle requires_grad
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        examples = adversary.perturb(examples, labels)
        for param in model.parameters():
            param.requires_grad = True
        model.train()

        outputs = model(examples)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()

        _, predictions = torch.max(outputs, 1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.shape[0]

    return train_loss / len(train_loader), 100 * train_correct / train_total


def validate(adversary_settings, model, criterion, val_loader, device):
    """
    Runs a single epoch of validation using Madry's adversarial method
    """
    # The model should not accumulate gradients
    # However, adversary will require gradients
    # Don't wrap this function with torch.no_grad()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    val_loss = 0.0
    val_correct = 0.0
    val_total = 0.0

    adv_epsilon, adv_iterations, adv_step = adversary_settings
    adversary = L2PGDAttack(
        model,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        eps=adv_epsilon,
        nb_iter=adv_iterations,
        eps_iter=adv_step,
        rand_init=True,
        clip_min=-1.0,
        clip_max=1.0,
        targeted=False,
    )

    for examples, labels in val_loader:
        examples, labels = examples.to(device), labels.to(device)

        outputs = model(examples)

        # Normal
        loss = criterion(outputs, labels)
        val_loss += loss.data.item()

        _, predictions = torch.max(outputs, 1)
        val_correct += (predictions == labels).sum().item()
        val_total += labels.shape[0]

        # Generate adversarial training examples
        adv_examples = adversary.perturb(examples, labels)

        adv_outputs = model(adv_examples)

        adv_loss = criterion(adv_outputs, labels)
        adv_val_loss += adv_loss.data.item()

        _, predictions = torch.max(adv_outputs, 1)
        adv_val_correct += (predictions == labels).sum().item()

    for param in model.parameters():
        param.requires_grad = True

    return val_loss / len(val_loader), 100 * val_correct / val_total, adv_val_loss / len(val_loader), 100 * adv_val_correct / val_total



def train_model(
    train_adversary_settings,
    val_adversary_settings,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    num_epochs,
    tensorboard_writer,
    device,
    save_settings,
):
    do_saving, save_directory, save_filename, starting_epoch, best_acc = save_settings

    # Initialize PGD adversary 
    adv_epsilon, adv_iterations, adv_step = train_adversary_settings
    adversary = L2PGDAttack(
        model,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        eps=adv_epsilon,
        nb_iter=adv_iterations,
        eps_iter=adv_step,
        rand_init=True,
        clip_min=0.0,
        clip_max=1.0,
        targeted=False,
    )

    for e in range(num_epochs):

       
        # Adversarial training loop
        train_loss, train_acc = train_loop_adversarial(
            adversary, model, optimizer, criterion, train_loader, device
        )
       
        tensorboard_writer.add_scalar(
            "Loss/Train", train_loss, starting_epoch + e)
        tensorboard_writer.add_scalar(
            "Acc/Train", train_acc, starting_epoch + e)
        
        val_loss, val_acc, adv_val_loss, adv_val_acc = validate(val_adversary_settings, model, criterion, val_loader, device)

        tensorboard_writer.add_scalar(
            "Loss/Val", val_loss, starting_epoch + e)
        tensorboard_writer.add_scalar(
            "Acc/Val", val_acc, starting_epoch + e)
        tensorboard_writer.add_scalar(
            "Loss/Adv_Val", adv_val_loss, starting_epoch + e)
        tensorboard_writer.add_scalar(
            "Acc/Adv_Val", adv_val_acc, starting_epoch + e)

        if do_saving:
            is_best = False
            if val_acc > best_acc:
                best_acc = val_acc
                is_best = True

            state = {
                "model_state": model.state_dict(),
                "accuracy": val_acc,
                "epoch": starting_epoch + e,
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(state, save_directory, save_filename, is_best)
