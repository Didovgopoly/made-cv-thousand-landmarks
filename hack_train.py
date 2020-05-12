"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""
#  ssh -i c:\Users\d.dovgopolyi\.ssh\MADE_student_key.pem ubuntu@52.58.162.44
import os
import pickle
import sys
from argparse import ArgumentParser
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.utils import data
from torchvision import transforms
from torch.nn import functional as fnn
from hack_utils import NUM_PTS, CROP_SIZE, Cutout, RandomBlur
from hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from hack_utils import ThousandLandmarksDataset
from hack_utils import restore_landmarks_batch, create_submission

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetune @ 6Gb of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--freeze","-f", default=0, type=int)
    parser.add_argument("--cont", "-c", default=0, type=int)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    with tqdm.tqdm(total=len(loader), position=0, leave=True) as pbar:
        for batch in tqdm.tqdm(loader, total=len(loader), desc="training...", position=0, leave=True):
            images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
            landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

            pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
            loss = loss_fn(pred_landmarks, landmarks)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    with tqdm.tqdm(total=len(loader), position=0, leave=True) as pbar:
        for batch in tqdm.tqdm(loader, total=len(loader), desc="validation...", position=0, leave=True):
            images = batch["image"].to(device)
            landmarks = batch["landmarks"]

            with torch.no_grad():
                pred_landmarks = model(images).cpu()
            loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
            val_loss.append(loss.item())

            pbar.update()

    return np.mean(val_loss)

def validate_full(model, loader, loss_fn, device):
    model.eval()
    val_loss = {}

    with tqdm.tqdm(total=len(loader), position=0, leave=True) as pbar:
        for batch in tqdm.tqdm(loader, total=len(loader), desc="validation...", position=0, leave=True):
            images = batch["image"].to(device)
            landmarks = batch["landmarks"]

            with torch.no_grad():
                pred_landmarks = model(images).cpu()
            loss = loss_fn(pred_landmarks, landmarks, reduction="none")

            for l, i, b in zip(loss, batch["name"], batch["size"]):
                ll = np.mean(l.numpy())
                val_loss[i] = (100 * ll.astype(float)) / float(b.item())

            pbar.update()

    return val_loss


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    with tqdm.tqdm(total=len(loader), position=0, leave=True) as pbar:
        for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...", position=0, leave=True)):
            images = batch["image"].to(device)

            with torch.no_grad():
                pred_landmarks = model(images).cpu()
            pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

            fs = batch["scale_coef"].numpy()  # B
            margins_x = batch["crop_margin_x"].numpy()  # B
            margins_y = batch["crop_margin_y"].numpy()  # B
            prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
            predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction
            pbar.update()

    return predictions


def main(args):
    # 1. prepare data & models
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        Cutout(10),
        RandomBlur(),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
    ])

    val_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
    ])

    print("Creating model...")
    device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
    model = models.resnet50(pretrained=True)

    if args.freeze > 0:
        ct = 0
        for child in model.children():
            ct += 1
            if ct <= args.freeze + 4:
                for param in child.parameters():
                    param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)

    startEpoch = args.cont
    if startEpoch > 0:
        with open(f"{args.name}_best_{startEpoch}.pth", "rb") as fp:
            best_state_dict = torch.load(fp, map_location="cpu")
            model.load_state_dict(best_state_dict)

    model.to(device)

    if args.test:
        val_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), val_transforms, split="train")
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                         shuffle=False, drop_last=False)
        val_loss_fn = fnn.mse_loss

        val_full = validate_full(model, val_dataloader, val_loss_fn, device=device)

        res = dict(sorted(val_full.items(), key=lambda x: x[1], reverse=True)[:100])
        js = json.dumps(res)
        with open(f"{args.name}.json", "w") as f:
            f.write(js)
        print(res)
        return

    if not args.predict:
        print("Reading data...")
        train_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), train_transforms, split="train")
        train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                           shuffle=True, drop_last=True)
        val_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), val_transforms, split="val")
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                         shuffle=False, drop_last=False)

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        train_loss_fn = nn.SmoothL1Loss(reduction="mean")
        val_loss_fn = fnn.mse_loss

        # 2. train & validate
        print("Ready for training...")
        best_val_loss = np.inf
        for epoch in range(startEpoch, args.epochs):
            train_loss = train(model, train_dataloader, train_loss_fn, optimizer, device=device)
            val_loss = validate(model, val_dataloader, val_loss_fn, device=device)
            scheduler.step(val_loss)
            print("Epoch #{:2}:\ttrain loss: {:.5f}\tval loss: {:.5f}".format(epoch, train_loss, val_loss))
            with open(f"{args.name}_res.txt", 'a+') as file:
                file.write("Epoch #{:2}:\ttrain loss: {:.5f}\tval loss: {:.5f}\n".format(epoch, train_loss, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                with open(f"{args.name}_best.pth", "wb") as fp:
                    torch.save(model.state_dict(), fp)

            if epoch > startEpoch and epoch % 5 == 0:
                best_val_loss = val_loss
                with open(f"{args.name}_best_{epoch}.pth", "wb") as fp:
                    torch.save(model.state_dict(), fp)

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'test'), val_transforms, split="test")
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                      shuffle=False, drop_last=False)

    with open(f"{args.name}_best.pth", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    for layer in model.modules():
        layer.eval()

    test_predictions = predict(model, test_dataloader, device)
    with open(f"{args.name}_test_predictions.pkl", "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, f"{args.name}_submit.csv")


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
