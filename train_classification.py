from operator import attrgetter
from rbm import RBMClassification
from optim import Adam, SGD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime
import os
import argparse


def reconst_loss(probs, targets):
    targets = F.one_hot(targets.long())  # (B, I, 2)
    probs = torch.stack([probs, 1 - probs], dim=-1)  # (B, I, 2)
    loss = targets * torch.log(probs + 1e-10)
    loss = loss.sum(dim=[-1, -2])
    return loss.mean()


def categorical_cross_entropy(probs, labels):
    loss = (labels * torch.log(probs + 1e-10)).sum(-1)
    return loss.mean()


def accuracy(preds, labels):
    labels = torch.argmax(labels, dim=-1)
    acc = (preds == labels).sum()
    acc = acc / labels.shape[0]
    return acc


def imshow(grid, path):
    plt.figure()
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Restricted Boltzmann Machine MNIST Classification"
    )
    parser.add_argument(
        "--id",
        type=str,
        default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--k_cd", type=int, default=1)
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="pcd_cont",
        choices=["pcd", "cd", "pcd_cont"],
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--log_dir", type=str, default="log_cl")
    parser.add_argument("--fig_dir", type=str, default="fig_cl")
    parser.add_argument("--device_id", type=str, default="cuda:0")
    parser.add_argument("--num_gib", type=int, default=300)
    parser.add_argument("--result_format", type=str, default="png")
    args = parser.parse_args()

    # tensorboard
    log_dir = f"./{args.log_dir}"
    os.makedirs(log_dir, exist_ok=True)
    summary_name = f"{log_dir}/mnist_{args.id}"
    writer = SummaryWriter(summary_name)

    # dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
            transforms.Lambda(lambda x: torch.where(x <= 0.5, 0.0, 1.0)),
        ]
    )
    train_sets = MNIST(root="./data", train=True, transform=transform, download=True)
    test_sets = MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_sets, batch_size=args.batch_size, shuffle=False)

    # model
    device = args.device_id if torch.cuda.is_available() else "cpu"
    train_loader_iter = iter(train_loader)
    test_loader_iter = iter(test_loader)
    inputs_train_const, labels_train_const = next(train_loader_iter)
    inputs_test_const, labels_test_const = next(test_loader_iter)
    inputs_train_const, inputs_test_const = (
        inputs_train_const.to(device),
        inputs_test_const.to(device),
    )
    labels_train_const, labels_test_const = (
        F.one_hot(labels_train_const).to(device),
        F.one_hot(labels_test_const).to(device),
    )

    H, W, label_size = 28, 28, 10
    model = RBMClassification(
        visible_size=H * W,
        label_size=label_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        k_cd=args.k_cd,
        device=device,
    )

    if args.optim == "adam":
        optimizer = Adam(model.param, lr=args.lr)
    elif args.optim == "sgd":
        optimizer = SGD(model.param, lr=args.lr)
    else:
        NotImplementedError()

    sum_train_reconst = 0
    sum_test_reconst = 0
    metrics = {
        "epoch": [],
        "fe_train": [],
        "fe_test": [],
        "fe_diff": [],
        "acc_train": [],
        "acc_test": [],
        "reconst_train": [],
        "reconst_test": [],
        "pl_train": [],
        "pl_test": [],
        "ce_train": [],
        "ce_test": [],
    }

    for epoch in tqdm(range(args.num_epoch)):
        sum_train_acc = 0
        sum_test_acc = 0
        sum_train_reconst = 0
        sum_test_reconst = 0
        sum_train_ce = 0
        sum_test_ce = 0
        sum_train_pl = 0
        sum_test_pl = 0
        for inputs, labels in train_loader:
            inputs, labels = (
                inputs.to(device),
                F.one_hot(labels, num_classes=label_size).to(device),
            )
            grad = model.grad(inputs, labels.float(), sample_type="pcd_cont")
            grad = optimizer.calc_grad(grad)
            model.update(grad)
            outputs_data, outputs_label, probs_data, probs_label = model.sample_by_v(
                inputs, labels.float(), num_gib=args.num_gib
            )
            preds = model.classification(inputs)
            sum_train_acc += accuracy(preds, labels).item()
            sum_train_reconst += reconst_loss(probs_data, inputs).item()
            sum_train_ce += categorical_cross_entropy(probs_label, labels).item()
            sum_train_pl += (
                -model.pseudo_likelihood(inputs, labels.float()).mean().item()
            )

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = (
                    inputs.to(device),
                    F.one_hot(labels, num_classes=label_size).to(device),
                )
                outputs_data, outputs_label, probs_data, probs_label = (
                    model.sample_by_v(inputs, labels.float(), num_gib=args.num_gib)
                )
                preds = model.classification(inputs)
                sum_test_acc += accuracy(preds, labels).item()
                sum_test_reconst += reconst_loss(probs_data, inputs).item()
                sum_test_ce += categorical_cross_entropy(probs_label, labels).item()
                sum_test_pl += (
                    -model.pseudo_likelihood(inputs, labels.float()).mean().item()
                )

        # reconstruction
        outputs_data, outputs_label, probs_data, probs_label = model.sample_by_v(
            inputs_train_const, labels_train_const.float(), num_gib=args.num_gib
        )
        inputs_data = inputs_train_const.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
        outputs_data = outputs_data.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)

        diff_data = ((inputs_data - outputs_data) + 1) / 2
        images = torch.cat([inputs_data[:10], outputs_data[:10], diff_data[:10]], dim=0)
        writer.add_image("reconst_img", make_grid(images, nrow=10), epoch)

        # calculate mean free-energy
        free_energy_train = (
            model.free_energy(inputs_train_const, labels_train_const.float())
            .mean()
            .item()
        )
        free_energy_test = (
            model.free_energy(inputs_test_const, labels_test_const.float())
            .mean()
            .item()
        )
        free_energy_diff = free_energy_train - free_energy_test
        writer.add_scalar("free_energy_train", free_energy_train, epoch)
        writer.add_scalar("free_energy_test", free_energy_test, epoch)
        writer.add_scalar("free_energy_diff", free_energy_diff, epoch)

        # calculate mean reconstruction-loss
        train_acc = sum_train_acc / len(train_loader)
        test_acc = sum_test_acc / len(test_loader)
        train_reconst = sum_train_reconst / len(train_loader)
        test_reconst = sum_test_reconst / len(test_loader)
        train_ce = sum_train_ce / len(train_loader)
        test_ce = sum_test_ce / len(test_loader)
        train_pl = sum_train_pl / len(train_loader)
        test_pl = sum_test_pl / len(test_loader)
        writer.add_scalar("acc_train", train_acc, epoch)
        writer.add_scalar("acc_test", test_acc, epoch)
        writer.add_scalar("train_reconst", train_reconst, epoch)
        writer.add_scalar("test_reconst", test_reconst, epoch)
        writer.add_scalar("train_ce", train_ce, epoch)
        writer.add_scalar("test_ce", test_ce, epoch)
        writer.add_scalar("train_pl", train_pl, epoch)
        writer.add_scalar("test_pl", test_pl, epoch)

        # metrics
        metrics["epoch"].append(epoch)
        metrics["fe_train"].append(free_energy_train)
        metrics["fe_test"].append(free_energy_test)
        metrics["fe_diff"].append(free_energy_diff)
        metrics["acc_train"].append(train_acc)
        metrics["acc_test"].append(test_acc)
        metrics["reconst_train"].append(train_reconst)
        metrics["reconst_test"].append(test_reconst)
        metrics["ce_train"].append(train_ce)
        metrics["ce_test"].append(test_ce)
        metrics["pl_train"].append(train_pl)
        metrics["pl_test"].append(test_pl)

        if epoch % 10 == 0:
            print(
                f"train_reconst_loss={train_reconst}, test_reconst_loss={test_reconst}"
            )
            print(f"train_pl={train_pl}, test_pl={test_pl}")
            print(f"train_ce={train_ce}, test_ce={test_ce}")

    # evaluate RBM
    try:
        inputs, labels = next(train_loader_iter)
        inputs, labels = (
            inputs.to(device),
            F.one_hot(labels, num_classes=label_size).to(device),
        )
    except StopIteration:
        train_loader_iter = iter(train_loader)
        inputs, labels = next(train_loader_iter)
        inputs, labels = (
            inputs.to(device),
            F.one_hot(labels, num_classes=label_size).to(device),
        )

    outputs_data, outputs_label, probs_data, probs_label = model.sample_by_v(
        inputs, labels.float(), num_gib=args.num_gib
    )
    inputs = inputs.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
    outputs = outputs_data.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
    diff = ((inputs - outputs) + 1) / 2
    images = torch.cat([inputs[:10], outputs[:10], diff[:10]], dim=0)
    grid = make_grid(images, nrow=10).cpu()
    format = args.result_format
    fig_dir = f"./{args.fig_dir}"
    os.makedirs(fig_dir, exist_ok=True)
    path = f"{fig_dir}/reconst_img.{format}"
    imshow(grid, path)

    plt.figure()
    plt.plot(metrics["epoch"], metrics["reconst_train"], label="train", marker=".")
    plt.plot(metrics["epoch"], metrics["reconst_test"], label="test", marker=".")
    plt.xlabel("epoch")
    plt.ylabel("reconst_loss")
    plt.legend()
    plt.savefig(f"{fig_dir}/result_reconst.{format}")

    plt.figure()
    plt.plot(metrics["epoch"], metrics["acc_train"], label="train", marker=".")
    plt.plot(metrics["epoch"], metrics["acc_test"], label="test", marker=".")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f"{fig_dir}/result_accuracy.{format}")

    plt.figure()
    plt.plot(metrics["epoch"], metrics["fe_train"], label="train", marker=".")
    plt.plot(metrics["epoch"], metrics["fe_test"], label="test", marker=".")
    plt.xlabel("epoch")
    plt.ylabel("free_energy")
    plt.legend()
    plt.savefig(f"{fig_dir}/result_fe.{format}")

    plt.figure()
    plt.plot(metrics["epoch"], metrics["fe_diff"], marker=".")
    plt.xlabel("epoch")
    plt.ylabel("free_energy_diff")
    plt.savefig(f"{fig_dir}/result_fe_diff.{format}")

    plt.figure()
    plt.plot(metrics["epoch"], metrics["pl_train"], marker=".", label="train")
    plt.plot(metrics["epoch"], metrics["pl_test"], marker=".", label="test")
    plt.xlabel("epoch")
    plt.ylabel("pseudo_likelihood")
    plt.legend()
    plt.savefig(f"{fig_dir}/result_pl.{format}")

    # evaluate accuracy
    sum_train_acc = 0
    sum_test_acc = 0
    for inputs, labels in train_loader:
        inputs, labels = (
            inputs.to(device),
            F.one_hot(labels, num_classes=label_size).to(device),
        )
        preds = model.classification(inputs)
        sum_train_acc += accuracy(inputs, labels).item()

    for inputs, labels in test_loader:
        inputs, labels = (
            inputs.to(device),
            F.one_hot(labels, num_classes=label_size).to(device),
        )
        preds = model.classification(inputs)
        sum_test_acc += accuracy(inputs, labels).item()
    train_acc = sum_train_acc / len(train_loader)
    test_acc = sum_test_acc / len(test_loader)
    print("=================================")
    print(f"train_acc={train_acc}, test_acc={test_acc}")

    writer.close()
