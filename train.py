from matplotlib._api.deprecation import suppress_matplotlib_deprecation_warning
import torch
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from rbm import RBM
from optim import Adam, SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import argparse
import datetime
import os


def imshow(grid, path):
    plt.figure()
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.savefig(path)


def reconst_loss(prob, target):
    target = torch.nn.functional.one_hot(target.long())  # (B, I, 2)
    prob = torch.stack([prob, 1 - prob], dim=-1)  # (B, I, 2)
    loss = target * torch.log(prob + 1e-10)
    loss = loss.sum(dim=[-1, -2])
    return loss.mean()


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser("Restricted Boltzman Machine")
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
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--fig_dir", type=str, default="fig")
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

    # model, optimizer
    device = args.device_id if torch.cuda.is_available() else "cpu"
    model = RBM(
        visible_size=28 * 28,
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

    # train
    H, W = 28, 28

    train_loader_iter = iter(train_loader)
    test_loader_iter = iter(test_loader)
    inputs_train_const, _ = next(train_loader_iter)
    inputs_test_const, _ = next(test_loader_iter)
    inputs_train_const, inputs_test_const = (
        inputs_train_const.to(device),
        inputs_test_const.to(device),
    )

    sum_train_reconst = 0
    sum_test_reconst = 0
    metrics = {
        "epoch": [],
        "fe_train": [],
        "fe_test": [],
        "fe_diff": [],
        "reconst_train": [],
        "reconst_test": [],
    }

    for epoch in tqdm(range(args.num_epoch)):
        sum_train_reconst = 0
        sum_test_reconst = 0
        sum_train_pl = 0
        sum_test_pl = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            # grad = model.grad(inputs, sample_type="pcd_cont")
            # grad = optimizer.calc_grad(grad)
            # model.update(grad)
            model.update(inputs, args.lr, args.sampling_method)
            outputs, prob = model.sample_by_v(inputs, num_gib=args.num_gib)
            sum_train_reconst += reconst_loss(prob, inputs).item()
            sum_train_pl += -model.pseudo_likelihood(inputs).mean().item()

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs, prob = model.sample_by_v(inputs, num_gib=args.num_gib)
                sum_test_reconst += reconst_loss(prob, inputs).item()
                sum_test_pl += -model.pseudo_likelihood(inputs).mean().item()

        outputs, prob = model.sample_by_v(inputs_train_const, num_gib=args.num_gib)
        inputs = inputs_train_const.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
        outputs = outputs.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)

        diff = ((inputs - outputs) + 1) / 2
        images = torch.cat([inputs[:10], outputs[:10], diff[:10]], dim=0)
        writer.add_image("reconst_img", make_grid(images, nrow=10), epoch)

        # calculate mean free-energy
        free_energy_train = model.free_energy(inputs_train_const).mean().item()
        free_energy_test = model.free_energy(inputs_test_const).mean().item()
        free_energy_diff = free_energy_train - free_energy_test
        writer.add_scalar("free_energy_train", free_energy_train, epoch)
        writer.add_scalar("free_energy_test", free_energy_test, epoch)
        writer.add_scalar("free_energy_diff", free_energy_diff, epoch)

        # calculate mean reconst loss
        train_reconst = sum_train_reconst / len(train_loader)
        test_reconst = sum_test_reconst / len(test_loader)
        train_pl = sum_train_pl / len(train_loader)
        test_pl = sum_test_pl / len(test_loader)
        writer.add_scalar("train_reconst", train_reconst, epoch)
        writer.add_scalar("test_reconst", test_reconst, epoch)
        writer.add_scalar("train_pl", train_pl, epoch)
        writer.add_scalar("test_pl", test_pl, epoch)

        # metrics
        metrics["epoch"].append(epoch)
        metrics["fe_train"].append(free_energy_train)
        metrics["fe_test"].append(free_energy_test)
        metrics["fe_diff"].append(free_energy_diff)
        metrics["reconst_train"].append(train_reconst)
        metrics["reconst_test"].append(test_reconst)
        if epoch % 10 == 0:
            print(
                f"train_reconst_loss={train_reconst}, test_reconst_loss={test_reconst}"
            )
            print(f"train_pl={train_pl}, test_pl={test_pl}")

    try:
        inputs, _ = next(train_loader_iter)
        inputs = inputs.to(device)
    except StopIteration:
        train_loader_iter = iter(train_loader)
        inputs, _ = next(train_loader_iter)
        inputs = inputs.to(device)

    outputs, prob = model.sample_by_v(inputs, num_gib=args.num_gib)
    inputs = inputs.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
    outputs = outputs.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
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
    plt.savefig(f"{fig_dir}/result_reconst.{format}")

    plt.figure()
    plt.plot(metrics["epoch"], metrics["fe_train"], label="train", marker=".")
    plt.plot(metrics["epoch"], metrics["fe_test"], label="test", marker=".")
    plt.xlabel("epoch")
    plt.ylabel("free_energy")
    plt.savefig(f"{fig_dir}/result_fe.{format}")

    plt.figure()
    plt.plot(metrics["epoch"], metrics["fe_diff"], marker=".")
    plt.xlabel("epoch")
    plt.ylabel("free_energy_diff")
    plt.savefig(f"{fig_dir}/result_fe_diff.{format}")

    writer.close()
