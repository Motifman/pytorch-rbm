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
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--grad_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--fig_dir", type=str, default="fig")
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--device_id", type=str, default="cuda:0")
    parser.add_argument("--num_gib", type=int, default=1000)
    parser.add_argument("--result_format", type=str, default="pdf")
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
            transforms.Lambda(lambda x: torch.where(x < 0.5, 0.0, 1.0)),
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
        k_cd=args.k_cd,
        device=device,
    ).to(device)
    model.requires_grad_(False)  # no-use pytorch BP
    if args.optim == "adam":
        optimizer = Adam(model.state_dict(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = SGD(model.state_dict(), lr=args.lr)
    else:
        NotImplementedError()

    # train
    H, W = 28, 28
    itr = 0

    train_loader_iter = iter(train_loader)
    test_loader_iter = iter(test_loader)
    inputs_train_const, _ = next(train_loader_iter)
    inputs_test_const, _ = next(test_loader_iter)
    train_loader_iter = iter(train_loader)
    test_loader_iter = iter(test_loader)

    sum_train_reconst = 0
    sum_test_reconst = 0
    metrics = {
        "step": [],
        "fe_train": [],
        "fe_test": [],
        "fe_diff": [],
        "reconst_train": [],
        "reconst_test": [],
    }

    for step in tqdm(range(args.grad_steps)):
        try:
            inputs, _ = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            inputs, _ = next(train_loader_iter)

        inputs = inputs.to(device)
        grad = model.grad(inputs, sample_type="pcd")
        grad = optimizer.calc_grad(grad)
        model.update(grad)

        try:
            inputs_test, _ = next(test_loader_iter)
        except StopIteration:
            test_loader_iter = iter(test_loader_iter)
            inputs_test, _ = next(test_loader_iter)

        # reconst error
        outputs, prob = model.sample_by_v(inputs, num_gib=args.num_gib)
        train_reconst = (inputs * torch.log(prob)).sum(-1).mean()
        sum_train_reconst += train_reconst
        outputs_test, prob_test = model.sample_by_v(inputs_test, num_gib=args.num_gib)
        test_reconst = (inputs * torch.log(prob_test)).sum(-1).mean()
        sum_test_reconst += test_reconst

        itr += 1

        if step % args.log_interval:
            # inputs, reconst, diff images
            outputs, prob = model.sample_by_v(inputs_train_const, num_gib=args.num_gib)
            inputs = inputs_train_const.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
            outputs = outputs.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
            diff = ((inputs - outputs) + 1) / 2
            images = torch.cat([inputs[:10], outputs[:10], diff[:10]], dim=0)
            writer.add_image("reconst_img", make_grid(images, nrow=10), step)

            # calculate mean free-energy
            free_energy_train = model.free_energy(inputs_train_const).mean()
            free_energy_test = model.free_energy(inputs_test_const).mean()
            free_energy_diff = free_energy_train - free_energy_test
            writer.add_scalar("free_energy_train", free_energy_train, step)
            writer.add_scalar("free_energy_test", free_energy_test, step)
            writer.add_scalar("free_energy_diff", free_energy_diff, step)

            # calculate mean reconst loss
            train_reconst = sum_train_reconst / itr
            test_reconst = sum_test_reconst / itr
            writer.add_scalar("train_reconst", train_reconst, step)
            writer.add_scalar("test_reconst", test_reconst, step)
            itr = 0

            # metrics
            metrics["step"].append(step)
            metrics["fe_train"].append(free_energy_train)
            metrics["fe_test"].append(free_energy_test)
            metrics["fe_diff"].append(free_energy_diff)
            metrics["reconst_train"].append(train_reconst)
            metrics["reconst_test"].append(test_reconst)

    # result pdf
    try:
        inputs, _ = next(train_loader_iter)
    except StopIteration:
        train_loader_iter = iter(train_loader)
        inputs, _ = next(train_loader_iter)
    outputs, prob = model.sample_by_v(inputs, num_gib=args.num_gib)

    inputs = inputs.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
    outputs = outputs.reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
    diff = ((inputs - outputs) + 1) / 2

    images = torch.cat([inputs[:10], outputs[:10], diff[:10]], dim=0)
    grid = make_grid(images, nrow=10)
    format = args.result_format
    fig_dir = f"./{args.fig_dir}"
    os.makedirs(fig_dir, exist_ok=True)
    path = f"{fig_dir}/reconst_img.{format}"
    imshow(grid, path)

    plt.figure()
    plt.plot(metrics["step"], metrics["reconst_train"], label="train", marker=".")
    plt.plot(metrics["step"], metrics["reconst_test"], label="test", marker=".")
    plt.xlabel("step")
    plt.ylabel("reconst_loss")
    plt.savefig(f"{fig_dir}/result_reconst.{format}")

    plt.figure()
    plt.plot(metrics["step"], metrics["fe_train"], label="train", marker=".")
    plt.plot(metrics["step"], metrics["fe_test"], label="test", marker=".")
    plt.xlabel("step")
    plt.ylabel("free_energy")
    plt.savefig(f"{fig_dir}/result_fe.{format}")

    plt.figure()
    plt.plot(metrics["step"], metrics["fe_diff"], marker=".")
    plt.xlabel("step")
    plt.ylabel("free_energy_diff")
    plt.savefig(f"{fig_dir}/result_fe_diff.{format}")

    writer.close()
