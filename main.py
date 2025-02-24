import wandb
import os, time
import argparse

from trainer import *
from data import load_data, create_data_loaders


os.environ["TORCH_HOME"] = "/ocean/projects/cis240109p/abollado/.cache"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--group", type=str)
parser.add_argument("--pretrained_path", type=str, default="")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--optim", type=str, default="adam", choices=["adam", "radam", "sgd"])
args = parser.parse_args()



if __name__ == "__main__":

    save_path = f"models/{args.group}/{args.name}"
    if os.path.exists(save_path):
        save_path += "_" + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_path, exist_ok=True)

    # get data
    train_dataset, val_dataset = load_data()
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

    # get model
    group = args.group
    model = load_model(args.model, device, pretrained_path=args.pretrained_path)

    # train model
    wandb.init(
        project="cnn-against-malaria",
        name=args.name,    
    )
    train(model, train_loader, val_loader, epochs=25, group=group, lr=args.lr, optim=args.optim)
    wandb.finish()
    torch.save(model.fc.state_dict(), f"{save_path}/model_fc.pt")

    test(model, output_csv=f"{save_path}/submission.csv")
