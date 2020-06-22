import sys
import argparse
import torch
import torch.optim as optim
import numpy as np
from src.dataset import ProtoPlanetaryDisks, MNIST
from src.ae_model import *
from src.ae_training import Trainer
from src.utils import count_parameters
import wandb

torch.autograd.set_detect_anomaly(True)

rnd_seed = 13
np.random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
torch.cuda.manual_seed_all(rnd_seed)
#os.environ['PYTHONHASHSEED'] = str(rnd_seed)

# Config #
parser = argparse.ArgumentParser(description='AutoEncoder')
parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                    default=False,
                    help='Load data and initialize models [False]')
parser.add_argument('--machine', dest='machine', type=str, default='local',
                    help='were to is running (local, colab, [exalearn])')

parser.add_argument('--data', dest='data', type=str, default='MNIST',
                    help='data used for training ([MNIST], PPD)')

parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                    help='learning rate [1e-4]')
parser.add_argument('--lr-sch', dest='lr_sch', type=str, default=None,
                    help='learning rate shceduler '+
                    '([None], step, exp, cosine, plateau)')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                    help='batch size [128]')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=100,
                    help='total number of training epochs [100]')
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    default=False, help='Early stoping')

parser.add_argument('--cond', dest='cond', type=str, default='F',
                    help='label conditional AE ([F],T)')
parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=32,
                    help='dimension of latent space [32]')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2,
                    help='dropout for all layers [0.2]')

parser.add_argument('--comment', dest='comment', type=str, default='',
                    help='extra comments')
args = parser.parse_args()

# Initialize W&B project
wandb.init(entity='GeepGen-PPD', project="PPD-AE", tags='VAE')
wandb.config.update(args)
wandb.config.rnd_seed = rnd_seed


# run main program
def run_code():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Load Data #
    if args.data == 'PPD':
        dataset = ProtoPlanetaryDisks()
    elif args.data == 'MNIST':
        dataset = MNIST(args.machine)
    else:
        print('Error: Wrong dataset (MNIST, Proto Planetary Disk)...')
        raise

    if len(dataset) == 0:
        print('No items in training set...')
        print('Exiting!')
        sys.exit()

    print('Dataset size: ', len(dataset))
    # data loaders for training and testing
    train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size,
                                                       shuffle=True,
                                                       test_split=.2,
                                                       random_seed=rnd_seed)
    img_dim = dataset[0][0].shape
    wandb.config.physics_dim = len(dataset.phy_names) if args.data == 'PPD' else 0
    print('Physic dimension: ', wandb.config.physics_dim)

    # Define AE model, Ops, and Train #
    model = TranConv_AutoEncoder(latent_dim=args.latent_dim,
                                img_dim=img_dim[-1])
    wandb.watch(model, log='gradients')

    wandb.config.n_train_params = count_parameters(model)
    print('Summary:')
    print(model)
    print('Num of trainable params: ', wandb.config.n_train_params)
    print('\n')

    # Initialize optimizers
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning Rate scheduler
    if args.lr_sch == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=20,
                                              gamma=0.5)
    elif args.lr_sch == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=0.985)
    elif args.lr_sch == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=50,
                                                         eta_min=1e-5)
    elif args.lr_sch == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=.5,
                                                         verbose=True)
    else:
        scheduler = None

    print('Optimizer    :', optimizer)
    print('LR Scheduler :', scheduler.__class__.__name__)

    # Train model
    print('########################################')
    print('########  Running in %4s  #########' % (device))
    print('########################################')

    trainer = Trainer(model, optimizer, args.batch_size, wandb,
                      scheduler=scheduler, print_every=500,
                      device=device)

    if args.dry_run:
        print('******** DRY RUN ******** ')
        return

    trainer.train(train_loader, test_loader, args.num_epochs,
                  save=True, early_stop=args.early_stop)


if __name__ == "__main__":
    print('Running in: ', args.machine, '\n')
    for key, value in vars(args).items():
        print('%15s\t: %s' % (key, value))

    run_code()
