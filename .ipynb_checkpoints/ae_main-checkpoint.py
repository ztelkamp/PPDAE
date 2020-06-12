import sys, os
import argparse
import torch
import torch.optim as optim
import numpy as np
#from src.datasets import *
from src.utils import *
from src.ae_model import AutoEncoder
from src.ae_training import Trainer
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
parser.add_argument('--machine', dest='machine', type=str, default='exalearn',
                    help='were to is running (local, colab, [exalearn])')

parser.add_argument('--data', dest='data', type=str, default='MNIST',
                    help='data used for training ([MNIST], ProtoPD)')

parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                    help='learning rate [1e-4]')
parser.add_argument('--lr-sch', dest='lr_sch', type=str, default=None,
                    help='learning rate shceduler '+
                    '([None], step, exp,cosine, plateau)')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=128,
                    help='batch size [128]')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=100,
                    help='total number of training epochs [100]')
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    default=False, help='Early stoping')

parser.add_argument('--cond', dest='cond', type=str, default='T',
                    help='label conditional VAE (F,[T])')
parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=6,
                    help='dimension of latent space [6]')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2,
                    help='dropout for lstm/tcn layers [0.2]')
parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=5,
                    help='kernel size for tcn conv, use odd ints [5]')

parser.add_argument('--comment', dest='comment', type=str, default='',
                    help='extra comments')
args = parser.parse_args()

# Initialize W&B project
wandb.init(project="PPD-AE")
wandb.config.update(args)


# run main program
def run_code():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Load Data #
    if args.data == 'PPD':
        dataset = PPD_data()
    elif args.data == 'MNIST':
        dataset = mnist
    else:
        print('Error: Wrong dataset (MNIST, Proto Planetary Disk)...')
        raise

    if len(dataset) == 0:
        print('No items in training set...')
        print('Exiting!')
        sys.exit()

    # data loaders for training and testing
    train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size,
                                                       shuffle=True,
                                                       test_split=.2,
                                                       random_seed=rnd_seed)
    print('\nTraining lenght: ', len(train_loader) * args.batch_size)
    print('Test lenght    : ', len(test_loader) * args.batch_size)

    wandb.config.physics_dim = len(dataset.phy_names)
    print('Physic dimension: ', wandb.config.physics_dim)

    # Define AE model, Ops, and Train #
    model = AutoEncoder()
    wandb.watch(model, log='gradients')

    print('Summary:')
    wandb.config.n_train_params = count_parameters(model)
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
                      scheduler=scheduler, print_every=50,
                      device=device)

    if args.dry_run:
        print('******** DRY RUN ******** ')
        return

    trainer.train(train_loader, test_loader, args.num_epochs,
                  machine=args.machine, save=True,
                  early_stop=args.early_stop)


if __name__ == "__main__":
    print('Running in: ', args.machine, '\n')
    for key, value in vars(args).items():
        print('%15s\t: %s' % (key, value))

    run_code()
