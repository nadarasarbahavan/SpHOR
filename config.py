# ----------------------
# PROJECT ROOT DIR
# ----------------------
project_root_dir = ''

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
exp_root = project_root_dir+ ''        # directory to store experiment output (checkpoints, logs, etc)
save_dir = project_root_dir+ ''   # Evaluation save dir

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = project_root_dir+ '/savedir/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
root_criterion_path = project_root_dir+ '/savedir/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------
cifar_10_root =    ''                                      # CIFAR10
cifar_100_root =   ''                                      # CIFAR100
cub_root =        ''                                               # CUB
aircraft_root =   ''                    # FGVC-Aircraft
pku_air_root =   ''                                 # PKU-AIRCRAFT-300
car_root =    ''                             # Stanford Cars
meta_default_path = '' # Stanford Cars Devkit
svhn_root =      ''                                           # SVHN
tin_train_root_dir = '/tinyimagenet/train'        # TinyImageNet Train
tin_val_root_dir = '/tinyimagenet/val/images'     # TinyImageNet Val

# ----------------------
# FGVC / IMAGENET OSR SPLITS
# ----------------------
osr_split_dir = project_root_dir+'/data/open_set_splits'



inat_2021_root = ''
inat21_osr_splits = ''
places_supervised_path = ''
imagenet_moco_path = ''
places_moco_path =''
imagenet_supervised_path =''
mnist_root =''                                            # MNIST

