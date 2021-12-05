from Data_Segmentation import *
from Train_Segmentation import *

import argparse

# Parse arguments
# Default arguments
parser = argparse.ArgumentParser()
# Model
parser.add_argument('--model_name', default='Unet')
# Preprocessing parameters
parser.add_argument('--resize', default=256, type=int)
parser.add_argument('--stride', default=5, type=int)
# Optimizer hyperparameters
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--beta_1', default=0.9, type=float)
parser.add_argument('--beta_2', default=0.99, type=float)
parser.add_argument('--bs', default=10, type=int)
# Training hyperparameter
parser.add_argument('--num_epochs', default=10, type=int)
args = parser.parse_args()

# Dataloading
# Defining useful paths     
path_source = "/data2/PETRAW/Training"
path_images = "Images"
path_masks = "GT"

# Define hyperparameters
size = args.resize
stride = args.stride

# Define transformations
transformations = transforms.Compose([
     transforms.ToTensor(),
     transforms.Resize((size,size*2))
])
      
# Create train_dataset
train_samples = [i for i in range (72)]
train_dataset = PETRAW_Dataset(root=path_source,image_folder=path_images,mask_folder=path_masks, 
                        transforms=transformations,
                        n_samples=n_samples,
                        stride=args.stride
                        )
                        
# Create validation dataset
val_samples = [i for i in range (72,90)]
val_dataset = PETRAW_Dataset(root=path_source,image_folder=path_images,mask_folder=path_masks, 
                        transforms=transformations,
                        n_samples=n_samples,
                        stride=args.stride
                        )
                        
print(train_dataset)
print(val_dataset)


# Model training
model_name = args.model_name
# Define hyperparameters
n_epochs = args.num_epochs
lr = args.lr
betas = (args.beta_1, args.beta_2)
batch_size = args.bs

# Define metrics dictionnary
metrics = {'jaccard_score' : jaccard_score}

# Define criterion
criterion = torch.nn.CrossEntropyLoss()

# Define path to save the results
path_results = "./Results"
save_dir = os.path.join(path_results,'{}_{}'.format(model_name,n_samples))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Train model
train_model(model_name, criterion, train_dataset, val_dataset, metrics, n_epochs, batch_size, lr, betas, bpath)
