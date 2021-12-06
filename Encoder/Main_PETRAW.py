# Import useful libraries
from Data_PETRAW import *
from Model_PETRAW import *
from Train_Eval_PETRAW import *
import argparse

# Default arguments
parser = argparse.ArgumentParser()
# Task-related arguments
parser.add_argument('--input_type', default='Images')
parser.add_argument('--task', default='multi')
parser.add_argument('--encoder', default='Resnet')
parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--freeze', default=False, type=bool)
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('--hierarchical', default=False, type=bool)
# Optimizer hyperparameters
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--beta_1', default=0.9, type=float)
parser.add_argument('--beta_2', default=0.99, type=float)
# Training hyperparameters
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--num_epochs', default=15, type=int)
parser.add_argument('--undersampling', default=True, type=bool)
# Metrics averaging
parser.add_argument('--metric_avg', default='macro')



args = parser.parse_args()
print(args)

# Dataloading
# Defining useful paths     
path_source = "/"
path_labels = "data2/PETRAW/Training/Procedural_description"
path_images = "data2/PETRAW/Training/Images"
#path_images = "data/michael/Train_samples_Unet"

# Define cumulative video lengths for samplers
folders = sorted(os.listdir(os.path.join(path_source,path_images)))
lengths = np.cumsum([len(os.listdir(os.path.join(path_source,path_images,folders[i]))) 
                     for i in range(len(folders))])
lengths = np.concatenate((np.array([0]), lengths))

# Define encoder & task
input_type = args.input_type
encoder = args.encoder
task = args.task
pretrained = args.pretrained
freeze = args.freeze
train = args.train


# Define transformations (can be changed)
if input_type == 'Images' :
     transformations = transforms.Compose([  
     transforms.Resize((256,512)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
else : 
    transformations = transforms.Compose([
     transforms.ToTensor(),  
     transforms.Resize((256,512))])
      
# Create train & val dataset
if train == True :
    train_samples = [i for i in range (1)]
    train_dataset = PETRAW_Dataset(root=path_source,image_folder=path_images,label_folder=path_labels, 
                        transforms=transformations,
                        n_samples=train_samples,
                        task=task,
                        input_type=input_type)
    print(train_dataset)
    
    val_samples = [i for i in range (1,2)]
    val_dataset = PETRAW_Dataset(root=path_source,image_folder=path_images,label_folder=path_labels, 
                        transforms=transformations,
                        n_samples=val_samples,
                        task=task,
                        input_type=input_type)

if train == False :
    test_samples = [i for i in range (2,3)]
    

# Model creation
hierarchical = args.hierarchical
print(hierarchical)

if task == 'multi':
    model = WRNet_Multi(encoder,hierarchical,input_type,pretrained,freeze,identity)
    print(model)
    if train == False:
        path_params = './unhierarchical.pth'
        model.load_state_dict(torch.load(path_params, map_location='cuda:0')['model'])      
#else : 
#    model = WRNet(encoder,task,input_type,pretrained,freeze,identity)
#    if train == False:
#        path_params = './unhierarchical.pth'
#        model.load_state_dict(torch.load(path_params, map_location='cuda:0')['model'])  

# Model training
# Define hyperparameters
n_epochs = args.num_epochs
lr = args.lr
betas = (args.beta_1, args.beta_2)
batch_size = args.bs
weight_decay = args.weight_decay
undersampling = args.undersampling

# Defining metrics dictionnary
metrics = {'recall_score' : recall_score}
average = args.metric_avg

# Define criterion
# Define criterion weights for LV and RV
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
LV = torch.tensor([0.61836507,2.81087508,1,0.17897085,0.08573446,1.26055737,7.57735043]).to(device)
RV = torch.tensor([0.61836507,2.81087508,1,0.17897085,0.08573446,1.26055737,7.57735043]).to(device)
# Define single task criterion
criterion_phase = torch.nn.CrossEntropyLoss()
criterion_step = torch.nn.CrossEntropyLoss()
criterion_LV = torch.nn.CrossEntropyLoss(weight=LV)
criterion_RV = torch.nn.CrossEntropyLoss(weight=RV)

# Define tasks weights in multitask criterion
criterions = [criterion_phase,criterion_step,criterion_LV,criterion_RV]
weights = [1,2,5,5]

# Define path to save the results
path_results = "/data/michael/Tests/Results"
save_dir = os.path.join(path_results,'{}_{}'.format(encoder,args.input_type)) # One can add other arguments if needed
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Train/Evaluate model
if train == True:
    train_model(model, criterions, weights, train_dataset, val_dataset, metrics, average, n_epochs, 
            batch_size, lr, betas, weight_decay, save_dir, task, undersampling, train_samples, lengths)
if train == False:
    for elem in (test_samples):
        data = PETRAW_Dataset(root=path_source,image_folder=path_images,label_folder=path_labels,
                        transforms=transformations,
                        n_samples=[elem],
                        stride = 1
                        )
        evaluate(model,data,save_dir,task,folders,elem)
