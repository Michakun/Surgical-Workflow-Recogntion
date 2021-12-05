# Import useful libraries
from Data_MSTCN import *
from Model_MSTCN import *
from Train_Eval_MSTCN import *
import argparse

# Parse arguments
# Default arguments
parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument('--model_name', default='MSTCN++')
parser.add_argument('--num_layers_PG', default=11, type=int)
parser.add_argument('--num_layers_R', default=10, type=int)
parser.add_argument('--num_f_maps', default=64, type=int)
parser.add_argument('--causal_conv', default=True, type=bool)
parser.add_argument('--combination', default='Concatenate')
parser.add_argument('--training', default=False, type=bool)
# Optimizer hyperparameters
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--beta_1', default=0.9, type=float)
parser.add_argument('--beta_2', default=0.99, type=float)
parser.add_argument('--bs', default=2, type=int)
# Training hyperparameters
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--metric_avg', default='macro')

# Need input
parser.add_argument('--num_R', type=int)
parser.add_argument('--input_type')

args = parser.parse_args()

# Defining useful paths
path_source = "/"
path_features_0 = "/data2/PETRAW/Training/Kinematic/"
path_features_1 = "/data/michael/Features/CV1/VGG/"
path_features_2 = "/data/michael/Features/CV1/97iou/"
path_labels = "data2/PETRAW/Training/Procedural_description/"

input_type = args.input_type
combination = args.combination

if input_type =='Kinematic':
    path_features = path_features_0
    dim = 28

elif input_type =='Video':
    path_features = path_features_1
    dim = 2048
    
elif input_type =='Segmentation':
    path_features = path_features_2
    dim = 2048

elif input_type =='KV':
    path_features = [path_features_0,path_features_1]
    dim = 2076

elif input_type =='VS':
    path_features = [path_features_1,path_features_2]
    if combination =='Add':
        dim = 2048
    else:
        dim = 4096

elif input_type =='KVS':
    path_features = [path_features_0,path_features_1,path_features_2]
    if combination =='Add':
        dim = 2076
    else:
        dim = 4124

else : 
    print('Invalid input type')
        
# Load train and validation datasets        
train = BatchGenerator(path_source,path_labels,path_features,args.input_type,args.combination)
train.read_data('./train_split.txt')

val = BatchGenerator(path_source,path_labels,path_features,args.input_type,args.combination)
val.read_data('./val_split.txt')

test = BatchGenerator(path_source,path_labels,path_features,args.input_type,args.combination)
test.read_data('./val_split.txt')

videos = [elem.split('.')[0] for elem in sorted(os.listdir(path_features))][:60]

# Define the model
# Define model hyperparameters
n_classes = [3,13,7,7]
model_name = args.model_name
num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps
causal_conv = args.causal_conv

# Create the model
training = args.training
if training == True:
    mstcn = Trainer(num_layers_PG,num_layers_R,num_R,num_f_maps,dim,n_classes,model_name,causal_conv)
elif training == False:
    mstcn = Evaluator(num_layers_PG,num_layers_R,num_R,num_f_maps,dim,n_classes,model_name,causal_conv)
    path_params = './MSTCN++_0.pth'
    # Load parameters (via params.pth file that has to be in the same directory)
    mstcn.model.load_state_dict(torch.load(path_params, map_location='cpu'))
    mstcn.model.eval()

# Training the model
# Define metrics
metrics = {'recall':recall_score}
average = args.metric_avg

# Define hyperparameters
n_epochs = args.num_epochs
lr = args.lr
betas = (args.beta_1, args.beta_2)
batch_size = args.bs
weight_decay = args.weight_decay

# Define directory to save the results
save_dir = '/data/michael/Tests/Results/MSTCN/{}_{}_{}'.format(input_type,model_name,num_R) # One can add other arguments if needed
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Train/Evaluate the model
if training == True:
    mstcn.train(train,val,save_dir,n_epochs,batch_size, metrics, average, lr, betas, weight_decay)
elif training == False:
    batch_size = 1
    mstcn.eval(test,save_dir,batch_size,videos)
