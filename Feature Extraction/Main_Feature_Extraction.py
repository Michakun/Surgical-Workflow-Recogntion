# Import useful libraries
from Data_Feature_Extraction import *
from Model_Feature_Extraction import *
from Feature_Extraction import *
import argparse

# Parse arguments
# Default arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_type', default='Images')
parser.add_argument('--task', default='multi')
parser.add_argument('--encoder', default='Resnet')
parser.add_argument('--hierarchical', default=False, type=bool)
parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--freeze', default=False, type=bool)

args = parser.parse_args()

# Dataloading
# Defining useful paths     
path_source = "/"
path_labels = "data2/PETRAW/Training/Procedural_description"
path_images = "data2/PETRAW/Training/Images"
path_params = "./hierarchical.pth"

# List of videos from which features will be extracted
videos = sorted(os.listdir(os.path.join(path_source,path_images)))

# Define transformations
if args.input_type == 'Images' :
    transformations = transforms.Compose([  
     transforms.Resize((256,512)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
else : 
    transformations = transforms.Compose([
     transforms.ToTensor(),  
     transforms.Resize(256)])

# Define model parameters
task = args.task
encoder = args.encoder
hierarchical = args.hierarchical

# Model creation
if task == 'multi':
    model = WRNet_Multi(encoder,hierarchical,args.input_type,identity)
else : 
    model = WRNet(encoder,task,identity)
    
# Load parameters (via params.pth file that has to be in the same directory)
model.load_state_dict(torch.load(path_params, map_location='cuda:0')['model'])
model.eval()

# Define results path
save_dir = "/data/michael/Tests/Results/{}_{}".format(encoder,args.input_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Extract features
for i in range (len(videos)) :
    feature_extraction(model,path_source, path_labels, path_images,transformations, i,save_dir,videos)
