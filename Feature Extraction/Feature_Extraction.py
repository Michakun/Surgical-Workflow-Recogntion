# Import useful libraries
from Data_Feature_Extraction import *
from Model_Feature_Extraction import *
import time
from torch.utils.data import DataLoader

def feature_extraction(model,path_source, path_labels, path_images,transformations, video_index,bpath,videos) :
    """
    Extract video features given an encoder for post processing in a TCN.
    
    Args:
        model : Workflow Recognition model.
        video_index (int): Index of the desired video in the list of videos.
        videos (list) : List of test videos.
        bpath (str): Path to save the results.
        interpolation (str) : Type of interpolation used to infer the segmentation on a full image.
    """ 
    # Set model to eval mode
    model.eval()
    
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #Initialize time
    time_start = time.time()
    
    # Load data
    data = PETRAW_Video(root=path_source,image_folder=path_images, 
                        transforms=transformations,
                        n_sample=video_index,
                        task='multi',
                        stride = 1
                        )

    data_loader = DataLoader(data, batch_size=10, shuffle=False)
                        
    print('DONE')
    
    # Initialize features tensor
    features = []
    
    # For each frame, concatenate extracted features
    for sample in iter(data_loader) : 
        outputs = model(sample['image'].float().to(device))
        features.append(outputs.detach().cpu().numpy())
    
    # Save features into dedicated folder
    features = np.concatenate(features)
    np.savez(os.path.join(bpath,'features'+str(videos[video_index])+'.npz'),features)
    
    # Computational time
    time_elapsed = time.time() - time_start
    print(time_elapsed)
