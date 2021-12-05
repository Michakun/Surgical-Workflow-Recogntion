# Importing useful libraries
from Data_Segmentation import *
from sklearn.metrics import jaccard_score
import segmentation_models_pytorch as smp
import time
import csv

# Function to train the model
def train_model(model_name, criterion, train_dataset, val_dataset, metrics, 
                num_epochs, batchsize, lr, betas, bpath):
    """
    Train the segmentation model.
    Models are pretrained with ImageNet and the encoder backbone is ResNet34.
    These parameters can be changed in the code. Alternatives are not tested here.
    
    Args:
        model_name (str): Name of the segmentation model.
        criterion (func): Loss function used to train the model.
        train_dataset (PETRAW_Dataset): Name of the training set.
        val_dataset (PETRAW_Dataset): Name of the validation set.
        optimizer (func): Gradient descent method used for optimization.
        metrics (dict): Dictionnary that contains the names of the metrics used for performance evalutation.
        num_epochs (int) : Number of epochs for training.
        batchsize (int): Batchsize used for training.
        lr (float): Learning rate for the optimizer.
        betas (tuple): Betas parameters for the Adam optimizer
        bpath (str): Path to save training results (csv and best model).
    """
    
    # Import the wanted model
    if model_name == 'Unet' :
        model = smp.Unet(encoder_name = 'resnet34',encoder_weights='imagenet',in_channels=3,classes=6)
    elif model_name == 'UnetPlusPlus' : 
        model = smp.UnetPlusPlus(encoder_name = 'resnet34',encoder_weights='imagenet',in_channels=3,classes=6)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas = betas)
    
    # Initialize time
    since = time.time()
    
    # Initialize best model and metrics
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_iou = 0
    best_epoch = 0
    
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Validation_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Validation_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'results.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader() 
    
    weights = torch.tensor([1 for i in range (len(train_dataset))])
    weights2 = torch.tensor([1 for i in range (len(val_dataset))])
    # Define samplers
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset)//10,replacement=False)
    sampler2 = torch.utils.data.sampler.WeightedRandomSampler(weights2, len(val_dataset)//5,replacement=False)
    
    # Load training and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=False, sampler=sampler,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, prefetch_factor=2,
               persistent_workers=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, sampler=sampler2,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, prefetch_factor=2,
               persistent_workers=False)
    
    # Epoch loop
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Validation']:
            total_loss=0 # Initialize epoch loss
            n_batches=0 
            if phase == 'Train':
                model.train() # Set model to training mode
                loader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                loader = val_loader

            # Iterate over data.
            for sample in (iter(loader)):
                n_batches += 1
                
                # load samples
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs) # model output calculation
                    loss = criterion(outputs, masks) # loss calculation for the current batch
                    total_loss += loss # add batch loss to total loss
                    y_pred = torch.argmax(outputs,axis=1).data.cpu().numpy().ravel() # Mask prediction from model output
                    y_true = masks.data.cpu().numpy().ravel()                     
                    
                    # calculate metrics for the current batch
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred, average='macro'))
                            
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            
            # Average loss on batches
            epoch_loss = total_loss/n_batches 
            
            # Write results into summary and print epoch loss
            batchsummary['epoch'] = epoch 
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        
        # Average metric on batches for train and validation phases
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        
        # Write results into csv file
        with open(os.path.join(bpath, 'results.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            
            # deep copy the model if the validation iou is better at the current epoch
            if phase == 'Validation':
                best_iou = batchsummary['Validation_jaccard_score']
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save best results into pth file
                results = {'model':best_model_wts, 'iou':best_iou}
                torch.save(results,os.path.join(bpath,'params_{}.pth'.format(epoch)))
                
    # Compute and print training duration and print lowest loss
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))
