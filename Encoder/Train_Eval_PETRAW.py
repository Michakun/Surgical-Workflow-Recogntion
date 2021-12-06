# Import useful libraries
from Data_PETRAW import *
from sklearn.metrics import recall_score
import segmentation_models_pytorch as smp
import time
import csv
import copy

# Define criterion
# Criterion for classification
def loss_function(outputs,labels,criterions,weights,task):
    """
    Define the loss function that will be used for the classification task.
    
    Args:
        outputs (tensor): Output of the classification model.
        labels (tensor): Labels associated with the model's output.
        weights (array): Weights for the multitask classification model.
        task (str): Task to be performed by the classification model.
    """
    
    loss = 0
    new_weigths = weights
    # Define final criterion and weights according to the wanted task
    if task == 'phase':
        new_criterions = criterions[0]
    elif task == 'step':
        new_criterions = criterions[1]
    elif task == 'LV':
        new_criterions = criterions[2]
    elif task == 'RV':
        new_criterions = criterions[3]
     
    # Loss calculation
    if task == 'multi':
        for i in range (len(new_weigths)) : 
            loss += new_weigths[i]*criterions[i](outputs[i],labels[:,i])
    else : 
        loss += new_criterions(outputs,labels)
    return loss

# Function to train the classification model
def train_model(model, criterions, weights, train_dataset, val_dataset, metrics, average, num_epochs, 
                batchsize, lr, betas, weight_decay, bpath, task, undersampling, n_samples, video_lengths):
    """
    Train the classification model.
    
    Args:
        model (str): Classification model.
        criterions (list): Criterions used to define the loss function used to train the model.
        weights (arr) : Weights used for the multitask classification model.
        train_dataset (PETRAW_Dataset): Name of the training set.
        val_dataset (PETRAW_Dataset): Name of the validation set.
        metrics (dict): Dictionnary that contains the names of the metrics used for performance evalutation.
        average (str): Average method for all metrics calculation.
        num_epochs (int) : Number of epochs for training.
        batchsize (int): Batchsize used for training.
        lr (float): Learning rate for the optimizer.
        betas (tuple): Betas parameters for the Adam optimizer.
        weight_decay (float): Parameter used for regularization.
        bpath (str): Path to save training results (csv and best model).
        task (str): Task to be performed by the classification model.
        undersampling (bool): Defines if undersampling is performed.
        n_samples (list) : List of sequences' indexes used in the training set.
        video_lengths (list) : Aggregated lengths of sequences used in the training set.
    """
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas = betas, weight_decay = weight_decay)
    
    # Initialize time
    since = time.time()
    
    # Initialize best model and loss
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_epoch = 0
    
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize the results file for training and testing loss and metrics
    if task == 'multi':
        fieldnames = ['epoch', 'Train_loss', 'Validation_loss'] + \
        [f'Train_{m}_phase' for m in metrics.keys()] + \
        [f'Validation_{m}_phase' for m in metrics.keys()] + \
        [f'Train_{m}_step' for m in metrics.keys()] + \
        [f'Validation_{m}_step' for m in metrics.keys()] + \
        [f'Train_{m}_LV' for m in metrics.keys()] + \
        [f'Validation_{m}_LV' for m in metrics.keys()] + \
        [f'Train_{m}_RV' for m in metrics.keys()] + \
        [f'Validation_{m}_RV' for m in metrics.keys()]
    else:
        fieldnames = ['epoch', 'Train_loss', 'Validation_loss'] + \
        [f'Train_{m}_{task}' for m in metrics.keys()] + \
        [f'Validation_{m}_{task}' for m in metrics.keys()] 
    with open(os.path.join(bpath, 'results.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader() 
        
    # Define training set samplers
    # Define sample weights for dataloader
    sample_weights = torch.tensor(np.load('weights.npz')['arr_0']).ravel()
    BV_weights = []
	
    for i in range (len(n_samples)):
        BV_weights += [sample_weights[j].item() for j in range (len(sample_weights))][video_lengths[n_samples[i]]:video_lengths[n_samples[i]+1]]
    BV_weights = torch.tensor(BV_weights)
    print(BV_weights.shape)
    
    # Define samplers
    BV_sampler = torch.utils.data.sampler.WeightedRandomSampler(BV_weights, len(train_dataset)//20,replacement=False)
    if undersampling == True :
        samplers = [None,None,BV_sampler,BV_sampler,BV_sampler]
    else : 
        samplers = [None,None,None,None,None]

    
    # Load training (according to the task) and validation sets
    if task == 'phase':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=False, sampler=samplers[0],
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, prefetch_factor=2,
               persistent_workers=False)
    if task == 'step':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=False, sampler=samplers[1],
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, prefetch_factor=2,
               persistent_workers=False)
    if task == 'LV':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=False, sampler=samplers[2],
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, prefetch_factor=2,
               persistent_workers=False)
    if task == 'RV':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=False, sampler=samplers[3],
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, prefetch_factor=2,
               persistent_workers=False)
    if task == 'multi':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=False, sampler=samplers[4],
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None, prefetch_factor=2,
               persistent_workers=False)
        
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, sampler=None,
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
            # Initialize epoch predictions
            singletask = torch.tensor([0])
            phases = torch.tensor([0])
            steps = torch.tensor([0])
            LV = torch.tensor([0])
            RV = torch.tensor([0])
            # Initialize epoch labels
            if task == 'multi': 
                GT = torch.tensor([0,0,0,0]).reshape((1,4))
            else : 
                GT = torch.tensor([0])
            
            if phase == 'Train':
                model.train() # Set model to training mode
                loader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                loader = val_loader

            # Iterate over data.
            for sample in (iter(loader)):
                n_batches += 1
                
                # Load samples
                inputs = sample['image'].to(device)
                labels = sample['labels'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    
                    #print(criterions)
                    #print(weights)
                    #print(outputs[0])
                    #print(labels[:,0])
                    loss = loss_function(outputs, labels, criterions, weights, task) # loss calculation for the current batch
                    total_loss += loss # add batch loss to total loss
                    # batch predictions for multitask classification
                    if task=='multi':
                        y_pred = [(torch.argmax(elem,axis=1).data.cpu()) for elem in outputs]
                        # add batch predictions to epoch predictions
                        phases = torch.cat((phases,y_pred[0]))
                        steps = torch.cat((steps,y_pred[1]))
                        LV = torch.cat((LV,y_pred[2]))
                        RV = torch.cat((RV,y_pred[3]))  
                    else:
                        y_pred = torch.argmax(outputs,axis=1).data.cpu()
                        singletask = torch.cat((singletask,y_pred))
                    # add batch labels to epoch labels
                    y_true = labels.data.cpu()
                    GT = torch.cat((GT,y_true),axis=0)
                    
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                        
            # Calculate metrics for the current epoch
            for name, metric in metrics.items():
                if task=='multi':
                    batchsummary[f'{phase}_{name}_phase']=metric(GT[1:,0], phases[1:], average=average)
                    batchsummary[f'{phase}_{name}_step']=metric(GT[1:,1], steps[1:], average=average)
                    batchsummary[f'{phase}_{name}_LV']=metric(GT[1:,2], LV[1:], average=average)
                    batchsummary[f'{phase}_{name}_RV']=metric(GT[1:,3], RV[1:], average=average)
                else:
                    batchsummary[f'{phase}_{name}_{task}']=metric(GT[1:], singletask[1:], average=average)
            
            # add info to batch summary
            batchsummary['epoch'] = epoch 
            epoch_loss = total_loss/n_batches
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        
        print(batchsummary)
        
        # Write epoch results in the results file
        with open(os.path.join(bpath, 'results.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Validation' :
                if loss < best_loss : 
                    best_epoch = epoch
                    best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save results into pth file
                results = {'model':best_model_wts, 'loss':loss}
                torch.save(results,os.path.join(bpath,'params_{}.pth'.format(epoch)))
                
    # Compute and print training duration and print lowest loss
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    print('best_epoch : {}'.format(best_epoch))

from torch.utils.data import DataLoader

# Useful dictionnaries
d0 = {'Idle': 0, 'Transfer Left to Right': 1, 'Transfer Right to Left': 2}
d1 = {'Block 1 L2R': 0, 'Block 1 R2L': 1,'Block 2 L2R': 2,'Block 2 R2L': 3,'Block 3 L2R': 4,'Block 3 R2L': 5,'Block 4 L2R': 6,'Block 4 R2L': 7,'Block 5 L2R': 8,'Block 5 R2L': 9,'Block 6 L2R': 10,'Block 6 R2L': 11,'Idle': 12}
d2 = {'Catch': 0, 'Drop': 1, 'Extract': 2, 'Hold': 3, 'Idle': 4, 'Insert': 5, 'Touch': 6}
reverse_d0 = {v:k for k,v in d0.items()}
reverse_d1 = {v:k for k,v in d1.items()}
reverse_d2 = {v:k for k,v in d2.items()}


# Function to evaluate a classification model on a test video
def evaluate(model,test_dataset,bpath,task,videos,video_index) :
    """
    Evaluate the classification model.
    
    Args:
        model (str): Classification model.
        video_index (int): Index of the desired video in the list of test videos.
        videos (list) : List of test videos.
        bpath (str): Path to save the results.
        task (str): Task to be performed by the classification model.
    """
    
    
    # Set model to eval mode
    model.eval()
    
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize time
    time_start = time.time()

    # Load data
    data_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    # Initialize predictions
    singletask = torch.tensor([0])
    phases = []
    steps = []
    LV = []
    RV = []   
    
    for sample in iter(data_loader) : 
        # frame prediction
        outputs = model(sample['image'].float().to(device))
        y_pred = [(torch.argmax(elem,axis=1).data.cpu()) for elem in outputs]
        
        # predictions for multitask classification
        y_pred = [(torch.argmax(elem,axis=1).data.cpu()) for elem in outputs]
        # add frame prediction to whole video prediction
        phases.append(y_pred[0].detach().cpu().numpy())
        steps.append(y_pred[1].detach().cpu().numpy())
        LV.append(y_pred[2].detach().cpu().numpy())
        RV.append(y_pred[3].detach().cpu().numpy())
    
    # Final predictions
    phases = np.concatenate(phases)
    steps=np.concatenate(steps)
    LV = np.concatenate(LV)
    RV = np.concatenate(RV)
    
    # Save predictions into csv file
    results = pd.DataFrame(columns = ['Frame','Phase', 'Step', 'Verb_Left','Verb_Right'])
    results['Frame'] = [i for i in range (len(phases))]
    results['Phase'] = [reverse_d0.get(elem,elem) for elem in phases]
    results['Step'] = [reverse_d1.get(elem,elem) for elem in steps]
    results['Verb_Left'] = [reverse_d2.get(elem,elem) for elem in LV]
    results['Verb_Right'] = [reverse_d2.get(elem,elem) for elem in RV]
    results.to_csv(os.path.join(bpath,'{}_Results_Task.txt'.format(videos[video_index])),header=True,index=False,sep='\t')
    
    # Compute and print training duration
    time_elapsed = time.time()-time_start
    print(time_elapsed)
