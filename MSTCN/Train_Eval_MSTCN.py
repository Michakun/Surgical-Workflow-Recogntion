# Import useful libraries
from Data_MSTCN import *
from Model_MSTCN import *
import time
import csv
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from torch import optim

# Define criterion
# Criterion for classification (ignore index -100 that is put in the input batches)
ce = nn.CrossEntropyLoss(ignore_index=-100)
mse = nn.MSELoss(reduction='none')
# Tasks weights
weights = [1,2,5,5]

def loss_function(outputs,labels,mask,weights):
    """
    Define the loss function that will be used for the classification task.
    
    Args:
        outputs (tensor): Output of the classification model.
        labels (tensor): Labels associated with the model's output.
        mask (tensor): Tensor indicating which outputs have to be taken into account for the loss calculation.
        weights (list): Weights to put on the different tasks.
    """
    loss=0
    for p in (outputs):
        loss += weights[0]*ce(p.transpose(2, 1).contiguous().view(-1, 30)[:,:3], labels.transpose(2, 1).contiguous().view(-1, 4)[:,0])
        loss += weights[1]*ce(p.transpose(2, 1).contiguous().view(-1, 30)[:,3:16], labels.transpose(2, 1).contiguous().view(-1, 4)[:,1])
        loss += weights[2]*ce(p.transpose(2, 1).contiguous().view(-1, 30)[:,16:23], labels.transpose(2, 1).contiguous().view(-1, 4)[:,2])
        loss += weights[3]*ce(p.transpose(2, 1).contiguous().view(-1, 30)[:,23:30], labels.transpose(2, 1).contiguous().view(-1, 4)[:,3])
        loss += weights[0]*0.15*torch.mean(torch.clamp(mse(F.log_softmax(p[:, :3, 1:], dim=1), F.log_softmax(p.detach()[:, :3, :-1], dim=1)), min=0, max=16)*mask[:, :3, 1:])
        loss += weights[1]*0.15*torch.mean(torch.clamp(mse(F.log_softmax(p[:, 3:16, 1:], dim=1), F.log_softmax(p.detach()[:, 3:16, :-1], dim=1)), min=0, max=16)*mask[:, 3:16, 1:])
        loss += weights[2]*0.15*torch.mean(torch.clamp(mse(F.log_softmax(p[:, 16:23, 1:], dim=1), F.log_softmax(p.detach()[:, 16:23, :-1], dim=1)), min=0, max=16)*mask[:, 16:23, 1:])
        loss += weights[3]*0.15*torch.mean(torch.clamp(mse(F.log_softmax(p[:, 23:30, 1:], dim=1), F.log_softmax(p.detach()[:, 23:30, :-1], dim=1)), min=0, max=16)*mask[:, 23:30, 1:])
    return loss


class Trainer:
    "Class to train a MSTCN or MSTCN++ model"
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes,model_name,causal_conv):
        """
        Args:
            model_name (str): Name of the wanted model.
        """
        if model_name == 'MSTCN++' :
            self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes,causal_conv)
        if model_name == 'MSTCN' :
            self.model = MS_TCN(num_R+1, num_layers_PG, num_f_maps, dim, num_classes)
        
        self.num_classes = num_classes
    
    # Function to train the model
    def train(self, train, val, save_dir, num_epochs, batch_size, metrics, average, lr, betas, weight_decay):
        """
        Args:
            train (BatchGenerator): Training set.
            val (BatchGenerator): Validation set.
            save_dir (str): Path to save training results (csv and best model).
            num_epochs (int) : Number of epochs for training.
            batch_size (int) : Batchsize used for training and validation.
            metrics (dict): Dictionnary that contains the names of the metrics used for performance evalutation.
            average (str): Average method for all metrics calculation.
            lr (float): Learning rate for the optimizer.
            betas (tuple): Betas parameters for the Adam optimizer.
            weight_decay (float): Regularization parameter for the optimizer.
            
        """
        # Initialize time
        since = time.time()
        
        # Initialize best model and best loss
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10
        best_epoch = 0
        
        # Use gpu if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Initialize the log file for training and testing loss and metrics
        fieldnames = ['epoch', 'Train_loss','Validation_loss'] + \
         [f'Train_{m}_phase' for m in metrics.keys()] + \
         [f'Validation_{m}_phase' for m in metrics.keys()] + \
         [f'Train_{m}_step' for m in metrics.keys()] + \
         [f'Validation_{m}_step' for m in metrics.keys()] + \
         [f'Train_{m}_LV' for m in metrics.keys()] + \
         [f'Validation_{m}_LV' for m in metrics.keys()] + \
         [f'Train_{m}_RV' for m in metrics.keys()] + \
         [f'Validation_{m}_RV' for m in metrics.keys()]
        with open(os.path.join(save_dir, 'results.csv'), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr,weight_decay=weight_decay)
        
        # For each epoch
        for epoch in range(num_epochs):
            if epoch%10==0 :
                print('Epoch {}/{}'.format(epoch, num_epochs))
                print('-' * 10)
            # Each epoch has a training and validation phase
            # Initialize batch summary
            batchsummary = {a: [0] for a in fieldnames}
        
            for phase in ['Train','Validation']:
                total_loss=0 # Initialize epoch loss
                n_batches = 0
                # Initialize epoch predictions
                phases = torch.tensor([0])
                steps = torch.tensor([0])
                LV = torch.tensor([0])
                RV = torch.tensor([0])
                # Initialize epoch labels
                GT = torch.tensor([0,0,0,0]).reshape((1,4))
            
                if phase == 'Train':
                    self.model.train() # Set model to training mode
                    batch_gen = train # Define train set
                else:
                    self.model.eval() # Set model to evaluate mode
                    batch_gen = val # Define validation set
                
                # While the samples from the set have not all gone through the batches in input
                while batch_gen.has_next():
                    n_batches+=1
                    # Generate next batch
                    batch_input, batch_target, mask,length_of_sequences = batch_gen.next_batch(batch_size)
                    batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'Train'):
                        predictions = self.model(batch_input)
                        loss = loss_function(predictions,batch_target,mask,weights) # loss calculation for the current batch
                        total_loss += loss.item() # add batch loss to total loss
                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            loss.backward()
                            optimizer.step()

                    # Take predictions from the last MSTCN stage
                    _, predicted_phase = torch.max(predictions[-1][:,0:3,:].data, 1)
                    _, predicted_step = torch.max(predictions[-1][:,3:16,:].data, 1)
                    _, predicted_LV = torch.max(predictions[-1][:,16:23,:].data, 1)
                    _, predicted_RV = torch.max(predictions[-1][:,23:30,:].data, 1)
                
                    # add batch predictions to epoch predictions
                    for i in range (batch_size) : 
                        phases = torch.cat((phases,predicted_phase[i,:length_of_sequences[i]].cpu()))
                        steps = torch.cat((steps,predicted_step[i,:length_of_sequences[i]].cpu()))
                        LV = torch.cat((LV,predicted_LV[i,:length_of_sequences[i]].cpu()))
                        RV = torch.cat((RV,predicted_RV[i,:length_of_sequences[i]].cpu()))
                        GT = torch.cat((GT,batch_target[i,:,:length_of_sequences[i]].T.cpu()),axis=0)
                
                # Calculate metrics for the current epoch
                for name, metric in metrics.items():
                    batchsummary[f'{phase}_{name}_phase']=metric(GT[1:,0], phases[1:], average=average)
                    batchsummary[f'{phase}_{name}_step']=metric(GT[1:,1], steps[1:], average=average)
                    batchsummary[f'{phase}_{name}_LV']=metric(GT[1:,2], LV[1:], average=average)
                    batchsummary[f'{phase}_{name}_RV']=metric(GT[1:,3], RV[1:], average=average)
                    
                # Write epoch results in batch summary
                batchsummary['epoch'] = epoch
                epoch_loss = total_loss/n_batches
                batchsummary[f'{phase}_loss'] = epoch_loss
                if epoch%10==0:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                
                # Reset batch generation
                batch_gen.reset()
            
            if epoch%10==0:
                print(batchsummary)
            
            with open(os.path.join(save_dir, 'results.csv'), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(batchsummary)
                # deep copy the model
                if phase == 'Validation' :
                    if loss<best_loss :
                        best_epoch = epoch
                        best_loss = loss
                    # save models
                    torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".pth")
        
        # Compute and print training duration and print lowest loss
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))    
            
# Useful dictionnaries
d0 = {'Idle': 0, 'Transfer Left to Right': 1, 'Transfer Right to Left': 2}
d1 = {'Block 1 L2R': 0, 'Block 1 R2L': 1,'Block 2 L2R': 2,'Block 2 R2L': 3,'Block 3 L2R': 4,'Block 3 R2L': 5,'Block 4 L2R': 6,'Block 4 R2L': 7,'Block 5 L2R': 8,'Block 5 R2L': 9,'Block 6 L2R': 10,'Block 6 R2L': 11,'Idle': 12}
d2 = {'Catch': 0, 'Drop': 1, 'Extract': 2, 'Hold': 3, 'Idle': 4, 'Insert': 5, 'Touch': 6}
reverse_d0 = {v:k for k,v in d0.items()}
reverse_d1 = {v:k for k,v in d1.items()}
reverse_d2 = {v:k for k,v in d2.items()}
          
class Evaluator:
    "Class to evaluate a MSTCN or MSTCN++ model"
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes,model_name,causal_conv):
        """
        Args:
            model_name (str): Name of the wanted model.
        """
        if model_name == 'MSTCN++' :
            self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes,causal_conv)
        if model_name == 'MSTCN' :
            self.model = MS_TCN(num_R+1, num_layers_PG, num_f_maps, dim, num_classes)
        
        self.num_classes = num_classes
    
    # Function to train the model
    def eval(self, test, save_dir, batch_size, videos):
        """
        Args:
            test (BatchGenerator): Test set.
            save_dir (str): Path to save training results (csv and best model).
            num_epochs (int) : Number of epochs for training.
            batch_size (int) : Batchsize used for training and validation.   
            videos (list) : List of test videos.         
        """
        
        since = time.time()
        
        # Use gpu if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
       
        # Set model to eval mode
        self.model.eval()        
        
        # Initialize video index
        video_index = 0
  
        batch_gen = test   
        
        while batch_gen.has_next():
        
            # Generate next batch
            batch_input,_,_,length_of_sequences = batch_gen.next_batch(batch_size)
            batch_input = batch_input.to(device)
            
            # Get predictions for the current video
            predictions = self.model(batch_input)[-1]
            _, predicted_phase = torch.max(predictions[:,0:3,:].data, 1)
            #print(predicted_phase.shape)
            #print(predicted_phase.cpu().numpy().shape)
            _, predicted_step = torch.max(predictions[:,3:16,:].data, 1)
            _, predicted_LV = torch.max(predictions[:,16:23,:].data, 1)
            _, predicted_RV = torch.max(predictions[:,23:30,:].data, 1)
    
            # Save predictions
            results = pd.DataFrame(columns = ['Frame','Phase', 'Step', 'Verb_Left','Verb_Right'])
            results['Frame'] = [i for i in range (len(predicted_phase[0]))]
            results['Phase'] = [reverse_d0.get(elem.item(),elem.item()) for elem in predicted_phase.cpu().numpy()[0]]
            results['Step'] = [reverse_d1.get(elem.item(),elem.item()) for elem in predicted_step.cpu().numpy()[0]]
            results['Verb_Left'] = [reverse_d2.get(elem.item(),elem.item()) for elem in predicted_LV.cpu().numpy()[0]]
            results['Verb_Right'] = [reverse_d2.get(elem.item(),elem.item()) for elem in predicted_RV.cpu().numpy()[0]]
            
            results.to_csv(os.path.join(save_dir,'{}_Results.txt'.format(videos[video_index])),header=True,index=False,sep='\t')
            
            video_index += 1
        
            
        # Compute and print training duration
        time_elapsed = time.time()-since
        print('-' * 10)
        print('Time of the Task : {:.2f}'.format(time_elapsed))
        print('-' * 10)
