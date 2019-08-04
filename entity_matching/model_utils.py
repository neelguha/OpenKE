# General functions for training, evaluating, and handling entity matching models in pytorch 

import torch, time

from entity_matching.data_utils import get_negative_samples
from tqdm import tqdm 



def train(model, optimizer, loss_fn, train_loader, val_loader, wd_candidates, neg_samples_per, device, log_interval, num_epochs=100, valid_steps=100): 
    ''' Trains model ''' 

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()

        epoch_loss = 0.0
        num_train_batches = int(len(train_loader.dataset) / train_loader.batch_size)
        # Iterate through batches
        for batch_idx, (data, target) in enumerate(train_loader):
            # get negative samples 
            neg_x, neg_y = get_negative_samples(data, neg_samples_per)

            # concatenate 
            x = torch.cat([data, neg_x]).to(device)
            y = torch.squeeze(torch.cat([target, neg_y]).long()).to(device)

            optimizer.zero_grad()

            output = model(x)
            loss = loss_fn(output, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        if epoch % log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f} \tTime: {:.2f} seconds'.format(epoch, loss.item(), time.time() - t0))
        
        if epoch+1 % valid_steps == 0:
            model.eval()
            # When validating performance, we measure the accuracy on positive matches 
            correct = 0.0
            incorrect = 0.0 
            num_val_batches = int(len(val_loader.dataset) / val_loader.batch_size)
            for batch_idx, (data, target) in tqdm(enumerate(val_loader), total=num_val_batches):
                assert(len(data) == 1)
                mbz_id = data[0, 0]
                pred_id, score = model.predict_matches(mbz_id, wd_candidates)
                #print(pred_id, score)
                if pred_id == data[0, 1]:
                    correct += 1
                else: 
                    incorrect += 1
            accuracy = correct / (correct + incorrect)
            
            print('Validation Epoch {}. Accuracy: {:.2f}'.format(epoch, accuracy))
        