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

def evaluate_model(model, test_loader, wd_candidates):

    model.eval()
    # When validating performance, we measure the accuracy on positive matches 
    x = []
    y_preds = []
    y_trues = []
    num_test_batches = int(len(test_loader.dataset) / test_loader.batch_size)
    for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=num_test_batches):
        assert(len(data) == 1)
        mbz_id = data[0, 0]
        wd_id = data[0, 1]
        pred_id, score = model.predict_matches(mbz_id, wd_candidates)

        if score < 0.5:
            pred_id ="None"
        else:
            pred_id = pred_id.item()

        x.append(mbz_id.item())
        y_preds.append(pred_id)
        y_trues.append(wd_id.item())
    
    return x, y_trues, y_preds
