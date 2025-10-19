from torch.autograd import Variable

# ----------------------------------------
#  training
# ----------------------------------------

def train(epoch, optimizer, model,training_data_loader,criterion,local_rank,tb_logger,current_step):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        # with torch.autograd.set_detect_anomaly(True):
        Z, Y, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

        optimizer.zero_grad()
        Z = Variable(Z).float()
        Y = Variable(Y).float()        
        X = Variable(X).float()

        HX = model(Z,Y)

        loss = criterion(HX, X).mean()

        epoch_loss += loss.item()
        if local_rank == 0:
            tb_logger.add_scalar('total_loss', loss.item(), current_step)
        current_step += 1

        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)