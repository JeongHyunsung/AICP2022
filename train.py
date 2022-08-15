import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from Dataset.preprocess import preprocess

from loss import cross_entropy
from optimizer import sgd


def train_one_epoch(epoch_index, tb_writer, training_loader, model_, loss_fn_, optimizer_, freq):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(training_loader):
        input_batch, target_batch = data
        optimizer_.zero_grad()
        output_batch = model_(input_batch)
        loss = loss_fn_(output_batch, target_batch)
        loss.backward()
        optimizer_.step()
        running_loss += loss.item()
        if i % freq == freq-1:
            last_loss = running_loss / freq
            print("batch {} loss: {}".format(i+1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

    return last_loss


def valid_one_epoch(validation_loader, model_, loss_fn_):
    running_loss = 0

    for i, data in enumerate(validation_loader):
        input_batch, target_batch = data
        output_batch = model_(input_batch)
        loss = loss_fn_(output_batch, target_batch)
        running_loss += loss
    running_loss /= len(validation_loader)
    return running_loss


if __name__ == "__main__":

    # preprocessing
    batch_size = 5
    train_loader, valid_loader, eval_loader, vocab_size = preprocess(batch_size)

    print(next(iter(train_loader)), vocab_size)

    # model, loss function, optimizer, summary writer, epoch setting
    learning_rate, momentum = 0.001, 0.9
    Epochs = 10
    model = Model(batch_size, vocab_size)
    loss_fn = cross_entropy()
    optimizer = sgd(model.parameters(), learning_rate, momentum)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    # training
    best_valid_loss = 100000000
    for epoch in range(Epochs):
        print("EPOCH {}:".format(epoch+1))

        model.train(True)
        avg_train_loss = train_one_epoch(epoch, writer, train_loader, model, loss_fn, optimizer, 200)
        model.train(False)
        avg_valid_loss = valid_one_epoch(valid_loader, model, loss_fn)
        print("LOSS train {} valid {}".format(avg_train_loss, avg_valid_loss))

        writer.add_scalars("Training vs Validation Loss", {"Training" : avg_train_loss, "Validation" : avg_valid_loss}, epoch + 1)
        writer.flush()

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_path = "model_{}_{}".format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)
















