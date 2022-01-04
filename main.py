import argparse
import sys

import torch
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data import mnist_loader, mnistDataset
from model import cnnModel, train, validation


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.005)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        n_epochs = 10
        
        # Load model and data
        model = cnnModel()
        train_images, train_labels, _, _ = mnist_loader()

        # TODO: conisder adding transform later
        train_set = mnistDataset(train_images, train_labels)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        # TODO: Implement training loop here
        train_losses = train(model, train_loader, criterion=criterion, 
                             optimizer=optimizer, epochs=n_epochs)

        print(range(n_epochs))
        print(np.shape(train_losses))
        print(train_losses)
        plt.figure(figsize=(5, 5))
        plt.plot(range(n_epochs), train_losses)
        plt.grid()
        plt.title('Training loss per epoch')
        plt.show()

        # TODO: Save model
        torch.save(model.state_dict(), 'checkpoint.pth')


    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="../models/checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = cnnModel()
        state_dict = torch.load(args.load_model_from)
        model.load_state_dict(state_dict)

        # Model in inference mode, dropout is off
        model.eval()

        criterion = torch.nn.NLLLoss()
        _, _, test_images, test_labels = mnist_loader()

        # TODO: conisder adding transform later
        test_set = mnistDataset(test_images, test_labels)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

        with torch.no_grad():
            test_loss, accuracy = validation(model, test_loader, criterion)

        print("Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

        # Make sure dropout and grads are on for training
        model.train()

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    