### YOUR CODE HERE
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from Network import MyNetwork
from os import path
from DataLoader import parse_data, load_data

INPUT_SHAPE = (32, 32, 3)

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs, name=None, input_shape=INPUT_SHAPE):
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.configs = configs
        self.input_shape = input_shape
        self.name = name

        configs.input_shape = input_shape
        self.network = MyNetwork(configs).to(self._device)

        self.configs.silent = self.configs.wandb if self.configs.wandb else self.configs.silent

    def train(self, x_train=None, y_train=None, x_valid=None, y_valid=None):

        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        valid_transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Import Data if None
        if x_train is not None:
            if self.configs.validation:
                if x_valid is None:
                    train_data = parse_data(x_train, y_train, shape=self.input_shape, batch_size=self.configs.batch_size, transform=transform, train_ratio=1)
                else:
                    train_data = parse_data(x_train, y_train, shape=self.input_shape, batch_size=self.configs.batch_size, transform=transform, train_ratio=1)
                    train_data = parse_data(x_valid, y_valid, shape=self.input_shape, batch_size=self.configs.batch_size, transform=transform, train_ratio=1)
            else: 
                train_data = parse_data(x_train, y_train, shape=self.input_shape, batch_size=self.configs.batch_size, transform=transform, train_ratio=1)
        else:
            train_data = torchvision.datasets.CIFAR10(root=self.configs.data_directory, train=True, download=True, transform=transform)
            train_data = torch.utils.data.DataLoader(train_data, batch_size=self.configs.batch_size, shuffle=True, num_workers=2)

            if self.configs.validation:
                valid_data = torchvision.datasets.CIFAR10(root=self.configs.data_directory, train=False, download=True, transform=valid_transform)
                valid_data = torch.utils.data.DataLoader(valid_data, batch_size=self.configs.batch_size, shuffle=False, num_workers=2)

        # Wandb setup
        if self.configs.wandb:
            # Initialize Model Name
            name = self.name if self.name is not None else 'MyModel'
            wandb.init(name=name, config=self.configs, dir=self.configs.log_directory, project="DL-project")
            wandb.watch(self.network)

        # Number of batches per epoch.  Used for logging to normalize data for epochs
        num_steps_per_epoch = len(train_data)

        # Optimizer
        if self.configs.adam: optimizer = torch.optim.Adam(self.network.parameters(), lr=self.configs.learning_rate,               weight_decay=self.configs.weight_decay)
        else:                 optimizer = torch.optim.SGD( self.network.parameters(), lr=self.configs.learning_rate, momentum=0.9, weight_decay=self.configs.weight_decay)

        # Step Scheduler
        if self.configs.step_schedule:
            if self.configs.cosine: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.configs.epochs)
            else:                   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

        # Criterion
        loss = torch.nn.CrossEntropyLoss()

        if self.configs.wandb:
            wandb.log({"training accuracy":1/self.configs.num_classes}, step=0)
            if self.configs.validation: wandb.log({'validation accuracy':1/self.configs.num_classes, "generalization":1}, step=0)

        if not self.configs.silent: print("--- Training Start ---")
        global_step = 0
        # Epoch Loop
        for i in range(self.configs.epochs):
            total_loss = 0.0
            total_correct = 0
            total_data = 0
            self.network.train()
            # Training Loop
            for img, label in train_data:
                # Set X and Y to Device
                img, label = img.to(self._device), label.to(self._device)

                # Compute network logit
                logit = self.network(img)
                # Compute loss
                loss_val = loss(logit, label)
                # Compute total correct predictions
                total_correct += np.sum(np.argmax(logit.detach().cpu().numpy(), axis=1) == label.cpu().numpy())
                total_data += img.shape[0]

                # Grad Step
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                global_step += 1
                total_loss += loss_val.item()

                # Plot loss
                if self.configs.wandb: wandb.log({"epoch":global_step/num_steps_per_epoch, "loss": loss_val.item()}, step=global_step)

            # Step Learning Rate
            if self.configs.step_schedule:
                if self.configs.cosine: scheduler.step()
                else:                   scheduler.step(total_loss)

            # Compute Train Accuracy
            if self.configs.validation: 
                train_accuracy = total_correct/total_data
                valid_accuracy = self.evaluate(test_data=valid_data)
                generalization = valid_accuracy/train_accuracy
            else:
                train_accuracy = total_correct/len(x_train)

            # Print if no logger
            if self.configs.wandb:
                import wandb
                wandb.log({"training accuracy":train_accuracy}, step=global_step)
                if self.configs.validation: wandb.log({'validation accuracy':valid_accuracy, "generalization":generalization}, step=global_step)
                if self.configs.step_schedule: wandb.log({'learning rate':optimizer.param_groups[0]['lr']}, step=global_step)
            elif not self.configs.silent: 
                print("Epoch: {:4d} -- Training Loss: {:10.6f} -- Accuracy: {:10.6f}".format(i, total_loss, train_accuracy), end='' if self.configs.validation else '\n')
                if self.configs.validation: print(" -- Generalization: {:10.6f}".format(generalization))

            if self.configs.save_interval is not None and self.configs.save_interval > 0 and i % self.configs.save_interval == 0 and i+1 < self.configs.epochs:
                self.save('i{}'.format(i))

        if self.configs.save_interval is not None and self.configs.save_interval >= 0:
            self.save()
            if not self.configs.silent: print("Saved: True")
        elif not self.configs.silent: print("Saved: False")
        if not self.configs.silent: print("--- Training Complete ---")

    def evaluate(self, x=None, y=None, test_data=None):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if self.configs.architecture == 'efficient': transform.transforms.insert(0,transforms.Resize(224))

        if test_data is None:
            test_data = torchvision.datasets.CIFAR10(root=self.configs.data_directory, train=False, download=True, transform=transform)
            test_data = torch.utils.data.DataLoader(test_data, batch_size=self.configs.batch_size, shuffle=False, num_workers=2)
        total_records = 0

        # Set network to eval mode
        self.network.eval()
        
        # Get predictions and return the percent correct as a decimal
        if not self.configs.silent: print('--- Testing Start ---')
        total_correct = 0
        for x, label in test_data:
            img, label = x.to(self._device), label.to(self._device)
            p = self.predict(img)
            total_records += p.shape[0]
            total_correct += np.sum(p == label.detach().cpu().numpy())
        if not self.configs.silent: print('--- Testing Complete ---')

        accuracy = total_correct/total_records
        if not self.configs.silent: print('Accuracy:',accuracy)
        return accuracy
        
    def predict(self, x):
        # Check for invalid shapes of input
        # Ensure the shape of the input
        return np.argmax(self._predict_logit(x), axis=1)

    def predict_prob(self, inputs):
        if not isinstance(inputs, np.ndarray):
            raise Exception('Invalid Type:{} | Numpy Array Expected'.format(type(x)))

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if self.configs.architecture == 'efficient': transform.transforms.insert(0,transforms.Resize(224))

        # Set network to eval mode
        self.network.eval()

        logits = []

        # Get predictions and return the percent correct as a decimal
        if not self.configs.silent: print('--- Prediction Start ---')
        for x in inputs:
            x = transform(x)
            img = x.reshape(-1,3,32,32).to(self._device)
            logits.append(self._predict_logit(img))
        if not self.configs.silent: print('--- Prediction Complete ---')

        logits = np.concatenate(logits,axis=0)
        predictions = torch.nn.functional.softmax(torch.tensor(logits),0).numpy()
        return predictions

    def _predict_logit(self, x):
        flatten_shape = (self.input_shape[0]*self.input_shape[1]*self.input_shape[2],)
        process_shape = (self.input_shape[2],*self.input_shape[:2])
        if self.configs.architecture == 'efficient': process_shape = (self.input_shape[2],224,224)

        # Check for invalid shapes of input for numpy and torch
        # - Numpy array can be in flattened form as well
        # - Torch tensor is assumed to have been processed
        if isinstance(x, np.ndarray):
            raise Exception('Invalid Type: Numpy Array')
            if ( (x.shape != flatten_shape    and not (len(x.shape) == 2 and x.shape[1:] == flatten_shape)) and 
                 (x.shape != self.input_shape and not (len(x.shape) == 4 and x.shape[1:] == self.input_shape)) ):
                raise Exception('Invalid numpy array shape passed into MyModel.predict: {}'.format(x.shape))
            x = x.reshape(-1,*self.input_shape)

            # Transforms expected for raw images
            pred_transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                 ])

            if self.configs.architecture == 'efficient':
                pred_transform = transforms.Compose([transforms.ToPILImage(),
                                                     transforms.Resize(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                     ])

            # Transforms applied for each image
            imgs = []
            for img in x:
                imgs.append(pred_transform(img).reshape(-1,*process_shape))
            x = torch.cat(imgs,0)

        elif isinstance(x, torch.Tensor):
            if x.shape != process_shape and not (len(x.shape) == 4 and x.shape[1:] == process_shape):
                raise Exception('Invalid torch tensor shape passed into MyModel.predict: {}'.format(x.shape))
            x = x.reshape(-1,*process_shape)

        else:
            raise Exception('Invalid type passed into MyModel.predict: {}'.format(type(x)))

        self.network.eval()
        return self.network(x).detach().cpu().numpy()

    def save(self, des=''):
        from torch import save
        from os import path
        from pathlib import Path

        # Save the configuration so that at load, different configs will not stop loading
        checkpoint = {'configs':          self.configs,
                      'model_state_dict': self.network.state_dict()}

        # Compute the full file path 
        name = self.name if self.name else ''
        save_dir = path.join(path.dirname(path.abspath(__file__)), self.configs.model_directory, name)
        save_path = path.join(save_dir, 'model'+des+'.th')

        # Ensures the save_directory exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save(checkpoint, save_path)

    def load(self, filePath=''):
        from torch import load
        from os import path

        def update_configs(args):
            for k, v in vars(args).items():
                vars(self.configs)[k] = v

        # Check to see if actual 'th' file passed into load.  If not, it is assumed that the value passed in is a path, and model.th is appended on.
        if filePath[-2:] != 'th': filePath = path.join(filePath,'model.th')
        # Check to see if file passed into load exists.
        if not path.exists(filePath): raise(Exception('Invalid filePath passed into MyModel.load: "{}" [{}]'.format(str(filePath), str(type(filePath)))))

        checkpoint = load(path.join(path.dirname(path.abspath(__file__)), filePath), map_location='cpu')
        update_configs(checkpoint['configs'])
        self.network = MyNetwork(self.configs).to(self._device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        return self.configs


### END CODE HERE
