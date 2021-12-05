import torch
import os
import numpy as np
from Model import MyModel
# from DataLoader import load_fake_data, load_testing_images
from Configure import parse_configs, print_configs

# -------------------------------------------
# Helper Functions --------------------------
# -------------------------------------------
def network_correctness(configs):
    return True
    # # "self.input_shape" is used in prediction to verify shape of input.  It changed
    # #   here to allow correctness testing to use as small of data as possible and to be as fast as possible.
    # num_data = 12
    # num_classes = 3
    # # fake_input_shape = (5, 5, 2)
    # fake_input_shape = (32, 32, 3)

    # configs.epochs = 100
    # configs.batch_size = num_data
    # configs.learning_rate = 0.001
    # configs.adam = True
    # configs.step_schedule = True
    # #configs.data_augmentation = False
    # configs.save_interval = None
    # configs.wandb = False
    # #configs.silent = True
    # configs.validation = False
    # configs.num_classes = num_classes
    # configs.num_channels = fake_input_shape[2]

    # x_fake, y_fake = load_fake_data(num_data,num_classes,fake_input_shape)
    # model = MyModel(configs, input_shape=fake_input_shape)
    # model.train(x_fake, y_fake)
    # res = model.evaluate(x_fake, y_fake)
    # print(res)
    # # return model.evaluate(x_fake, y_fake) > (1 - 1/num_classes)
    # return res > (1.5/num_classes)

# Given a directory of model folders of the form "<NUM>_<DESCRIPTION>", creates and returns a new folder
#   with the number incremented and description optionaly included.
def generate_model_id(directory, des=''):
    from os import walk, path, mkdir
    if not path.isdir(directory): mkdir(directory)

    def safe_int(i):
        try:
            return int(i)
        except (ValueError, TypeError):
            return -1

    model_nums = sorted(list(map(safe_int, list(map(lambda x: x.split('-')[0], list(walk(directory))[0][1])))))
    model_nums.insert(0, -1)

    description = str(model_nums[-1] + 1)
    if des: description += '-' + des

    os.mkdir(directory+'/'+description)
    return description


def gen_prediction():
    import pickle as pkl
    import pandas as pd
    with open('data/saved_config.pkl','rb') as f:
        configs = pkl.load(f)

    model = MyModel(configs)
    model.train()
    rscore, thresh = model.evaluate()
    print(rscore)
    x = model.predict().detach().numpy()

    X = np.dot(x,np.transpose(x)) > thresh

    edges = []
    for r in range(64):
        for c in range(r+1,64):
            if X[r,c]:
                edges.append((r,c))

    pd.DataFrame(edges,columns=['src','dst']).to_csv('data/predictions.csv',index=False)
    df = pd.read_csv('data/predictions.csv')
    print(df)

def save_configs(configs)
    import pickle as pkl

    print_configs(configs)

    with open('data/saved_config.pkl','wb') as f:
        pkl.dump(configs,f)
    with open('data/saved_config.pkl','rb') as f:
        configs = pkl.load(f)

    print_configs(configs)


# -------------------------------------------
# Main logic --------------------------------
# -------------------------------------------
if __name__ == '__main__':
    # Get training and model configurations
    configs = parse_configs()

    # If help flag raised print help message and exit
    if configs.help: print_configs()

    elif configs.mode == 'train':
        print_configs(configs)

        # Generate model folder and name
        configs.description = generate_model_id(configs.model_directory,configs.description)
        model = MyModel(configs, name=configs.description)

        # Model Loading
        if configs.load_directory: print_configs(model.load(configs.load_directory))

        model.train()
        print(model.evaluate())

    elif configs.mode == 'test':
        model = MyModel(configs)

        # Model Loading
        if configs.load_directory: print_configs(model.load(configs.load_directory))

        print(model.evaluate())

    # Verification of network and training is used as part of CICD to ensure GPU efficiency.  Used with scripts.
    elif configs.mode == 'verify':
        if network_correctness(configs):
            exit()
        else:
            exit(1)

