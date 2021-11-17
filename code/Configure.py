# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

# argparse can contain the values 'action', 'type', 'choices', 'help'
# key corresponds to dest in argparse
# default corresponds to default in argparse
# flags corresponds to flags in argparse

default_configs = {
    'epochs':            {'default': 1,
                          'flags':   ['--epochs', '-E'],
                          'argparse':{'type':int}},
    'learning_rate':     {'default': 0.001,
                          'flags':   ['--learning_rate', '-L'],
                          'argparse':{'type':float}},
    'batch_size':        {'default': 128,
                          'flags':   ['--batch_size', '-B'],
                          'argparse':{'type':int}},
    'weight_decay':      {'default': 0.0002,
                          'flags':   ['--weight_decay', '-W'],
                          'argparse':{'type':float}},
    'save_interval':     {'default': None,
                          'flags':   ['--save_int', '-S'],
                          'argparse':{'type':int,'help':'Save Interval in epochs.  A value of 0 saves at completion only.  All valid interval values save on completion in model.th in specified directory.'}},

    'step_schedule':     {'default': False,
                          'flags':   ['--step_schedule', '-s'],
                          'argparse':{'action':'store_true','help':'Learning rate decay.'}},
    'cosine':            {'default': False,
                          'flags':   ['--cosine'],
                          'argparse':{'action':'store_true','help':'CosineAnnealing Learning Rate Scheduler.'}},
    'adam':              {'default': False,
                          'flags':   ['--adam', '-a'],
                          'argparse':{'action':'store_true','help':'Adam optimizer used.'}},
    'data_augmentation': {'default': False,
                          'flags':   ['--data_aug','-d'],
                          'argparse':{'action':'store_true','help':'Data Augmentation takes place during training.'}},

    'augmentation_type': {'default': 'normal',
                          'flags':   ['--Aug'],
                          'argparse':{'type':str,
                                      'choices':['all','affine','crop','rotate','simple','light-heavy','croprot','normal'],
                                      'help':'Data Aug used in the training.  Options include: "all", "affine", "crop", "rotate", "simple", "light-heavy", "croprot", "normal"'}},
    'dropout_rate':      {'default': 0.0,
                          'flags':   ['--dropout_rate', '-D'],
                          'argparse':{'type':float}},
    'activation':        {'default': 'relu',
                          'flags':   ['--Act'],
                          'argparse':{'type':str,
                                      'choices':['relu','swish'],
                                      'help':'Activation Function used by the model.  Options include: "relu", "swish"'}},
    'validation':        {'default': True,
                          'flags':   ['--no_validation','-v'],
                          'argparse':{'action':'store_false','help':'Turns validation off.'}},

    'wandb':             {'default': True,
                          'flags':   ['--no_logging', '-n'],
                          'argparse':{'action':'store_false','help':'Turns logging on Weights&Bias off.'}},
    'description':       {'default': '',
                          'flags':   ['--des'],
                          'argparse':{'type':str,'help':'Model Description, used in folder and log naming.'}},
    'verify':            {'default': False,
                          'flags':   ['--verify'],
                          'argparse':{'action':'store_true','help':'Sets verification mode.'}},
    'mode':              {'default': 'train',
                          'flags':   ['--mode'],
                          'argparse':{'type':str,'choices':['train','test','verify','predict'],'help':'Program mode.  Options include: "train", "test", "verify", "predict"'}},
    'silent':            {'default': False,
                          'flags':   ['--silent'],
                          'argparse':{'action':'store_true','help':'Model training and evaluation runs in silent mode.'}},


    'data_directory':    {'default': 'data',
                          'flags':   ['--data_dir'],
                          'argparse':{'type':str,'help':'Path of the directory the data used in training and evaluation.'}},
    'load_directory':    {'default': None,
                          'flags':   ['--load'],
                          'argparse':{'type':str,'help':'Path of the file the network weights are initialzed with.  If any value is passed, the weights are loaded.'}},
    'log_directory':     {'default': 'logs',
                          'flags':   ['--log_dir'],
                          'argparse':{'type':str,'help':'Path of the log directory for W&B.'}},
    'model_directory':   {'default': 'models',
                          'flags':   ['--model_dir'],
                          'argparse':{'type':str,'help':'Path of the directory the model is saved to.'}},


    'num_classes':       {'default': 10,
                          'flags':   ['--num_classes'],
                          'argparse':{'type':int}},
}

def print_configs(configs=None):
    print("\n-------------------------------------------------------------------------------")
    print("| {:^75} |".format('Configuration'))
    print("| {:^23} | {:^23} | {:^23} |".format('Option Descriptions','Option Strings', 'Values Passed In' if configs else 'Default Values'))
    print("-------------------------------------------------------------------------------")
    for k,c in default_configs.items():
        value = c['default']
        if configs: value = getattr(configs, k)
        if value is None: value = 'None'
        if isinstance(value, bool): value = str(value)
        print("| {:<23} | {:<23} | {:<23} |".format(k, ' , '.join(c['flags']), value))
        if configs is None and 'help' in c['argparse'].keys(): print("    Help: {} \n".format(c['argparse']['help']))
    print("-------------------------------------------------------------------------------\n")

def parse_configs():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    # Programatically add default_configs to parser arguments
    for k,c in default_configs.items():
        parser.add_argument(*c['flags'], dest=k, default=c['default'], **c['argparse'])
    parser.add_argument('-h', '--help', dest='help', action='store_true', default=False)
    configs = parser.parse_args()

    return configs

### END CODE HERE
