# Test/tcn_configs.py

import itertools
import random


#Broad Hyperparameter_Tuning
#nb_filters_list    = [16, 32, 64, 128]
#kernel_size_list   = [3, 5, 7]
#nb_stacks_list     = [1, 2, 3]
#dilations_list     = [[1,2,4], [1,2,4,8],[1, 2, 4, 8, 16]]
#dropout_list       = [0.0, 0.1, 0.2, 0.3]
#padding_list       = ['causal']
#return_seq_list    = [False, True]
#skip_list          = [True, False]
#batchnorm_list     = [True,False]
#layernorm_list     = [True,False]
#goback_list        = [True,False]
#return_state_list  = [False]


# NEW hyperparameters
#batch_size_list    = [64, 128, 256]
#dense_units_list   = [64, 128, 256, 512]

# Dense layer hyperparameters
#num_dense_layers_list = [0,1,2,3] 
#dense_activation_list = ['relu']
#dense_dropout_list = [0.0, 0.2, 0.3]




#1) Limited range1

#nb_filters_list    = [32, 64]          # limited to 2 options
#kernel_size_list   = [3, 5]            # limited to 2 options
#nb_stacks_list     = [1, 2]            # limited to 2 options
#dilations_list     = [
#    [1, 2, 4],
#    [1, 2, 4, 8]
#]                                       # limited to 2 options
#dropout_list       = [0.0, 0.3]         # limited to 2 options
#padding_list       = ['causal']         # 1 option
#return_seq_list    = [False]            # 1 option
#skip_list          = [True]              # 2 options
#batchnorm_list     = [True, False]      # 2 options
#layernorm_list     = [False]            # 1 option (to avoid batchnorm+layernorm conflict easily)
#goback_list        = [False]            # 1 option
#return_state_list  = [False]            # 1 option

# Dense Layers After TCN
#batch_size_list       = [128]             # 1 option
#dense_units_list      = [64, 128]         # 2 options
#num_dense_layers_list = [1, 2]            # 2 options
#dense_activation_list = ['relu']          # 1 option
#dense_dropout_list    = [0.0, 0.2]        # 2 options


##2) Limited range 2

#nb_filters_list    = [128, 256]          # Changed 
#kernel_size_list   = [3, 5]               
#nb_stacks_list     = [1, 2, 3]                       
#dilations_list     = [
#    [1, 2, 4],
#    [1, 2, 4, 8],
#    [1, 2, 4, 8, 16]
#]                                      
#dropout_list       = [0.0, 0.3]        
#padding_list       = ['causal']         
#return_seq_list    = [False]             
#skip_list          = [True]              
#batchnorm_list     = [True, False]     
#layernorm_list     = [False]         
#goback_list        = [False]            
#return_state_list  = [False]            

## Dense Layers After TCN
#batch_size_list       = [128]             
#dense_units_list      = [64, 128]         
#num_dense_layers_list = [1, 2, 3]           
#dense_activation_list = ['relu']          
#dense_dropout_list    = [0.0, 0.2, 0.3]        




##3) Limited range 3

#nb_filters_list        = [128, 256, 512]           # Changed
#kernel_size_list       = [3, 5, 7]                 # 7 is added
#nb_stacks_list         = [2, 3, 4, 5]
#dilations_list         = [
#    [1, 2, 4],
#    [1, 2, 4, 8],
#    [1, 2, 4, 8, 16],
#    [1, 2, 4, 8, 16, 32],               # added
#    [1, 2, 4, 8, 16, 32, 64],          # added
#]
#dropout_list           = [0.0, 0.3]
#padding_list           = ['causal']
#return_seq_list        = [False]
#skip_list              = [True]
#batchnorm_list         = [True, False]
#layernorm_list         = [False]
#goback_list            = [False]
#return_state_list      = [False]

## Dense Layers After TCN
#batch_size_list        = [128]
#dense_units_list       = [64, 128,256]
#num_dense_layers_list  = [1, 2, 3, 4]              # 4 is added
#dense_activation_list  = ['relu']
#dense_dropout_list     = [0.0, 0.2, 0.3]



##5) Limited range 5

#nb_filters_list        = [128, 256]           # 512 is removed
#kernel_size_list       = [ 5, 7, 9]                 #9 is added & 3 is removed
#nb_stacks_list         = [ 4, 5, 6,7]           #2,3 are removed and 6 & 7 are added 
#dilations_list         = [

#    [1, 2, 4, 8, 16, 32],               # added
#    [1, 2, 4, 8, 16, 32, 64],         # added
#]
#dropout_list           = [0.0]              #0.3 is removed 
#padding_list           = ['causal']
#return_seq_list        = [False]
#skip_list              = [True]
#batchnorm_list         = [True, False]
#layernorm_list         = [False]
#goback_list            = [False]
#return_state_list      = [False]

## Dense Layers After TCN
#batch_size_list        = [128]
#dense_units_list       = [128,256, 512]         # 64 is removed & 512 is added
#num_dense_layers_list  = [2, 3, 4, 5]              # 5 is added
#dense_activation_list  = ['relu']
#dense_dropout_list     = [0.0]            #0.2 & 0.3 are removed 


##6) Limited range 6

#nb_filters_list        = [128]          
#kernel_size_list       = [7,9,11,15]                 
#nb_stacks_list         = [ 4, 5]          
#dilations_list         = [

#    [1, 2, 4, 8, 16, 32],              
#    [1, 2, 4, 8, 16, 32, 64],        
#]
#dropout_list           = [0.0,0.3]             
#padding_list           = ['causal']
#return_seq_list        = [False]
#skip_list              = [True]
#batchnorm_list         = [True, False]
#layernorm_list         = [False]
#goback_list            = [False]
#return_state_list      = [False]

# Dense Layers After TCN
#batch_size_list        = [128]
#dense_units_list       = [128,256]        
#num_dense_layers_list  = [3, 4, 5,6]              
#dense_activation_list  = ['relu']
#dense_dropout_list     = [0.0,0.3]           


##7) range 7


#nb_filters_list        = [128]          
#kernel_size_list       = [2,9,11,15,17, 19]                 
#nb_stacks_list         = [4, 5]          
#dilations_list         = [

##    [1, 3, 9, 27, 81],
##    [1, 3, 9, 27, 81, 243], 

   # [1, 2, 4, 8, 16, 32],              
   # [1, 2, 4, 8, 16, 32, 64],        
#]
#dropout_list           = [0.0,0.3]             
#padding_list           = ['causal']
#return_seq_list        = [False]
#skip_list              = [True]
#batchnorm_list         = [True, False]
#layernorm_list         = [False]
#goback_list            = [False]
#return_state_list      = [False]

# Dense Layers After TCN
#batch_size_list        = [128]
#dense_units_list       = [128,256]        
#num_dense_layers_list  = [4, 5,6, 7, 8]              
#dense_activation_list  = ['relu']
#dense_dropout_list     = [0.0,0.3] 




##8) range 8 (first & parameters are executed)


#nb_filters_list        = [128]
#kernel_size_list       = [2,3,9,11,15,17,19]
#nb_stacks_list         = [6]
#dilations_list         = [

##    [1, 3, 9, 27, 81],
##    [1, 3, 9, 27, 81, 243],

#     [1, 2, 4, 8, 16, 32],
#    [1, 2, 4, 8, 16, 32, 64],
#]
#dropout_list           = [0.0]
#padding_list           = ['causal']
#return_seq_list        = [False]
#skip_list              = [True]
#batchnorm_list         = [True, False]
#layernorm_list         = [False]
#goback_list            = [False]
#return_state_list      = [False]

## Dense Layers After TCN
#batch_size_list        = [128]
#dense_units_list       = [128,256]
#num_dense_layers_list  = [8, 9, 10]
#dense_activation_list  = ['relu']
#dense_dropout_list     = [0.0]


##8) range 9


nb_filters_list        = [128]
kernel_size_list       = [21,23]
nb_stacks_list         = [6]
dilations_list         = [

#    [1, 3, 9, 27, 81],
#    [1, 3, 9, 27, 81, 243],

     [1, 2, 4, 8, 16, 32],
    [1, 2, 4, 8, 16, 32, 64],
]
dropout_list           = [0.0]
padding_list           = ['causal']
return_seq_list        = [False]
skip_list              = [True]
batchnorm_list         = [True, False]
layernorm_list         = [False]
goback_list            = [False]
return_state_list      = [False]

# Dense Layers After TCN
batch_size_list        = [128]
dense_units_list       = [128,256]
num_dense_layers_list  = [8, 9, 10]
dense_activation_list  = ['relu']
dense_dropout_list     = [0.0]








tcn_configs = []
for combo in itertools.product(
        nb_filters_list, kernel_size_list, nb_stacks_list, dilations_list,
        dropout_list, padding_list, return_seq_list, skip_list,
        batchnorm_list, layernorm_list, goback_list, return_state_list,
        batch_size_list, dense_units_list,
        num_dense_layers_list, dense_activation_list, dense_dropout_list
):
    cfg = {
        "nb_filters":    combo[0],
        "kernel_size":   combo[1],
        "nb_stacks":     combo[2],
        "dilations":     combo[3],
        "dropout_rate":  combo[4],
        "padding":       combo[5],
        "return_sequences": combo[6],
        "use_skip_connections": combo[7],
        "use_batch_norm":      combo[8],
        "use_layer_norm":      combo[9],
        "go_backwards":        combo[10],
        "return_state":        combo[11],
        "batch_size":          combo[12],
        "dense_units":         combo[13],
        "num_dense_layers": combo[14],     
        "dense_activation": combo[15],      
        "dense_dropout": combo[16],         
}
   
   # skip invalid configs
    if cfg["use_batch_norm"] and cfg["use_layer_norm"]:
        continue   
    tcn_configs.append(cfg)
    # if len(tcn_configs) >= 2:
    #   break

# Shuffle the configurations

print(f"Total number of valid configurations before shuffling: {len(tcn_configs)}")
random.seed(42)  # fixed seed for reproducibility
random.shuffle(tcn_configs)

print(f"Total number of valid configurations after shuffling: {len(tcn_configs)}")


# At the bottom of your `tcn_configs.py`:
sampled_configs = tcn_configs[:5000]  # or any number you want
tcn_configs = sampled_configs
print(f"Sampled {len(tcn_configs)} configurations.")


