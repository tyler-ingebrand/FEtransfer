

dataset_names = {   "Polynomial": "Polynomial",
                    "CIFAR": "CIFAR",
                    "7Scenes": "7Scenes",
                    "Ant": "Ant"
}
alg_names = {   "LS": "FE (LS)",
                "IP": "FE (IP)",
                "AE": "AE",
                "Transformer": "Trans.",
                "TFE": "TFE",
                "Oracle": "Oracle",
                "BFB": "BFB",
                "BF": "BF",
                "MAML1": "MAML (n=1)",
                "MAML5": "MAML (n=5)",
                "Siamese": "Siamese Net",
                "Proto": "ProtoTyp."
}

alg_colors = {
    "LS": "tab:blue",   
    "IP": "tab:cyan",  
    "AE": "tab:green",   
    "Transformer": "tab:red",  
    "LS-Parallel": "tab:red",  
    "TFE": "tab:pink", 
    "Oracle": "tab:purple", 
    "BFB": "tab:brown",  
    "BF": "tab:olive", 
    "MAML1": "#ff8000", # orange 
    "MAML5": "#ffbf00", # other orange
    "Siamese": "slategrey",  
    "Proto": "lightsteelblue"   
}

titles = {  "train/accuracy": "Train",
            "type1/accuracy": "Type 1 Transfer",
            "type2/accuracy": "Type 2 Transfer",
            "type3/accuracy": "Type 3 Transfer",
            "train/mean_distance_squared": "Train",
            "type1/mean_distance_squared": "Type 1 Transfer",
            "type2/mean_distance_squared": "Type 2 Transfer",
            "type3/mean_distance_squared": "Type 3 Transfer"
}
yaxis = {   "train/accuracy": "Accuracy",
            "type1/accuracy": "Accuracy",
            "type2/accuracy": "Accuracy",
            "type3/accuracy": "Accuracy",
            "train/mean_distance_squared": "L2 Distance Squared",
            "type1/mean_distance_squared": "L2 Distance Squared",
            "type2/mean_distance_squared": "L2 Distance Squared",
            "type3/mean_distance_squared": "L2 Distance Squared"
}

ylims = {"Polynomial":{
    "Train": None, # NONE = default, usually fine if there are no outliers that make the graph ugly. 
    "Type 1 Transfer": None,
    "Type 2 Transfer": None,
    "Type 3 Transfer": None,
},
"CIFAR":{
    "Train": None,
    "Type 1 Transfer": None,
    "Type 3 Transfer": None,
},
"7Scenes":{
    "Train": None,
    "Type 1 Transfer": None,
    "Type 3 Transfer": None,
},
"Ant":{
    "Train": (1e-3, 15),
    "Type 1 Transfer": (3, 1e2),
    "Type 2 Transfer": None,
    "Type 3 Transfer": (40, 1e4),
}
}

FONT_SIZE = 19
FONT = "serif"