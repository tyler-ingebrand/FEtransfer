

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

alg_colors = {  "LS": "#2151ff", # blue
                "IP": "#87cbff", # other blue
                "AE": "#06c910", # green
                "Transformer": "#f53bff", # pink
                "TFE": "#f00505", # red
                "Oracle": "#f07605", # orange
                "BFB": "#5a7a22", # forest
                "BF": "#c4fa66", # lime
                "MAML1": "#431347", # dark purple
                "MAML5": "#bb8ebf", # lilac
                "Siamese": "#4cad92", # aqua
                "Proto": "#daff5e" # yellow
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