from typing import List

import torch
from FunctionEncoder import *

from src.algs.AutoEncoder import AutoEncoder
from src.algs.BruteForceBasis import BruteForceBasis
from src.algs.BruteForceMLP import BruteForceMLP
from src.algs.MAML import MAML, MAML_INTERNAL_LEARNING_RATE
from src.algs.MLP import MLP, MLPModel
from src.algs.Oracle import Oracle
from src.algs.PrototypicalNetwork import ProtoTypicalNetwork
from src.algs.Siamese import SiameseNetwork
from src.algs.Transformer import Transformer
from src.algs.TransformerFunctionalEncoding import TransformerFunctionalEncoding
from src.algs.TransformerWithPositions import TransformerWithPositions

def predict_number_params(alg:str, n_basis:int, input_size:tuple[int], output_size:tuple[int], model_type, model_kwargs):
    num_params = 0
    if alg in ["FE", "LS", "IP"]:
        num_params = FunctionEncoder.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs, use_residuals_method=True )
    elif alg == "AE":
        num_params = AutoEncoder.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif alg == "Transformer":
        num_params = Transformer.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif alg == "TFE":
        num_params = TransformerFunctionalEncoding.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif alg == "Oracle":
        num_params = Oracle.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif alg == "BF":
        num_params = BruteForceMLP.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif alg == "BFB":
        num_params = BruteForceBasis.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif "MAML" in alg:
        num_params = MAML.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif alg == "TransformerWithPositions":
        num_params = TransformerWithPositions.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif alg == "Siamese":
        num_params = SiameseNetwork.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif alg == "Proto":
        num_params = ProtoTypicalNetwork.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    elif alg == "MLP":
        num_params = MLPModel.predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs)
    else:
        raise ValueError(f"Algorithm {alg} not recognized.")
    return num_params

def get_number_params(model):
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters())
    elif isinstance(model, torch.Tensor):
        return model.numel()
    elif type(model) is list:
        return sum([get_number_params(m) for m in model])
    elif type(model) is dict:
        return sum([get_number_params(m) for m in model.values()])
    else:
        raise ValueError(f"Model type {type(model)} not recognized, expected one of [torch.nn.Module, torch.tensor, List, dict].")


def get_hidden_size(model:str, train_dataset:BaseDataset, num_params:int, n_basis:int, n_layers:int, n_heads:int):
    model_type = "MLP" if len(train_dataset.input_size) == 1 else "CNN"
    def loss_function(hidden_size):
        model_kwargs = {"hidden_size": hidden_size, "n_layers": n_layers, "n_heads": n_heads, "oracle_size": train_dataset.oracle_size, "n_examples": train_dataset.n_examples_per_sample}
        current_num_params = predict_number_params(model, n_basis, train_dataset.input_size, train_dataset.output_size, model_type, model_kwargs)
        return abs(current_num_params - num_params)

    def ternary_search(start, end):
        while end - start > 2:
            mid1 = start + (end - start) // 3
            mid2 = end - (end - start) // 3

            loss1 = loss_function(mid1)
            loss2 = loss_function(mid2)

            if loss1 < loss2:
                end = mid2
            else:
                start = mid1

        best_input = start
        min_loss = loss_function(start)
        # for x in range(start, end + 1):
        #     current_loss = loss_function(x)
        #     if current_loss < min_loss:
        #         min_loss = current_loss
        #         best_input = x

        return best_input, min_loss

    start = 10
    end = int(1e5)

    best_input, min_loss = ternary_search(start, end)
    # print(f'The best input is {best_input} with a loss of {min_loss}')
    # print(f"Target number of parameters: {target_n_parameters}")
    # print(f"Predicted number of parameters: {predict_number_params(model_type, n_sensors, n_basis, best_input, n_layers, src_input_space, src_output_space, tgt_input_space, tgt_output_space, transformation_type)}")
    return best_input


def get_model(alg:str, train_dataset:BaseDataset, n_basis:int, n_layers:int, n_heads:int, hidden_size:int, maml_steps:int, cross_entropy:bool=False, maml_internal_learning_rate=None, device="cuda", gradient_accumulation=1):
    model_type = "MLP" if len(train_dataset.input_size) == 1 else "CNN"
    if alg == "FE" or alg == "LS": # least squares generally performs better than inner product, so its the default for the function encoder.
        return FunctionEncoder(input_size=train_dataset.input_size,
                               output_size=train_dataset.output_size,
                               data_type=train_dataset.data_type,
                               n_basis=n_basis,
                               model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                               use_residuals_method=True,
                               method="least_squares",
                               gradient_accumulation=gradient_accumulation,
                               ).to(device)
    elif alg == "IP":
        return FunctionEncoder(input_size=train_dataset.input_size,
                               output_size=train_dataset.output_size,
                               data_type=train_dataset.data_type,
                               n_basis=n_basis,
                               model_type=model_type,
                               model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                               use_residuals_method=True,
                               method="inner_product",
                               gradient_accumulation=gradient_accumulation,
                               ).to(device)
    elif alg == "AE":
        return AutoEncoder(input_size=train_dataset.input_size,
                           output_size=train_dataset.output_size,
                           n_basis=n_basis,
                           data_type=train_dataset.data_type,
                           model_type=model_type,
                            model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                           cross_entropy=cross_entropy,
                           gradient_accumulation=gradient_accumulation,
                           ).to(device)
    elif alg == "Transformer":
        return Transformer(input_size=train_dataset.input_size,
                           output_size=train_dataset.output_size,
                           n_basis=n_basis,
                           data_type=train_dataset.data_type,
                           model_type=model_type,
                           model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers, "n_heads": n_heads},
                           cross_entropy=cross_entropy,
                           gradient_accumulation=gradient_accumulation,
                           ).to(device)

    elif alg == "TFE":
        return TransformerFunctionalEncoding(input_size=train_dataset.input_size,
                                            output_size=train_dataset.output_size,
                                            n_basis=n_basis,
                                            data_type=train_dataset.data_type,
                                             model_type=model_type,
                                            model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers, "n_heads": n_heads},
                                             cross_entropy=cross_entropy,
                                             gradient_accumulation=gradient_accumulation,
                                             ).to(device)
    elif alg == "Oracle":
        return Oracle(input_size=train_dataset.input_size,
                        output_size=train_dataset.output_size,
                        oracle_size=train_dataset.oracle_size,
                        data_type=train_dataset.data_type,
                        model_type=model_type,
                        model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                      cross_entropy=cross_entropy,
                      gradient_accumulation=gradient_accumulation,
                      ).to(device)
    elif alg == "BF":
        return BruteForceMLP(input_size=train_dataset.input_size,
                            output_size=train_dataset.output_size,
                            data_type=train_dataset.data_type,
                            num_data=train_dataset.n_examples_per_sample,
                            model_type=model_type,
                            model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                             cross_entropy=cross_entropy,
                             gradient_accumulation=gradient_accumulation,
                             ).to(device)
    elif alg == "BFB":
        return BruteForceBasis(input_size=train_dataset.input_size,
                                output_size=train_dataset.output_size,
                                data_type=train_dataset.data_type,
                                num_data=train_dataset.n_examples_per_sample,
                                n_basis=n_basis,
                                model_type=model_type,
                                model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                               cross_entropy=cross_entropy,
                               gradient_accumulation=gradient_accumulation,
                               ).to(device)
    elif "MAML" in alg:
        return MAML(input_size=train_dataset.input_size,
                    output_size=train_dataset.output_size,
                    data_type=train_dataset.data_type,
                    model_type=model_type,
                    n_maml_update_steps=maml_steps,
                    model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                    cross_entropy=cross_entropy,
                    internal_learning_rate=maml_internal_learning_rate if maml_internal_learning_rate else MAML_INTERNAL_LEARNING_RATE[alg][train_dataset.__class__.__name__],
                    gradient_accumulation=gradient_accumulation,
                    ).to(device)
    elif alg == "TransformerWithPositions":
        return TransformerWithPositions(input_size=train_dataset.input_size,
                                        output_size=train_dataset.output_size,
                                        data_type=train_dataset.data_type,
                                        n_basis=n_basis,
                                        max_example_size=2*train_dataset.n_examples_per_sample,
                                        model_type=model_type,
                                        model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers, "n_heads": n_heads},
                                        cross_entropy=cross_entropy,
                                        gradient_accumulation=gradient_accumulation,
                                        ).to(device)
    elif alg == "Siamese":
        return SiameseNetwork(input_size=train_dataset.input_size,
                              output_size=train_dataset.output_size,
                              data_type=train_dataset.data_type,
                              n_basis=n_basis,
                              model_type=model_type,
                              model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                              cross_entropy=cross_entropy,
                              gradient_accumulation=gradient_accumulation,
                              ).to(device)
    elif alg == "Proto":
        return ProtoTypicalNetwork(input_size=train_dataset.input_size,
                                      output_size=train_dataset.output_size,
                                      data_type=train_dataset.data_type,
                                      n_basis=n_basis,
                                      model_type=model_type,
                                      model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                                      cross_entropy=cross_entropy,
                                   gradient_accumulation=gradient_accumulation,
                                   ).to(device)
    elif alg == "MLP":
        return MLPModel(input_size=train_dataset.input_size,
                    output_size=train_dataset.output_size,
                    data_type=train_dataset.data_type,
                    model_type=model_type,
                    model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                    cross_entropy=cross_entropy,
                        gradient_accumulation=gradient_accumulation,
                        ).to(device)
    else:
        raise ValueError(f"Algorithm {alg} not recognized.")