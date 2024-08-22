from typing import List

import torch
from FunctionEncoder import *

from src.algs.AutoEncoder import AutoEncoder
from src.algs.BruteForceBasis import BruteForceBasis
from src.algs.BruteForceMLP import BruteForceMLP
from src.algs.MAML import MAML
from src.algs.Oracle import Oracle
from src.algs.Transformer import Transformer
from src.algs.TransformerFunctionalEncoding import TransformerFunctionalEncoding


def predict_number_params(alg:str, n_basis:int, n_layers:int, input_size:List[int], output_size:List[int], hidden_size:int, n_heads:int, oracle_size:int, n_examples:int):
    num_params = 0
    if alg == "FE":
        # function encoder
        in_size = input_size[0]
        out_size = output_size[0] * n_basis
        num_params += in_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * out_size + out_size

        # average function
        in_size = input_size[0]
        out_size = output_size[0]
        num_params += in_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * out_size + out_size
    elif alg == "AE":
        # encoder
        in_size = input_size[0] + output_size[0]
        out_size = n_basis
        num_params += in_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * out_size + out_size

        # decoder
        in_size = input_size[0] + n_basis
        out_size = output_size[0]
        num_params += in_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * out_size + out_size
    elif alg == "Transformer":
        # transformer size
        num_trans_layers = 4
        kvq_size = 3

        # first norm layer
        norm1 = n_basis * 2
        # size of each layer
        multihead_attention_size = (kvq_size * n_basis +
                                    kvq_size * n_basis * n_basis +
                                    n_basis * n_basis + n_basis)
        linear1_size = n_basis * hidden_size + hidden_size
        linear2_size = hidden_size * n_basis + n_basis
        norm1_size = 2 * n_basis
        norm2_size = 2 * n_basis

        # full encoder layer size
        encoder_layer_size = multihead_attention_size + linear1_size + linear2_size + norm1_size + norm2_size
        transformer_encoder_size = num_trans_layers * encoder_layer_size + norm1

        # decoder side of transformer
        norm1 = n_basis * 2
        # size of each layer
        multihead_attention_size = (kvq_size * n_basis +
                                    kvq_size * n_basis * n_basis +
                                    n_basis * n_basis + n_basis)
        linear1_size = n_basis * hidden_size + hidden_size
        linear2_size = hidden_size * n_basis + n_basis
        norm1_size = 2 * n_basis
        norm2_size = 2 * n_basis
        norm3_size = 2 * n_basis
        self_attention_size = (kvq_size * n_basis +
                                kvq_size * n_basis * n_basis +
                                n_basis * n_basis + n_basis)
        layer_size = multihead_attention_size + linear1_size + linear2_size + norm1_size + norm2_size + norm3_size + self_attention_size
        transformer_decoder_size = num_trans_layers * layer_size + norm1



        # encoder examples
        encode_size1 = 0
        encode_size1 += (input_size[0] + output_size[0]) * hidden_size + hidden_size
        encode_size1 += hidden_size * n_basis + n_basis

        # encoder prediction
        encode_size2 = 0
        encode_size2 += input_size[0] * hidden_size + hidden_size
        encode_size2 += hidden_size * n_basis + n_basis


        # decoder
        decode_size = 0
        decode_size += n_basis * hidden_size + hidden_size
        decode_size += hidden_size * output_size[0] + output_size[0]

        # final
        num_params += transformer_encoder_size + transformer_decoder_size + encode_size1 + encode_size2 + decode_size
    elif alg == "TFE":
        num_trans_layers = 4
        kvq_size = 3

        # transformer size - decoder only model
        # size of each layer
        linear1_size = n_basis * hidden_size + hidden_size
        linear2_size = hidden_size * n_basis + n_basis
        norm1_size = 2 * n_basis
        norm2_size = 2 * n_basis
        self_attention_size = (kvq_size * n_basis +
                                kvq_size * n_basis * n_basis +
                                n_basis * n_basis + n_basis)
        layer_size = linear1_size + linear2_size + norm1_size + norm2_size + self_attention_size
        transformer_decoder_size = num_trans_layers * layer_size
        num_params += transformer_decoder_size

        # encoder examples
        num_params += (input_size[0] + output_size[0]) * hidden_size + hidden_size
        num_params += hidden_size * n_basis + n_basis

        # basis functions
        num_params += input_size[0] * hidden_size + hidden_size
        num_params += 2 * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * n_basis * output_size[0] + n_basis * output_size[0]

        # decoder
        num_params += n_basis * hidden_size + hidden_size
        num_params += hidden_size * n_basis + n_basis

    elif alg == "Oracle":
        in_size = input_size[0] + oracle_size
        out_size = output_size[0]
        num_params += in_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * out_size + out_size
    elif alg == "BF":
        # model size
        ins = (input_size[0] + output_size[0]) * n_examples + input_size[0]
        outs = output_size[0]
        num_params += ins * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * outs + outs
    elif alg == "BFB":
        # example to basis size
        in_size = (input_size[0] + output_size[0]) * n_examples
        out_size = n_basis
        num_params += in_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * out_size + out_size

        # basis size
        in_size = input_size[0]
        out_size = n_basis * output_size[0]
        num_params += in_size * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * out_size + out_size
    elif alg == "MAML":
        # model size
        ins = input_size[0]
        outs = output_size[0]
        num_params += ins * hidden_size + hidden_size
        num_params += (n_layers - 2) * (hidden_size * hidden_size + hidden_size)
        num_params += hidden_size * outs + outs


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


def get_hidden_size(model:str, train_dataset:BaseDataset, num_params:int, n_basis:int, n_layers:int, n_heads:int ):
    def loss_function(hidden_size):
        current_num_params = predict_number_params(model, n_basis, n_layers, train_dataset.input_size, train_dataset.output_size, hidden_size, n_heads, train_dataset.oracle_size, train_dataset.n_examples_per_sample)
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


def get_model(alg:str, train_dataset:BaseDataset, n_basis:int, n_layers:int, n_heads:int, hidden_size:int, method:str, maml_steps:int):
    if alg == "FE":
        model_type = "MLP" if len(train_dataset.input_size) == 1 else "CNN"
        return FunctionEncoder(input_size=train_dataset.input_size,
                               output_size=train_dataset.output_size,
                               data_type=train_dataset.data_type,
                               n_basis=n_basis,
                               model_type=model_type,
                               model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                               use_residuals_method=True,
                               method=method,
                               )
    elif alg == "AE":
        return AutoEncoder(input_size=train_dataset.input_size,
                           output_size=train_dataset.output_size,
                           n_basis=n_basis,
                           data_type=train_dataset.data_type,
                           model_type="MLP",
                            model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                            )
    elif alg == "Transformer":
        return Transformer(input_size=train_dataset.input_size,
                           output_size=train_dataset.output_size,
                           n_basis=n_basis,
                           data_type=train_dataset.data_type,
                           model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers, "n_heads": n_heads},
                           )

    elif alg == "TFE":
        return TransformerFunctionalEncoding(input_size=train_dataset.input_size,
                                            output_size=train_dataset.output_size,
                                            n_basis=n_basis,
                                            data_type=train_dataset.data_type,
                                            model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers, "n_heads": n_heads},
                                            )
    elif alg == "Oracle":
        return Oracle(input_size=train_dataset.input_size,
                        output_size=train_dataset.output_size,
                        oracle_size=train_dataset.oracle_size,
                        data_type=train_dataset.data_type,
                        model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                        )
    elif alg == "BF":
        return BruteForceMLP(input_size=train_dataset.input_size,
                            output_size=train_dataset.output_size,
                            data_type=train_dataset.data_type,
                            num_data=train_dataset.n_examples_per_sample,
                            model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                            )
    elif alg == "BFB":
        return BruteForceBasis(input_size=train_dataset.input_size,
                                output_size=train_dataset.output_size,
                                data_type=train_dataset.data_type,
                                num_data=train_dataset.n_examples_per_sample,
                                num_basis=n_basis,
                                model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                               )
    elif alg == "MAML":
        return MAML(input_size=train_dataset.input_size,
                    output_size=train_dataset.output_size,
                    data_type=train_dataset.data_type,
                    n_maml_update_steps=maml_steps,
                    model_kwargs={"hidden_size": hidden_size, "n_layers": n_layers},
                    )
    else:
        raise ValueError(f"Algorithm {alg} not recognized.")