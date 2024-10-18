import torch


def _deterministic_inner_product(fs: torch.tensor,
                                 gs: torch.tensor, ) -> torch.tensor:
    # reshaping
    unsqueezed_fs, unsqueezed_gs = False, False
    if len(fs.shape) == 3:
        fs = fs.unsqueeze(-1)
        unsqueezed_fs = True
    if len(gs.shape) == 3:
        gs = gs.unsqueeze(-1)
        unsqueezed_gs = True

    # compute inner products via MC integration
    element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
    inner_product = torch.mean(element_wise_inner_products, dim=1)

    # undo reshaping
    if unsqueezed_fs:
        inner_product = inner_product.squeeze(-2)
    if unsqueezed_gs:
        inner_product = inner_product.squeeze(-1)
    return inner_product

def _stochastic_inner_product(fs: torch.tensor,
                              gs: torch.tensor, ) -> torch.tensor:
    assert len(fs.shape) in [3, 4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
    assert len(gs.shape) in [3, 4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
    assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
    assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
    assert fs.shape[2] == gs.shape[2] == 1, f"Expected fs and gs to have the same output size, which is 1 for the stochastic case since it learns the pdf(x), got {fs.shape[2]} and {gs.shape[2]}"

    # reshaping
    unsqueezed_fs, unsqueezed_gs = False, False
    if len(fs.shape) == 3:
        fs = fs.unsqueeze(-1)
        unsqueezed_fs = True
    if len(gs.shape) == 3:
        gs = gs.unsqueeze(-1)
        unsqueezed_gs = True
    assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

    # compute means and subtract them
    mean_f = torch.mean(fs, dim=1, keepdim=True)
    mean_g = torch.mean(gs, dim=1, keepdim=True)
    fs = fs - mean_f
    gs = gs - mean_g

    # compute inner products
    element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
    inner_product = torch.mean(element_wise_inner_products, dim=1)
    # Technically we should multiply by volume, but we are assuming that the volume is 1 since it is often not known

    # undo reshaping
    if unsqueezed_fs:
        inner_product = inner_product.squeeze(-2)
    if unsqueezed_gs:
        inner_product = inner_product.squeeze(-1)
    return inner_product

def _categorical_inner_product(fs: torch.tensor,
                               gs: torch.tensor, ) -> torch.tensor:
    assert len(fs.shape) in [3, 4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
    assert len(gs.shape) in [3, 4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
    assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
    assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
    assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, which is the number of categories in this case, got {fs.shape[2]} and {gs.shape[2]}"

    # reshaping
    unsqueezed_fs, unsqueezed_gs = False, False
    if len(fs.shape) == 3:
        fs = fs.unsqueeze(-1)
        unsqueezed_fs = True
    if len(gs.shape) == 3:
        gs = gs.unsqueeze(-1)
        unsqueezed_gs = True
    assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

    # compute means and subtract them
    mean_f = torch.mean(fs, dim=2, keepdim=True)
    mean_g = torch.mean(gs, dim=2, keepdim=True)
    fs = fs - mean_f
    gs = gs - mean_g

    # compute inner products
    element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
    inner_product = torch.mean(element_wise_inner_products, dim=1)

    # undo reshaping
    if unsqueezed_fs:
        inner_product = inner_product.squeeze(-2)
    if unsqueezed_gs:
        inner_product = inner_product.squeeze(-1)
    return inner_product

def _inner_product(fs: torch.tensor,
                   gs: torch.tensor,
                   data_type:str) -> torch.tensor:
    if data_type == "deterministic":
        return _deterministic_inner_product(fs, gs)
    elif data_type == "stochastic":
        return _stochastic_inner_product(fs, gs)
    elif data_type == "categorical":
        return _categorical_inner_product(fs, gs)
    else:
        raise ValueError(f"Unknown data type: '{data_type}'. Should be 'deterministic', 'stochastic', or 'categorical'")

def _norm(fs: torch.tensor, data_type:str, squared=False) -> torch.tensor:
    norm_squared = _inner_product(fs, fs, data_type)
    if not squared:
        return norm_squared.sqrt()
    else:
        return norm_squared

def _distance(fs: torch.tensor, gs: torch.tensor, data_type:str, squared=False) -> torch.tensor:
    return _norm(fs - gs, data_type, squared=squared)