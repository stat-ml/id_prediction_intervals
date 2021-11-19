import inspect


def get_line_info():
    return f"{inspect.stack()[1][1]}: {inspect.stack()[1][2]}: {inspect.stack()[1][3]}"


def print_model_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data}")
