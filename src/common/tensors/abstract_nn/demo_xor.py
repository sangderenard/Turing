
import pandas as pd
import matplotlib.pyplot as plt
from . import Linear, Model, Tanh, Sigmoid, BCEWithLogitsLoss, MSELoss, Adam, train_loop, set_seed
from .utils import from_list_like
from ..abstraction import AbstractTensor

def get_operator_dataset(op_name, like):
    X = from_list_like(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], like=like
    )
    if op_name == 'xor':
        Y = from_list_like([[0.0], [1.0], [1.0], [0.0]], like=like)
    elif op_name == 'and':
        Y = from_list_like([[0.0], [0.0], [0.0], [1.0]], like=like)
    elif op_name == 'or':
        Y = from_list_like([[0.0], [1.0], [1.0], [1.0]], like=like)
    elif op_name == 'nand':
        Y = from_list_like([[1.0], [1.0], [1.0], [0.0]], like=like)
    elif op_name == 'nor':
        Y = from_list_like([[1.0], [0.0], [0.0], [0.0]], like=like)
    else:
        raise ValueError(f"Unknown operator: {op_name}")
    return X, Y

import sys
import threading
import time

def run_operator_demo(op_name, loss_type, like, debug_hooks=None, until_stop=False):
    set_seed(0)
    X, Y = get_operator_dataset(op_name, like)
    model = Model(
        layers=[
            Linear(2, 8, like=like, init="xavier"),
            Linear(8, 1, like=like, init="xavier"),
        ],
        activations=[Tanh(), Sigmoid() if loss_type == 'mse' else None],
    )
    opt = Adam(model.parameters(), lr=1e-2 if loss_type == 'mse' else 3e-3)
    if loss_type == 'mse':
        loss_fn = MSELoss()
    else:
        loss_fn = BCEWithLogitsLoss()
    debug_data = []
    if debug_hooks:
        for event, hook in debug_hooks.items():
            loss_fn.register_hook(event, hook(debug_data))
    print(f"-- Operator: {op_name.upper()} | Loss: {loss_type.upper()} --")
    stop_flag = {'stop': False}
    def wait_for_key():
        print("Press any key to stop training and show results...")
        try:
            # Windows
            import msvcrt
            msvcrt.getch()
        except ImportError:
            # Unix
            import sys, select
            select.select([sys.stdin], [], [], None)
        stop_flag['stop'] = True
    if until_stop:
        t = threading.Thread(target=wait_for_key, daemon=True)
        t.start()
        epoch = 0
        while not stop_flag['stop']:
            epoch += 1
            train_loop(model, loss_fn, opt, X, Y, epochs=1, log_every=1)
            time.sleep(0.01)
        print("\nStopped by user after", epoch, "epochs.")
    else:
        train_loop(model, loss_fn, opt, X, Y, epochs=10000, log_every=1)
    return debug_data

def loss_debug_hook(debug_data):
    def hook(pred, target, output, **kwargs):
        debug_data.append({
            'pred': pred.tolist(),
            'target': target.tolist(),
            'loss': float(output)
        })
    return hook

def main():
    ops = AbstractTensor.get_tensor(faculty=None)
    operators = ['xor', 'and', 'or', 'nand', 'nor']
    loss_types = ['bce', 'mse']
    idx = 0
    while True:
        op = operators[idx % len(operators)]
        for loss_type in loss_types:
            mode = input(f"Run {op.upper()} with {loss_type.upper()} in 'until I stop you' mode? (y/n): ")
            until_stop = (mode.strip().lower() == 'y')
            debug_data = run_operator_demo(
                op, loss_type, ops,
                debug_hooks={'after_forward': loss_debug_hook},
                until_stop=until_stop
            )
            df = pd.DataFrame(debug_data)
            plt.figure()
            plt.plot(df['loss'])
            plt.title(f'{op.upper()} - {loss_type.upper()} Loss per Step')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.show()
        user = input(f"Continue to next operator? (y/n/select [0-{len(operators)-1}]): ")
        if user.lower() == 'n':
            break
        elif user.isdigit() and 0 <= int(user) < len(operators):
            idx = int(user)
        else:
            idx += 1

if __name__ == "__main__":
    main()

