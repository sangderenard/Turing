
import pandas as pd
import matplotlib.pyplot as plt
from . import Linear, Model, Tanh, Sigmoid, BCEWithLogitsLoss, MSELoss, Adam, train_loop, set_seed
from .utils import from_list_like
from ..abstraction import AbstractTensor

def get_operator_dataset(op_name, like):
    X = from_list_like(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], like=like
    )
    # Zero-center the inputs for tanh stability
    X = X * 2.0 - 1.0
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
    # Use the same learning rate for both losses; BCE needs a higher LR
    opt = Adam(model.parameters(), lr=1e-2)
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
    grad_log = None
    if until_stop:
        t = threading.Thread(target=wait_for_key, daemon=True)
        t.start()
        epoch = 0
        while not stop_flag['stop']:
            epoch += 1
            _, grad_log = train_loop(model, loss_fn, opt, X, Y, epochs=1, log_every=1)
            time.sleep(0.01)
        print("\nStopped by user after", epoch, "epochs.")
    else:
        _, grad_log = train_loop(model, loss_fn, opt, X, Y, epochs=10000, log_every=1)
    return debug_data, grad_log

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
            debug_data, grad_log = run_operator_demo(
                op, loss_type, ops,
                debug_hooks={'after_forward': loss_debug_hook},
                until_stop=until_stop
            )
            df = pd.DataFrame(debug_data)
            plt.figure(figsize=(10, 6))
            plt.plot(df['loss'], label='Loss')
            # Normalize gradients to loss range for overlay
            if grad_log is not None:
                import numpy as np
                loss_arr = np.array(df['loss'])
                grad_req = np.array(grad_log['requested'])
                grad_cap = np.array(grad_log['capped'])
                grad_preclip = np.array([
                    v if v is not None else np.nan for v in grad_log['preclip']
                ])
                # Normalize to loss range
                def norm_to_loss(arr):
                    arr = np.array(arr)
                    if np.all(np.isnan(arr)):
                        return arr
                    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
                    loss_min, loss_max = np.min(loss_arr), np.max(loss_arr)
                    if arr_max - arr_min < 1e-8:
                        return np.full_like(arr, loss_min)
                    return (arr - arr_min) / (arr_max - arr_min) * (loss_max - loss_min) + loss_min
                grad_req_norm = norm_to_loss(grad_req)
                grad_cap_norm = norm_to_loss(grad_cap)
                grad_preclip_norm = norm_to_loss(grad_preclip)
                plt.plot(grad_req_norm, label='Requested Grad (normed)', linestyle='--', alpha=0.7)
                plt.plot(grad_cap_norm, label='Capped Grad (normed)', linestyle='-.', alpha=0.7)
                if not np.all(np.isnan(grad_preclip_norm)):
                    plt.plot(grad_preclip_norm, label='Pre-clip Grad (normed)', linestyle=':', alpha=0.7)
            plt.title(f'{op.upper()} - {loss_type.upper()} Loss & Gradients per Step')
            plt.xlabel('Step')
            plt.ylabel('Loss / Normalized Grad')
            plt.legend()
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

