# Abstract NN Hook System (Patch Panel)

The `HookPanel` system in `abstract_nn` provides a universal, extensible way to observe, log, and modify the behavior of neural network training and inference. It is designed for diagnostics, provenance, and future extensibility.

## How to Use

### 1. Registering a Hook
Hooks are functions that are called at specific events during training or inference. You can register a hook for any event:

```python
from src.common.tensors.abstract_nn.hooks import hook_panel

def my_loss_logger(model, pred, y, loss, **kwargs):
    print(f"Loss: {loss}")

hook_panel.register('loss', my_loss_logger)
```

### 2. Removing a Hook
```python
hook_panel.remove('loss', my_loss_logger)
```

### 3. Enabling Documentation Policy
This controls whether detailed documentation/recording is enabled for provenance hooks:
```python
hook_panel.enable_document_policy(True)
```

### 4. Clearing Hooks
```python
hook_panel.clear()  # Remove all hooks
hook_panel.clear('loss')  # Remove all hooks for 'loss' event
```

### 5. Writing Custom Hooks
Hooks can accept any arguments relevant to the event. See event documentation below for available arguments.

---

## Event Names and Their Meanings

| Event Name    | When It Fires                        | Arguments Passed                                                      |
|--------------|--------------------------------------|-----------------------------------------------------------------------|
| `step_start` | At the start of each training step    | `model`, `x`, `y`                                                     |
| `forward`    | After model forward pass              | `model`, `x`, `pred`                                                  |
| `loss`       | After loss is computed                | `model`, `pred`, `y`, `loss`                                          |
| `backward`   | After loss backward (grad computed)   | `model`, `grad_pred`                                                  |
| `step_end`   | At the end of each training step      | `model`, `x`, `y`, `loss`                                             |
| `epoch_end`  | At the end of each epoch              | `epoch`, `loss`                                                       |
| `log`        | When a log/print would occur          | `epoch`, `loss`                                                       |
| `debug`      | During debug output (if enabled)      | `layer`, `i`, `W`, `gW`, `b0`                                         |

- All hooks for an event are called in the order they were registered.
- You can register multiple hooks for the same event.
- Hooks can be used for logging, provenance, visualization, or even modifying behavior.

---

## Example: Collecting Loss History

```python
import pandas as pd
from src.common.tensors.abstract_nn.hooks import hook_panel

losses = []
def collect_loss(epoch, loss, **kwargs):
    losses.append((epoch, loss))

hook_panel.register('epoch_end', collect_loss)
# ...run training...
df = pd.DataFrame(losses, columns=['epoch', 'loss'])
```

---

## Extending
You can add new events to the system by calling `hook_panel.run('my_event', ...)` at any point in your code.

---

## Disabling/Enabling Documentation
Use `hook_panel.enable_document_policy(True/False)` to control whether hooks that record or document are active.

---

For more advanced usage, see the source code in `src/common/tensors/abstract_nn/hooks.py`.
