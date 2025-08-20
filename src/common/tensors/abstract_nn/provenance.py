import pandas as pd
from collections import defaultdict

class ProvenanceTracker:
    def __init__(self):
        self.history = defaultdict(list)

    def record(self, step, loss, model=None):
        self.history['step'].append(step)
        self.history['loss'].append(loss)
        if model is not None:
            # Record per-layer weight/grad norms
            for i, layer in enumerate(getattr(model, 'layers', [])):
                W = getattr(layer, 'W', None)
                gW = getattr(layer, 'gW', None)
                if W is not None:
                    self.history[f'layer_{i}_W_norm'].append(float(((W * W).sum()).sqrt().item()))
                if gW is not None:
                    self.history[f'layer_{i}_gW_norm'].append(float(((gW * gW).sum()).sqrt().item()))

    def to_dataframe(self):
        return pd.DataFrame(self.history)

    def plot(self):
        import matplotlib.pyplot as plt
        df = self.to_dataframe()
        df.plot(x='step', y='loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.show()
