from ..abstraction import AbstractTensor
from . import Linear, Model, ReLU, MSELoss, Adam, train_loop, set_seed
from .utils import from_list_like

set_seed(0)

# Instantiate a 'like' tensor to choose backend (DEFAULT_FACULTY)
ops = AbstractTensor.get_tensor(faculty=None)

X = from_list_like([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]], like=ops)
Y = from_list_like([[0.0],[1.0],[1.0],[0.0]], like=ops)

model = Model(layers=[Linear(2, 8, like=ops), Linear(8, 1, like=ops)], activation=ReLU())
opt = Adam(model.parameters(), lr=0.1)
loss_fn = MSELoss()

losses = train_loop(model, loss_fn, opt, X, Y, epochs=200, log_every=50)
