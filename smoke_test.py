from workshoplib.datagen import make_toy_data
from workshoplib.model import make_model
from workshoplib.optimization import make_optimizer

x, y = make_toy_data()
model = make_model()
opt = make_optimizer("adam", model.parameters())

print("x shape:", tuple(x.shape))
print("y shape:", tuple(y.shape))
print("model:", model.__class__.__name__)
print("optimizer:", opt.__class__.__name__)