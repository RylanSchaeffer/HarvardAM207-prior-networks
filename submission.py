# import packages
import numpy as np
import torch
from matplotlib import pyplot as plt

from utils import data, measures, models, plot, run

train_data = data.create_data(
    create_data_functions=[
        data.create_data_mixture_of_gaussians,
    ],
    functions_args=[
        data.mog_three_in_distribution
    ])

labels_np = train_data['targets'].numpy()
samples_np = train_data['samples'].numpy()
train_labels_idx = np.where(labels_np != 3)[0]
x_train_np = samples_np[train_labels_idx]
y_train_np = labels_np[train_labels_idx]

# # works, but I don't want to see it every time
# plot.plot_training_data(
#     samples=train_data['samples'].numpy(),
#     labels=train_data['targets'].numpy(),
#     labels_names=['Cured by Treatment 1', 'Cured by Treatment 2', 'Cured by Treatment 3'],
#     plot_title='Training Data',
#     xaxis=dict(title='Patient Feature 1 (e.g. age)'),
#     yaxis=dict(title='Patient Feature 2 (e.g. BMI)')
# )


# create the model, optimizer, training data
model = run.create_model(in_dim=2, out_dim=3, args={})
optimizer = run.create_optimizer(model=model, args={'lr': 0.001})
loss_fn = run.create_loss_fn(loss_fn_str='kl', args={})

# fit the model
model, optimizer, training_loss = run.train_model(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_data=train_data,
    args={},
    n_epochs=500,
    batch_size=32)

# this works, but I don't want to see the plot
# plot.plot_training_loss(training_loss=training_loss)


new_data = data.create_data(
    create_data_functions=[data.create_data_mixture_of_gaussians, ],
    functions_args=[data.mog_three_in_distribution_one_out])

new_patient_data_indices = new_data['targets'] == 3
new_patient_samples = new_data['samples'][new_patient_data_indices]
new_patient_model_output = model(new_patient_samples)
print(new_patient_model_output['y_pred'].detach().numpy())

plot.plot_decision_surface(model=model,
                           samples=train_data['samples'],
                           labels=train_data['targets'],
                           z_fn=measures.entropy_categorical)
