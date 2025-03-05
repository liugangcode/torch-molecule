GNN_ENCODER_MODELS = ["gin-virtual", "gcn-virtual", "gin", "gcn"]
GNN_ENCODER_READOUTS = ["sum", "mean", "max"]

GNN_ENCODER_PARAMS = [
    # Model Hyperparameters
    "encoder_type", "readout", "num_layer", "hidden_size", "drop_ratio", "norm_layer",
    # Training Parameters  
    "batch_size", "epochs", "learning_rate", "weight_decay", "grad_clip_value",
    # Scheduler Parameters
    "use_lr_scheduler", "scheduler_factor", "scheduler_patience",
    # Other Parameters
    "fitting_epoch", "fitting_loss", "device", "verbose", "model_name"
]