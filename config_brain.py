from scripts.mnist.utils import to_nametuple


vxm = {
    "default": to_nametuple({
        # train
        'lr': 1e-4,
        'ndim': 2,
        'epochs': 30,
        'batch_size_train': 32,
        'log_interval': 30,
        'steps_per_epoch': 30,
        'inshape': (192, 160),

        # Model
        'fix': 5,
        'moving': 5,
        'λ': 0.5,
        'image_loss': "mse",
        'nb_features': [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
        ]
    }),

    "lambda-0": to_nametuple({
        # train
        'lr': 1e-4,
        'ndim': 2,
        'epochs': 30,
        'batch_size_train': 32,
        'log_interval': 10,
        'steps_per_epoch': 30,
        'inshape': (192, 160),

        # Model
        'fix': 5,
        'moving': 5,
        'λ': 0,
        'image_loss': "mse",
        'nb_features': [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
        ]
    }),

    "lambda-0_01": to_nametuple({
        # train
        'lr': 1e-4,
        'ndim': 2,
        'epochs': 30,
        'batch_size_train': 32,
        'log_interval': 10,
        'steps_per_epoch': 30,
        'inshape': (192, 160),

        # Model
        'fix': 5,
        'moving': 5,
        'λ': 0.01,
        'image_loss': "mse",
        'nb_features': [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
        ]
    })
}

inverse = {
    "default": to_nametuple({
        # train
        'lr': 1e-4,
        'ndim': 2,
        'epochs': 30,
        'batch_size_train': 32,
        'log_interval': 10,
        'steps_per_epoch': 30,
        'inshape': (192, 160),

        # Model
        'fix': 5,
        'moving': 5,
        'image_loss': "mse",
        'inverse': 0.05,
        'antifold': 100000,
        'smooth': 0.5,
        'nb_features': [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
        ]
    }),
    
    
}
