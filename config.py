from scripts.mnist.utils import to_nametuple

device = "cpu"

vxm = {
    "default": to_nametuple({
        # train
        'lr': 1e-3,
        'ndim': 2,
        'epochs': 10,
        'batch_size_train': 32,
        'log_interval': 10,
        'steps_per_epoch': 30,
        'device': device,
        'inshape': (32, 32),

        # Model
        'fix': 5,
        'moving': 5,
        'λ': 0.5,
        'image_loss': "mse",
    }),

    "lambda-0": to_nametuple({
        # train
        'lr': 1e-3,
        'ndim': 2,
        'epochs': 10,
        'batch_size_train': 32,
        'log_interval': 10,
        'steps_per_epoch': 30,
        'device': device,
        'inshape': (32, 32),

        # Model
        'fix': 5,
        'moving': 5,
        'λ': 0,
        'image_loss': "mse",
    }),

    "lambda-0_01": to_nametuple({
        # train
        'lr': 1e-3,
        'ndim': 2,
        'epochs': 10,
        'batch_size_train': 32,
        'log_interval': 10,
        'steps_per_epoch': 30,
        'device': device,
        'inshape': (32, 32),

        # Model
        'fix': 5,
        'moving': 5,
        'λ': 0.01,
        'image_loss': "mse",
    })
}

inverse = {
    "default": to_nametuple({
        # train
        'lr': 1e-3,
        'ndim': 2,
        'epochs': 10,
        'batch_size_train': 32,
        'log_interval': 10,
        'steps_per_epoch': 30,
        'device': device,
        'inshape': (32, 32),

        # Model
        'fix': 5,
        'moving': 5,
        'image_loss': "mse",
        'inverse': 0,
        'antifold': 100000,
        'smooth': 0.5
    }),
}
