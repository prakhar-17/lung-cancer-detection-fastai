
from fastai.vision.all import *
from pathlib import Path

# Base path to Google Drive project
BASE = Path('/content/drive/MyDrive/lung-cancer-detection-fastai')

# Data directory with train/valid split
DATA_PATH = BASE / 'data/interim'

# Export location
EXPORT_PATH = BASE / 'exported/lung_model.pkl'


def get_dataloaders(img_size=224, bs=32):
    """
    Loads training and validation data using FastAI DataLoaders.
    """
    dls = ImageDataLoaders.from_folder(
    DATA_PATH,
    train='train',
    valid='valid',
    item_tfms=Resize(img_size),
    batch_tfms=aug_transforms(),
    bs=bs
)
    return dls


def train_model(epochs=5, lr=None):
    """
    Trains a ResNet34 model and exports it.
    Returns: Learner object
    """
    dls = get_dataloaders()

    # Create learner
    learn = cnn_learner(dls, resnet34, metrics=[accuracy])

    # Find optimal learning rate if not provided
    if lr is None:
        lr = learn.lr_find(suggest_funcs=(minimum, steep))[0]
        print(f"📈 Suggested learning rate: {lr:.1e}")

    # Train model
    learn.fine_tune(epochs, base_lr=lr)

    # Export the trained model
    learn.export(EXPORT_PATH)
    print(f"✅ Model exported to: {EXPORT_PATH}")

    return learn
