"""
ESA SCOPE project DOC model

This module contains the DOC model for the ESA SCOPE project.
The model is based on the Torch library.

"""

import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
# from sklearn.inspection import permutation_importance
from sklearn import preprocessing

from scope_config import Rrs, units, MODEL_DIR

def get_torch_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    return device


# data loader
class DOCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, variables: list, Rrs: list, val_size: float, batch_size: int = 16, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.variables = variables
        self.val_size = val_size
        self.batch_size = batch_size
        self.Rrs = Rrs
        self.names = self.Rrs + self.variables
        self.num_workers = num_workers


    def setup(self, stage=None):
        allvars = ['DOC'] + self.Rrs + self.variables
        data = pd.read_hdf(self.data_dir)
        data = data.sort_values(by='time')
        data = data[allvars].dropna()
        y = data['DOC'].values
        X = data[self.Rrs + self.variables].values.copy()
        nobs = len(y)
        self.X = X
        self.y = y

        self.scaler = preprocessing.StandardScaler().fit(X)
        Xtrans = self.scaler.transform(X)
        st = np.repeat(np.c_[np.zeros(50), np.ones(50)], nobs//100 + 100)[:nobs]
        X_train, X_val, y_train, y_val = train_test_split(Xtrans, y,
                                                    test_size=self.val_size, shuffle=True, stratify=st)
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    def train_dataloader(self):
        spectral = self.X_train[:, :len(self.Rrs)].unsqueeze(1)
        features = self.X_train[:, len(self.Rrs):]
        dataset = TensorDataset(spectral, features, self.y_train)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        spectral = self.X_val[:, :len(self.Rrs)].unsqueeze(1)
        features = self.X_val[:, len(self.Rrs):]
        dataset = TensorDataset(spectral, features, self.y_val)
        return DataLoader(dataset, batch_size=1, shuffle=False,
                          num_workers=self.num_workers)
    

# model
class DOCModule(pl.LightningModule):
    def __init__(self, **hparams):
        super(DOCModule, self).__init__()
        self.save_hyperparameters()
        self.lr = hparams['lr']
        self.num_features = hparams['num_features']
        self.num_Rrs = hparams['num_Rrs']

        self.Rrs_encoder_channels = 8
        self.Rrs_encoder_kernel = 2
        
        self.Rrs_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.Rrs_encoder_channels, kernel_size=self.Rrs_encoder_kernel),
            nn.ReLU(),
            nn.Flatten(),
        )
        Rrs_encoder_shape = (self.num_Rrs - (self.Rrs_encoder_kernel - 1)) * self.Rrs_encoder_channels ## == 40
        self.linear = nn.Sequential(
            nn.Linear(Rrs_encoder_shape + self.num_features, 32),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, Rrs, features):
        encoded_Rrs = self.Rrs_encoder(Rrs)
        x = self.linear(torch.cat((encoded_Rrs, features), dim=1))
        x = self.out(x)
        return x
    
    def loss_fn(self, prediction, target):
        return F.mse_loss(prediction, target)
    
    
    def training_step(self, batch, batch_idx):
        Rrs_batch, features_batch, y_batch = batch
        outputs = self.forward(Rrs_batch, features_batch)
        loss = self.loss_fn(outputs, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    
    def validation_step(self, batch, batch_idx):
        Rrs_batch, features_batch, y_batch = batch
        outputs = self.forward(Rrs_batch, features_batch)
        loss = self.loss_fn(outputs, y_batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        spectral = X[:, :self.num_Rrs].unsqueeze(1)
        features = X[:, self.num_Rrs:]
        with torch.no_grad():
            y = self.eval()(spectral, features)
            y = y.cpu().numpy()
        return y


def load_model(model_version, modeldir=MODEL_DIR, device='mps'):
    
    datafile = os.path.join(modeldir, f'torch_data_{model_version}.pkl')
    modelfile = os.path.join(modeldir, f'torch_model_{model_version}.pth')

    data = pickle.load(open(datafile, 'rb'))

    checkpoint = torch.load(modelfile, map_location=device, weights_only=True)
    hparams = checkpoint["hyper_parameters"]
    model = DOCModule(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    return data, model


def save_model(model_version, data, model, modeldir=MODEL_DIR, device='mps'):
    
    datafile = os.path.join(modeldir, f'torch_data_{model_version}.pkl')
    modelfile = os.path.join(modeldir, f'torch_model_{model_version}.pth')

    pickle.dump(
        {
            'X_val': data.X_val.cpu().numpy(),
            'y_val': data.y_val.cpu().numpy(),
            'X_train': data.X_train.cpu().numpy(),
            'y_train': data.y_train.cpu().numpy(),
            'names': data.names,
            'scaler': data.scaler,
            'model_version': model_version,
            'model_type': 'torch'
            },
            open(datafile, 'wb'))

    if isinstance(model, str):
        model = torch.load(model, map_location=device, weights_only=True)

    torch.save(model, modelfile)
    return


def estimate_DOC(ds, model, data, mindoc=0.0001):
    """Estimate DOC using the model."""
    names = data['names']
    scaler = data['scaler']
    df = ds.to_dataframe()  # to pandas data frame
    df = df.reset_index(level=['lat', 'lon'])
    X = df[names].copy().values
    X = scaler.transform(X)  # scale input data
    inds = np.isfinite(X).all(axis=1)

    mpred = np.zeros(X.shape[0], dtype=np.float32) * np.nan
    mpred[inds] = model.predict(X[inds, :]).ravel()
    mpred = np.maximum(mpred, mindoc)  # some predictions might be negative
    
    out = xr.Dataset(coords=ds.coords)  # generate new dataset for output
    out['DOC'] = (['lat', 'lon'], mpred.reshape((len(out.lat), len(out.lon))))
    out['DOC'].attrs['long_name'] = f'estimated DOC'
    out['DOC'].attrs['units'] = units['DOC']
    return out


def get_trainer(save_dir="/tmp", accelerator='mps', max_epochs=400):

    tb_logger = TensorBoardLogger(save_dir=save_dir)

    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                      monitor="val_loss_epoch",
                                      mode="min")

    early_stopping_callback = EarlyStopping(monitor="val_loss_epoch",
                                        mode="min", patience=10)
    trainer = pl.Trainer(
        accelerator=accelerator,
        enable_progress_bar=True,
        enable_checkpointing=True,
        logger=tb_logger,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=4
    )
    return trainer

