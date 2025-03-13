import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import tiny_diffusion.datasets as datasets
from tiny_diffusion.model import MLP
from tiny_diffusion.noise_schedulers import NoiseScheduler

import mlflow
import tempfile

class MLFlowLogger:
    def __init__(self, config):
        self.config = config
        mlflow.set_experiment(config.experiment_name)
        mlflow.start_run()
        mlflow.log_params(vars(config))

    def log_metrics(self, metrics, step):
        mlflow.log_metrics(metrics, step=step)

    def end_run(self):
        mlflow.end_run()

    def checkpoint_model(self, model,step):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = f"{tmpdir}/model.pth"
            torch.save(model.state_dict(), tmpfile)
            mlflow.log_artifact(tmpfile, artifact_path="models")


def parse_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=10)
    config = parser.parse_args()
    return config

class Trainer:
    def __init__(self, config):
        self.config = config
        # load dataset
        self.dataset = datasets.get_dataset(config.dataset)
        # create dataloader
        self.dataloader = DataLoader(
            self.dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
        # create model
        self.model = MLP(
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            emb_size=config.embedding_size,
            time_emb=config.time_embedding,
            input_emb=config.input_embedding)
        # create noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_schedule=config.beta_schedule)
        # create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        # track number of steps completed in training
        self.global_step = 0
        # track generated images
        self.frames = []
        # track batch losses
        self.losses = []
        # create MLFlow logger
        self.logger = MLFlowLogger(config)

    def train(self):
        print("Training model...")
        for epoch in range(self.config.num_epochs):
            self.model.train()
            progress_bar = tqdm(total=len(self.dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for batch in self.dataloader:
                batch = batch[0]
                noise = torch.randn(batch.shape)
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, (batch.shape[0],)
                ).long()

                noisy = self.noise_scheduler.add_noise(batch, noise, timesteps)
                noise_pred = self.model(noisy, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": self.global_step}
                self.losses.append(loss.detach().item())
                progress_bar.set_postfix(**logs)
                self.global_step += 1
                self.logger.log_metrics({"loss": loss.detach().item()}, self.global_step)
            progress_bar.close()

            if epoch % self.config.save_images_step == 0 or epoch == self.config.num_epochs - 1:
                self.generate_images()
                self.logger.checkpoint_model(self.model, self.global_step)
        self.logger.end_run()

    def generate_images(self):
        self.model.eval()
        sample = torch.randn(self.config.eval_batch_size, 2)
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, self.config.eval_batch_size)).long()
            with torch.no_grad():
                residual = self.model(sample, t)
            sample = self.noise_scheduler.step(residual, t[0], sample)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(sample[:, 0], sample[:, 1])
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            img_path = f"/tmp/frame_{i:04}.png"
            fig.savefig(img_path)
            plt.close(fig)
            mlflow.log_artifact(img_path, artifact_path="frames")

    def save_images(self):
        print("Saving images...")
        outdir = f"exps/{self.config.experiment_name}"
        imgdir = f"{outdir}/images"
        for i, frame in enumerate(self.frames):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(frame[:, 0], frame[:, 1])
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            img_path = f"{imgdir}/{i:04}.png"
            fig.savefig(img_path)
            plt.close(fig)
            mlflow.log_artifact(img_path, artifact_path="images")

if __name__ == "__main__":
    config = parse_config()
    print("Starting training...")
    trainer = Trainer(config)
    trainer.train()
    print("Training complete.")
    mlflow.end_run()
