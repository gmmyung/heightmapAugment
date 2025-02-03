import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.data import TensorDataset, DataLoader


class FootHoldPredictor:
    def __init__(self, cfg, writer):
        self.num_envs = cfg["environment"]["num_envs"]
        # self.footholds = [[] for _ in range(self.num_envs)]
        self.sample_per_env = 10000
        self.footholds = [
            deque(maxlen=self.sample_per_env) for _ in range(self.num_envs)
        ]
        self.flattened_footholds = None

        self.input_dim = 7
        self.hidden_dim = 32
        self.output_dim = 28
        self.seq_len = 20

        self.lstm_model = LSTMFootholdModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            output_dim=self.output_dim,
        )

        # Custom loss for footID + 4 separate Gaussians
        self.criterion = MixedFootholdProbLoss()
        self.optimizer = optim.Adam(self.lstm_model.parameters(), lr=1e-3)

        self.writer = writer

    def step(self, coords):
        # Vectorized contact check using sum of squares for speed
        contacts = np.sum(coords**2, axis=-1) > 0.0  # shape (num_envs, 4)

        # Find indices where contacts occur
        indices = np.where(contacts)
        num_contacts = indices[0].shape[0]
        if num_contacts == 0:
            return  # No contacts, nothing to process

        # Preallocate array for new footholds for all contacts at once
        new_footholds = np.zeros((num_contacts, 7))

        # Use indices to set one-hot encoding for foot information
        new_footholds[np.arange(num_contacts), indices[1]] = 1.0

        # Fill coordinate part for each contact
        new_footholds[:, 4:7] = coords[indices[0], indices[1]]

        # Add new footholds to the deque
        for env_idx, foothold in zip(indices[0], new_footholds):
            if len(self.footholds[env_idx]) > 0:
                foothold[4:7] = foothold[4:7] - self.footholds[env_idx][-1][4:7]
                self.footholds[env_idx].append(foothold)
                if len(self.footholds[env_idx]) > self.sample_per_env:
                    self.footholds[env_idx].popleft()
            else:
                self.footholds[env_idx].append(foothold)

    def flatten_footholds(self):
        # Use list comprehension to stack arrays for each non-empty environment
        stacked = [np.stack(flist) for flist in self.footholds if len(flist) > 0]
        if stacked:
            self.flattened_footholds = np.concatenate(stacked, axis=0)
        else:
            self.flattened_footholds = np.empty((0, 7))

    def reset(self):
        # Reset footholds storage
        self.footholds = [
            deque(maxlen=self.sample_per_env) for _ in range(self.num_envs)
        ]
        self.flattened_footholds = None

    # ------------------------------------------------------
    # 2) Helper to create sequences from flattened footholds
    # ------------------------------------------------------
    def create_sequences(self, data, seq_len):
        """
        data: np.ndarray of shape (N, 7)
        seq_len: int, length of each sequence

        Returns two arrays:
            X of shape (num_sequences, seq_len, 7)
            y of shape (num_sequences, 7)
        where y is the next foothold after each sequence.
        """
        sequences = []
        targets = []

        # We want to shift by 1 to predict the *next* foothold
        for i in range(len(data) - seq_len):
            seq = data[i : i + seq_len]
            target = data[i + seq_len]  # next foothold
            sequences.append(seq)
            targets.append(target)

        X = np.array(sequences)
        y = np.array(targets)
        return X, y

    # ------------------------------------------------------
    # 3) Training loop for the LSTM
    # ------------------------------------------------------

    def train_lstm(self, epochs=1, update=0, batch_size=2048):
        self.lstm_model.train()
        if self.flattened_footholds is None or len(self.flattened_footholds) == 0:
            print("No foothold data to train on.")
            return

        X, y = self.create_sequences(self.flattened_footholds, self.seq_len)
        if len(X) == 0:
            print("Not enough data to form sequences.")
            return

        # Convert NumPy arrays to PyTorch tensors
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).float()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model.to(device)

        # Create a Dataset and DataLoader for mini-batching
        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.lstm_model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                self.optimizer.zero_grad()

                # Forward pass
                predictions, _ = self.lstm_model(batch_X)

                # Compute loss
                loss = self.criterion(predictions, batch_y)

                # Backprop and update
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Optionally, record the final loss on TensorBoard
        self.writer.add_scalar("Loss/train", avg_loss, update)

    # ------------------------------------------------------
    # 4) Use model to predict next foothold(s)
    # ------------------------------------------------------
    def predict_next_foothold(self, seq):
        self.lstm_model.eval()
        seq_torch = torch.from_numpy(seq).float().unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_torch = seq_torch.to(device)
        self.lstm_model.to(device)

        with torch.no_grad():
            pred, _ = self.lstm_model(seq_torch)
            pred = pred[0]  # (28,)

        # 1) Foot ID
        foot_id_logits = pred[:4]
        foot_id_probs = torch.softmax(foot_id_logits, dim=-1)

        # 2) Extract the 4 separate Gaussians
        dist_params = pred[4:].reshape(4, 6)
        mu_all = dist_params[:, :3]
        log_sigma_all = dist_params[:, 3:]
        sigma_all = torch.exp(log_sigma_all)

        return (
            foot_id_probs.cpu().numpy(),
            mu_all.cpu().numpy(),
            sigma_all.cpu().numpy(),
        )

    def log_probability(self, coords):
        """
        Computes the log probability for all newly stepped contacts in `coords`
        using the last `self.seq_len` footholds from each environment in
        `self.footholds` as the past context. Returns an array of shape
        (num_envs,) with the sum of log probabilities for each environment's
        contacts. If there is no contact or not enough foothold history, 0.0
        is placed for that environment.

        Args:
            coords (np.ndarray): shape (num_envs, 4, 3)
                Coordinates for potential new contacts. A value of (x,y,z) = 0
                indicates no contact. Non-zero => contact.

        Returns:
            log_probs (np.ndarray): shape (num_envs,)
                The sum of log probabilities for each environment’s newly
                stepped contact(s). 0.0 if no contact or not enough data.
        """
        import math

        num_envs = self.num_envs
        seq_len = self.seq_len

        # --------------------------------------------------------
        # 1) Identify (env, foot) pairs that actually made contact
        # --------------------------------------------------------
        contacts = np.sum(coords**2, axis=-1) > 0.0  # shape => (num_envs, 4)
        env_idxs, foot_idxs = np.where(contacts)  # each is 1D array
        num_contacts = len(env_idxs)
        if num_contacts == 0:
            # No contacts in any env => return zeros
            return np.zeros(num_envs, dtype=np.float32)

        # --------------------------------------------------------
        # 2) Build "next foothold" data for each contact
        #    shape => (num_contacts, 7)
        # --------------------------------------------------------
        new_footholds = np.zeros((num_contacts, 7), dtype=np.float32)
        new_footholds[np.arange(num_contacts), foot_idxs] = 1.0  # one-hot foot ID
        for i in range(num_contacts):
            e_i = env_idxs[i]
            f_i = foot_idxs[i]
            new_footholds[i, 4:7] = coords[e_i, f_i]

        # --------------------------------------------------------
        # 3) Gather last seq_len footholds for each contact
        #    Skip if not enough data in self.footholds[e_i].
        #    We'll accumulate valid samples in lists.
        # --------------------------------------------------------
        valid_past_sequences = []
        valid_next_footholds = []
        valid_envs = []  # which env each sample belongs to

        for i in range(num_contacts):
            e_i = env_idxs[i]
            foot_history = self.footholds[e_i]  # list of arrays with shape (7,)
            if len(foot_history) < seq_len:
                # Not enough data => skip
                continue
            # Gather the last seq_len footholds
            past_seq = np.array(
                # foot_history[-seq_len:], dtype=np.float32
                list(foot_history)[-seq_len:],
                dtype=np.float32,
            )  # (seq_len, 7)
            next_foot = new_footholds[i]  # (7,)

            valid_past_sequences.append(past_seq)
            valid_next_footholds.append(next_foot)
            valid_envs.append(e_i)  # remember which env this belongs to

        # If we have no valid samples, all envs get 0.0
        if len(valid_past_sequences) == 0:
            return np.zeros(num_envs, dtype=np.float32)

        # --------------------------------------------------------
        # 4) Stack into batch => single forward pass
        # --------------------------------------------------------
        X = np.stack(valid_past_sequences, axis=0)  # shape => (B, seq_len, 7)
        Y = np.stack(valid_next_footholds, axis=0)  # shape => (B, 7)
        B = X.shape[0]

        self.lstm_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_torch = torch.from_numpy(X).float().to(device)  # (B, seq_len, 7)
        Y_torch = torch.from_numpy(Y).float().to(device)  # (B, 7)
        self.lstm_model.to(device)

        with torch.no_grad():
            # pred => (B, 28)
            pred, _ = self.lstm_model(X_torch)

        # --------------------------------------------------------
        # 5) Parse out foot ID + coords to get log p
        # --------------------------------------------------------
        foot_id_logits = pred[:, :4]  # (B, 4)
        foot_id_log_probs = torch.log_softmax(foot_id_logits, dim=1)  # (B, 4)

        foot_id_true_one_hot = Y_torch[:, :4]  # (B, 4)
        foot_id_true_class = foot_id_true_one_hot.argmax(dim=1)  # (B,)

        row_idx = torch.arange(B, device=device)
        log_p_foot_id = foot_id_log_probs[row_idx, foot_id_true_class]  # => (B,)

        dist_params = pred[:, 4:].reshape(B, 4, 6)  # (B, 4, 6)
        mu_all = dist_params[:, :, :3]  # (B, 4, 3)
        log_sigma_all = dist_params[:, :, 3:]  # (B, 4, 3)

        coords_true = Y_torch[:, 4:7]  # (B, 3)

        mu_selected = mu_all[row_idx, foot_id_true_class, :]  # (B, 3)
        log_sigma_selected = log_sigma_all[row_idx, foot_id_true_class, :]  # (B, 3)
        sigma_selected = torch.exp(log_sigma_selected)  # (B, 3)

        diff = coords_true - mu_selected
        # diag Gaussian log prob for 3D coords
        log_prob_dims = (
            -0.5 * (diff**2 / (sigma_selected**2))
            - log_sigma_selected
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_p_coords = log_prob_dims.sum(dim=1)  # => (B,)

        log_p_total = log_p_foot_id + log_p_coords  # => (B,)

        # --------------------------------------------------------
        # 6) Accumulate log prob into (num_envs,) array
        #    - If an env has multiple contacts, we sum them.
        #    - If an env has none or is invalid, it remains 0.0.
        # --------------------------------------------------------
        log_probs = np.zeros(num_envs, dtype=np.float32)

        log_p_total_np = log_p_total.cpu().numpy()  # shape => (B,)
        for i in range(B):
            e_i = valid_envs[i]
            log_probs[e_i] += log_p_total_np[i]

        return log_probs

    def plot_evaluation(self, update):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # We need at least 20 points to form a single sequence + next-step prediction
        if self.flattened_footholds is None or len(self.flattened_footholds) < 20:
            print("Not enough data to plot.")
            return

        # We'll plot up to 9 subsequences
        max_plots = min(9, len(self.flattened_footholds) - 20)

        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        ax = ax.flatten()  # flatten for easy indexing

        for idx in range(max_plots):
            # Slice out 20 consecutive footholds
            seq = self.flattened_footholds[idx : idx + 20]

            # Predict from the first 19 frames
            foot_id_probs, mu_all, sigma_all = self.predict_next_foothold(seq[:-1])

            # Time-index for coloring actual data
            t = np.arange(seq.shape[0])  # 0..19
            # Plot the (x, y) for the entire 20-step window, colored by time
            ax[idx].scatter(seq[:, 4], seq[:, 5], c=t, cmap="viridis")

            # Plot predicted means + 1-sigma ellipse for each of the 4 feet
            colors = ["red", "green", "blue", "orange"]
            for foot_i in range(4):
                mu_x, mu_y = mu_all[foot_i, 0], mu_all[foot_i, 1]
                sigma_x, sigma_y = sigma_all[foot_i, 0], sigma_all[foot_i, 1]

                # Probability determines how opaque (alpha) the foot is
                alpha_val = float(foot_id_probs[foot_i])

                # Scatter the predicted mean
                ax[idx].scatter(
                    mu_x, mu_y, color=colors[foot_i], marker="x", s=80, alpha=alpha_val
                )

                # Draw the 1-sigma ellipse (diagonal covariance => no rotation)
                ellipse = patches.Ellipse(
                    xy=(mu_x, mu_y),
                    width=2 * sigma_x,  # 1-sigma => diameter is 2*sigma
                    height=2 * sigma_y,
                    angle=0,  # no rotation for diagonal
                    fill=False,
                    edgecolor=colors[foot_i],
                    alpha=alpha_val,
                )
                ax[idx].add_patch(ellipse)

        plt.tight_layout()

        # Add figure to TensorBoard
        self.writer.add_figure("EvaluationPlot", fig, global_step=update)

        # Close the figure to free memory
        plt.close(fig)


class LSTMFootholdModel(nn.Module):
    """
    Outputs 28 dimensions:
      - indices [0..3]: foot ID logits  (4 values)
      - for each foot i in {0,1,2,3}:
          mu_i(3) + log_sigma_i(3) = 6
        i.e. foot0: [4..9]
             foot1: [10..15]
             foot2: [16..21]
             foot3: [22..27]
    """

    def __init__(self, input_dim=7, hidden_dim=32, num_layers=1, output_dim=28):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.shape[0]
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, device=x.device
            )
            c_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, device=x.device
            )
            hidden = (h_0, c_0)

        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        # Take the last time-step
        last_output = lstm_out[:, -1, :]  # shape => (B, hidden_dim)
        preds = self.fc(last_output)  # shape => (B, 28)
        return preds, (h_n, c_n)


class MixedFootholdProbLoss(nn.Module):
    """
    For each sample:
      1) Cross-entropy for foot ID (4-class).
      2) NLL for coordinates, but only for the foot that stepped.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        pred: (B, 28) => [foot_id_logits(4),
                          mu_foot0(3), log_sigma_foot0(3),
                          mu_foot1(3), log_sigma_foot1(3),
                          mu_foot2(3), log_sigma_foot2(3),
                          mu_foot3(3), log_sigma_foot3(3)]
        target: (B, 7) => [footID one-hot(4), coords(3)]
        """
        B = pred.shape[0]
        device = pred.device

        # 1) Foot ID classification
        foot_id_pred = pred[:, :4]  # (B, 4)
        foot_id_true_one_hot = target[:, :4]  # (B, 4)
        foot_id_true_class = foot_id_true_one_hot.argmax(dim=1)  # (B,)
        foot_id_loss = self.ce(foot_id_pred, foot_id_true_class)

        # 2) Per-foot Gaussian NLL
        coords_true = target[:, 4:7]  # (B, 3)

        # Reshape the distribution parameters for convenience:
        # shape => (B, 4, 6)
        # i.e. foot0 => indexes [4..9], foot1 => [10..15], foot2 => [16..21], foot3 => [22..27]
        # We'll chunk them into (B,4,3) for mu, and (B,4,3) for log_sigma
        dist_params = pred[:, 4:].reshape(B, 4, 6)  # => (B, 4, 6)
        mu_all = dist_params[:, :, :3]  # => (B, 4, 3)
        log_sigma_all = dist_params[:, :, 3:]  # => (B, 4, 3)

        # For each sample in the batch, pick out the relevant foot distribution.
        # foot_id_true_class is in {0,1,2,3}
        row_idx = torch.arange(B, device=device)  # 0..B-1
        foot_idx = foot_id_true_class  # shape => (B,)

        # Gather the mu and log_sigma for the actual foot:
        # shape => (B, 3)
        mu_selected = mu_all[row_idx, foot_idx, :]
        log_sigma_selected = log_sigma_all[row_idx, foot_idx, :]

        sigma_selected = torch.exp(log_sigma_selected)

        # Negative log-likelihood for diagonal Gaussian
        # NLL = 1/2 * sum( ((x - mu)/sigma)^2 + 2*log_sigma )
        nll_per_batch = 0.5 * torch.sum(
            ((coords_true - mu_selected) / sigma_selected) ** 2
            + 2 * log_sigma_selected,
            dim=1,
        )
        coords_loss = nll_per_batch.mean()

        total_loss = foot_id_loss + coords_loss
        return total_loss

def main():
    pass

if __name__ == "__main__":
    main()

    
