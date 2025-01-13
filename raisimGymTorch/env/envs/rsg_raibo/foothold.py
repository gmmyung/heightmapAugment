import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class FootHoldPredictor:
    def __init__(self, cfg, writer):
        self.num_envs = cfg["environment"]["num_envs"]
        self.footholds = [[] for _ in range(self.num_envs)]
        self.flattened_footholds = None
        self.eye4 = np.eye(4)  # Precomputed identity matrix for one-hot encoding

        # -----------------------------
        # 1) Instantiate the LSTM model
        # -----------------------------
        self.input_dim = 7  # same as the data dimension (4 one-hot + (x,y,z))
        self.hidden_dim = 32
        self.output_dim = 7
        self.seq_len = 20  # how many time steps per sequence, tweak as needed

        self.lstm_model = LSTMFootholdModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            output_dim=self.output_dim,
        )

        # Define an optimizer and loss function
        self.optimizer = optim.Adam(self.lstm_model.parameters(), lr=1e-3)
        self.criterion = MixedFootholdLoss()
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

        for env_idx, foothold in zip(indices[0], new_footholds):
            if len(self.footholds[env_idx]) > 0:
                foothold[4:7] = foothold[4:7] - self.footholds[env_idx][-1][4:7]
                self.footholds[env_idx].append(foothold)
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
        self.footholds = [[] for _ in range(self.num_envs)]
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
    def train_lstm(self, epochs=10, update=0):
        """
        Example training loop. Make sure `self.flattened_footholds`
        is populated (call flatten_footholds first).
        """
        self.lstm_model.train()
        if self.flattened_footholds is None or len(self.flattened_footholds) == 0:
            print("No foothold data to train on.")
            return

        # Create sequences
        X, y = self.create_sequences(self.flattened_footholds, self.seq_len)

        if len(X) == 0:
            print("Not enough data to form sequences.")
            return

        # Convert to torch tensors
        X_torch = torch.from_numpy(X).float()  # shape: (B, T, 7)
        y_torch = torch.from_numpy(y).float()  # shape: (B, 7)

        # Move to GPU if available (optional)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model.to(device)
        X_torch = X_torch.to(device)
        y_torch = y_torch.to(device)

        # Training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Forward pass
            predictions, _ = self.lstm_model(X_torch)
            # predictions => (B, 7)

            # Compute loss
            loss = self.criterion(predictions, y_torch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        self.writer.add_scalar("Loss/train", loss.item(), update)

    # ------------------------------------------------------
    # 4) Use model to predict next foothold(s)
    # ------------------------------------------------------
    def predict_next_foothold(self, seq):
        """
        seq: np.ndarray of shape (seq_len, 7) - the input sequence
        returns: np.ndarray of shape (7,) - the predicted next foothold
        """
        self.lstm_model.eval()

        # Convert seq to torch
        seq_torch = torch.from_numpy(seq).float().unsqueeze(0)  # shape => (1, T, 7)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_torch = seq_torch.to(device)
        self.lstm_model.to(device)

        with torch.no_grad():
            pred, _ = self.lstm_model(seq_torch)

        # pred => (1, 7)
        return pred[0].cpu().numpy()

    def plot_evaluation(self, update):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(3, 3)
        for idx in range(9):
            seq = self.flattened_footholds[idx : idx + 20]
            pred = self.predict_next_foothold(seq[:-1])

            t = np.arange(20)
            
            ax[idx // 3, idx % 3].scatter(seq[:, 4], seq[:, 5], c=t)
            ax[idx // 3, idx % 3].scatter(pred[4], pred[5], c="red")


class LSTMFootholdModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=32, num_layers=1, output_dim=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )  # batch_first => input shape (B, T, D)

        # A fully connected layer to map the LSTM outputs to desired dimension
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        """
        x shape: (batch_size, sequence_length, input_dim)
        hidden is an optional tuple of (h_0, c_0) for the LSTM.
        """
        if hidden is None:
            # Initialize hidden state if not provided
            batch_size = x.shape[0]
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, device=x.device
            )
            c_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, device=x.device
            )
            hidden = (h_0, c_0)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(
            x, hidden
        )  # shape of lstm_out => (B, T, hidden_dim)

        # We often only care about the last output in the sequence:
        last_output = lstm_out[:, -1, :]  # (B, hidden_dim)

        # Map last output to final predictions
        preds = self.fc(last_output)  # (B, output_dim)
        return preds, (h_n, c_n)


class MixedFootholdLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # For cross-entropy: log-softmax + NLL loss, or direct CrossEntropyLoss
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        pred: (B, 7) => [4 for footID logits, 3 for coords]
        target: (B, 7) => [4 for footID one-hot, 3 for coords]
        """
        # 1) Foot ID Loss (Cross-Entropy)
        foot_id_pred = pred[:, :4]  # shape (B, 4) => logit for each of 4 classes
        foot_id_true_one_hot = target[:, :4]  # shape (B, 4) => one-hot
        foot_id_true_class = foot_id_true_one_hot.argmax(dim=1)  # shape (B,)

        foot_id_loss = self.ce(foot_id_pred, foot_id_true_class)

        # 2) Coordinate Loss (MSE)
        coords_pred = pred[:, 4:]  # shape (B, 3)
        coords_true = target[:, 4:]  # shape (B, 3)
        coords_loss = self.mse(coords_pred, coords_true)

        # Combine
        total_loss = foot_id_loss + coords_loss
        return total_loss
