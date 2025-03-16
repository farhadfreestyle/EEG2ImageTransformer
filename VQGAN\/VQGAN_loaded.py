import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
import numpy as np

# First, let's make sure we have the required repositories
if not os.path.exists('taming-transformers'):
    !git clone https://github.com/CompVis/taming-transformers.git

# Add to Python path
sys.path.append('./taming-transformers')

# Download the VQGAN models
# Fixed URLs for proper download
print("Downloading VQGAN model files...")
!wget -c https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt -O vqgan_imagenet_f16_16384.ckpt

# Load the model
try:
    from taming.models.vqgan import VQModel

    # Load configuration
    config = OmegaConf.load('model.yaml')

    # Initialize model
    model = VQModel(**config.model.params)

    # Load checkpoint
    ckpt = torch.load('last.ckpt', map_location='cpu', weights_only=False)

    # Fix state dict if needed
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']

        # Remove 'model.' prefix if present
        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    print("VQGAN model loaded successfully!")

    # Move to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model moved to {device}")

    # Create a wrapper class for easier use
    class VQGANWrapper(nn.Module):
        def __init__(self, vqgan_model):
            super().__init__()
            self.vqgan = vqgan_model
            # Get the actual codebook size and dimension
            self.codebook_size = self.vqgan.quantize.n_e
            self.latent_dim = self.vqgan.quantize.e_dim

        def encode(self, x):
            """Encode images to latent space"""
            with torch.no_grad():
                quant_z, _, info = self.vqgan.encode(x)
                indices = info[2]
            return quant_z, indices

        def decode(self, z):
            """Decode from latent space"""
            with torch.no_grad():
                return self.vqgan.decode(z)

        def decode_indices(self, indices):
            """Decode from indices"""
            with torch.no_grad():
                z = self.vqgan.quantize.get_codebook_entry(indices, shape=None)
                return self.vqgan.decode(z)

    # Create the wrapper
    vqgan_wrapper = VQGANWrapper(model)
    print(f"VQGAN wrapper created with codebook size: {vqgan_wrapper.codebook_size}")

    # ------------------------------------------------------------------------
    # Define a simple Transformer model that works with the VQGAN
    # ------------------------------------------------------------------------

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(PositionalEncoding, self).__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)

            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class EEG2ImageTransformer(nn.Module):
        def __init__(self,
                     eeg_dim=128,
                     d_model=512,
                     nhead=8,
                     num_encoder_layers=6,
                     num_decoder_layers=6,
                     dim_feedforward=2048,
                     codebook_size=16384,
                     max_seq_len=256):
            super(EEG2ImageTransformer, self).__init__()

            # Embedding for EEG features
            self.eeg_embedding = nn.Linear(eeg_dim, d_model)

            # Token embedding for VQGAN codebook indices
            self.token_embedding = nn.Embedding(codebook_size, d_model)

            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                     dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

            # Transformer decoder
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                     dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

            # Output projection to codebook size
            self.output_projection = nn.Linear(d_model, codebook_size)

            # Save parameters
            self.d_model = d_model
            self.max_seq_len = max_seq_len

        def forward(self, eeg_features, target_indices=None, mask_ratio=0.0):
            """
            Forward pass through the transformer
            Args:
                eeg_features: [batch_size, eeg_dim]
                target_indices: [batch_size, seq_len] or None for inference
                mask_ratio: float, portion of tokens to mask for training
            """
            batch_size = eeg_features.shape[0]

            # Embed EEG features
            eeg_embed = self.eeg_embedding(eeg_features)  # [batch_size, d_model]
            eeg_embed = eeg_embed.unsqueeze(1)  # [batch_size, 1, d_model]

            # Apply positional encoding to EEG embedding
            eeg_embed = self.pos_encoder(eeg_embed)

            # Pass through encoder
            memory = self.transformer_encoder(eeg_embed)

            if target_indices is not None:
                # Training mode
                seq_len = target_indices.shape[1]

                # Create target embedding from indices
                target_embed = self.token_embedding(target_indices)  # [batch_size, seq_len, d_model]
                target_embed = self.pos_encoder(target_embed)

                # Create masking for training
                if mask_ratio > 0:
                    mask = torch.bernoulli(torch.ones_like(target_indices, dtype=torch.float) * mask_ratio).bool()
                    masked_indices = target_indices.clone()
                    masked_indices[mask] = 0  # Use 0 as mask token

                    # Embed masked sequence
                    target_embed_masked = self.token_embedding(masked_indices)
                    target_embed_masked = self.pos_encoder(target_embed_masked)
                else:
                    mask = None
                    target_embed_masked = target_embed

                # Generate attention mask to prevent looking ahead
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(eeg_features.device)

                # Decoder pass
                output = self.transformer_decoder(target_embed_masked, memory, tgt_mask=tgt_mask)

                # Project to vocabulary size
                logits = self.output_projection(output)  # [batch_size, seq_len, codebook_size]

                return logits, mask

            else:
                # Inference mode - generate tokens autoregressively
                # Start with a sequence of just a start token (we'll use index 0)
                current_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=eeg_features.device)

                # Generate tokens one by one
                for i in range(self.max_seq_len):
                    # Embed current sequence
                    tgt_embed = self.token_embedding(current_indices)
                    tgt_embed = self.pos_encoder(tgt_embed)

                    # Generate tgt mask to prevent looking ahead
                    tgt_len = current_indices.shape[1]
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(eeg_features.device)

                    # Decoder forward pass
                    output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)

                    # Get predictions for the next token only
                    next_token_logits = self.output_projection(output[:, -1])  # [batch_size, codebook_size]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # [batch_size, 1]

                    # Concatenate with current sequence
                    current_indices = torch.cat([current_indices, next_token], dim=1)

                    # Stop if all sequences have an end token (we could use a special token for this)
                    # For simplicity, we'll just generate a fixed-size sequence

                # Remove the start token
                generated_indices = current_indices[:, 1:]

                return generated_indices

    # Create an instance of the transformer model
    eeg_transformer = EEG2ImageTransformer(
        eeg_dim=128,  # Dimension of EEG features from your paper
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        codebook_size=vqgan_wrapper.codebook_size,
        max_seq_len=256  # Adjust based on your VQGAN's latent resolution
    ).to(device)

    print(f"EEG2Image Transformer created with {sum(p.numel() for p in eeg_transformer.parameters())} parameters")

    # ------------------------------------------------------------------------
    # Define a mock EEG dataset for testing
    # ------------------------------------------------------------------------

    class MockEEGDataset(Dataset):
        def __init__(self, num_samples=100, eeg_dim=128, img_size=256):
            self.num_samples = num_samples
            self.eeg_dim = eeg_dim
            self.img_size = img_size

            # Generate random EEG features
            self.eeg_features = torch.randn(num_samples, eeg_dim)

            # Generate random images (just for demonstration)
            self.images = torch.randn(num_samples, 3, img_size, img_size)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.eeg_features[idx], self.images[idx]

    # Create a function to test the full pipeline
    def test_eeg2image_pipeline():
        # Create a mock dataset
        mock_dataset = MockEEGDataset(num_samples=5)
        mock_dataloader = DataLoader(mock_dataset, batch_size=2, shuffle=True)

        # Get a sample
        eeg_batch, img_batch = next(iter(mock_dataloader))
        eeg_batch = eeg_batch.to(device)
        img_batch = img_batch.to(device)

        print(f"EEG batch shape: {eeg_batch.shape}")
        print(f"Image batch shape: {img_batch.shape}")

        # Normalize images to [-1, 1] as expected by VQGAN
        img_batch = (img_batch - 0.5) * 2.0

        # Encode images to get VQGAN indices
        z, indices = vqgan_wrapper.encode(img_batch)
        print(f"VQGAN encoded shape: {z.shape}")
        print(f"VQGAN indices shape: {indices.shape}")

        # Forward pass through transformer with the indices as target
        logits, _ = eeg_transformer(eeg_batch, indices)
        print(f"Transformer output shape: {logits.shape}")

        # Get predictions
        pred_indices = torch.argmax(logits, dim=-1)
        print(f"Predicted indices shape: {pred_indices.shape}")

        # Decode indices to images
        reconstructed_imgs = vqgan_wrapper.decode_indices(pred_indices)
        print(f"Reconstructed images shape: {reconstructed_imgs.shape}")

        # Test inference mode (generation)
        generated_indices = eeg_transformer(eeg_batch)
        print(f"Generated indices shape: {generated_indices.shape}")

        # Decode generated indices
        generated_imgs = vqgan_wrapper.decode_indices(generated_indices)
        print(f"Generated images shape: {generated_imgs.shape}")

        print("Test completed successfully!")

    # Run the test
    print("\nTesting EEG2Image pipeline...")
    test_eeg2image_pipeline()

    # ------------------------------------------------------------------------
    # Example of how to train the model
    # ------------------------------------------------------------------------

    def train_eeg2image_model(dataloader, num_epochs=10, lr=1e-4):
        """
        Example training function for the EEG2Image transformer

        Args:
            dataloader: DataLoader with (eeg_features, images) pairs
            num_epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(eeg_transformer.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0

            for eeg_batch, img_batch in dataloader:
                # Move to device
                eeg_batch = eeg_batch.to(device)
                img_batch = img_batch.to(device)

                # Normalize images to [-1, 1]
                img_batch = (img_batch - 0.5) * 2.0

                # Encode images to get target indices
                _, indices = vqgan_wrapper.encode(img_batch)

                # Forward pass with 15% masking
                logits, mask = eeg_transformer(eeg_batch, indices, mask_ratio=0.15)

                # Compute loss only on masked tokens
                if mask is not None:
                    # Reshape for computing loss
                    logits_flat = logits.reshape(-1, logits.size(-1))
                    indices_flat = indices.reshape(-1)
                    mask_flat = mask.reshape(-1)

                    # Compute loss only on masked tokens
                    loss = criterion(logits_flat[mask_flat], indices_flat[mask_flat])
                else:
                    # Compute loss on all tokens
                    loss = criterion(logits.reshape(-1, logits.size(-1)), indices.reshape(-1))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

    print("\nTraining function prepared. Ready to use with your EEG dataset!")

    # ------------------------------------------------------------------------
    # Function to generate images from EEG features
    # ------------------------------------------------------------------------

    def generate_image_from_eeg(eeg_features):
        """
        Generate an image from EEG features

        Args:
            eeg_features: Tensor of shape [batch_size, eeg_dim]

        Returns:
            Tensor of generated images
        """
        # Ensure EEG features are on the right device
        eeg_features = eeg_features.to(device)

        # Forward pass through transformer (inference mode)
        with torch.no_grad():
            indices = eeg_transformer(eeg_features)

            # Decode indices to images
            images = vqgan_wrapper.decode_indices(indices)

            # Convert from [-1, 1] to [0, 1] for visualization
            images = (images + 1) / 2

        return images

    print("\nImage generation function prepared!")
    print("You can now use the VQGAN-Transformer pipeline for your EEG-to-Image project.")

except Exception as e:
    print(f"Error loading VQGAN model: {e}")
    print("Please check if the model files were downloaded correctly.")
