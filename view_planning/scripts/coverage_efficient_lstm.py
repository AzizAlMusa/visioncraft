import torch
import torch.nn as nn
import torch.nn.functional as F

class CoverageEfficientLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(CoverageEfficientLSTM, self).__init__()
        
        # LSTM layer to process sequence of viewpoints
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layers to predict position and quaternion
        self.fc_position = nn.Linear(hidden_dim, 3)  # Outputs x, y, z position
        self.fc_orientation = nn.Linear(hidden_dim, 4)  # Outputs quaternion (q_w, q_x, q_y, q_z)
        
    def forward(self, object_voxel, coverage_mask, hidden=None):
        """
        Forward pass for the model.
        
        Parameters:
        - object_voxel: Tensor of shape (batch_size, seq_len, voxel_dim) representing the 3D object.
        - coverage_mask: Tensor of shape (batch_size, seq_len, voxel_dim) showing covered areas.
        
        Returns:
        - positions: Tensor of predicted positions (batch_size, seq_len, 3)
        - orientations: Tensor of predicted quaternions (batch_size, seq_len, 4)
        - hidden: Hidden state of the LSTM for continuity between sequences
        """
        # Combine object and coverage mask as input
        x = torch.cat((object_voxel, coverage_mask), dim=-1)  # Concatenate along feature dimension

        # Pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Predict position and orientation
        positions = torch.tanh(self.fc_position(lstm_out))  # Position constrained to [-1, 1]
        orientations = self.fc_orientation(lstm_out)
        orientations = F.normalize(orientations, dim=-1)  # Normalize quaternion to unit length
        
        return positions, orientations, hidden


# Custom loss function for training
def viewpoint_loss(predicted_positions, predicted_orientations, true_coverage_gain, length_penalty=0.1, redundancy_penalty=0.05):
    """
    Custom loss function for viewpoint prediction.
    
    Parameters:
    - predicted_positions: Predicted positions from the model
    - predicted_orientations: Predicted orientations from the model
    - true_coverage_gain: Coverage gain obtained after applying each viewpoint
    - length_penalty: Weight for sequence length penalty
    - redundancy_penalty: Penalty for viewpoints with low coverage gain
    
    Returns:
    - loss: Combined loss value
    """
    # Coverage gain loss (maximize gain)
    coverage_loss = -torch.sum(true_coverage_gain)
    
    # Sequence length penalty (penalizes number of viewpoints)
    length_loss = length_penalty * predicted_positions.shape[1]
    
    # Redundant viewpoint penalty (penalizes low coverage gain)
    redundancy_loss = redundancy_penalty * torch.sum((true_coverage_gain < 0.1).float())
    
    # Total loss
    loss = coverage_loss + length_loss + redundancy_loss
    return loss


# Example usage: Instantiating the model and defining optimizer
if __name__ == "__main__":
    # Define input and hidden dimensions for example
    input_dim = 32768 * 2  # Example voxel size * 2 for combined object and mask
    hidden_dim = 128
    
    # Instantiate model
    model = CoverageEfficientLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy input for testing the forward pass
    batch_size = 2
    seq_len = 5
    voxel_dim = 32768
    object_voxel = torch.randn(batch_size, seq_len, voxel_dim)
    coverage_mask = torch.zeros(batch_size, seq_len, voxel_dim)
    
    # Forward pass
    positions, orientations, hidden = model(object_voxel, coverage_mask)
    
    # Print shapes of the outputs
    print("Positions shape:", positions.shape)        # Expected: (batch_size, seq_len, 3)
    print("Orientations shape:", orientations.shape)  # Expected: (batch_size, seq_len, 4)
