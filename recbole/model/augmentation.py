import torch
import torch.nn as nn


class CropAugmentation(nn.Module):
    """Randomly crop consecutive items in sequences.
    
    This augmentation randomly selects a continuous subsequence from the original sequence,
    maintaining the temporal order of items within the cropped segment.
    """
    
    def __init__(self, crop_ratio, min_length=3):
        """Initialize crop augmentation.
        
        Args:
            crop_ratio (float): The ratio of items to keep in the sequence (0.0 to 1.0)
            min_length (int): Minimum length of the cropped sequence
        """
        super().__init__()
        self.crop_ratio = crop_ratio
        self.min_length = min_length
    
    def forward(self, item_seq, item_seq_len):
        """Apply crop augmentation to sequences.
        
        Args:
            item_seq (torch.Tensor): Input sequences with shape (batch_size, max_seq_length)
            item_seq_len (torch.Tensor): Actual sequence lengths with shape (batch_size,)
            
        Returns:
            torch.Tensor: Cropped sequences with shape (batch_size, max_seq_length)
            torch.Tensor: Updated sequence lengths with shape (batch_size,)
            
        Example:
            Input:  [1, 3, 5, 4, 8, 6, 0, 0] with length 6
            Output: [1, 3, 5, 0, 0, 0, 0, 0] with length 3 (crop_ratio=0.5)
        """
        if not self.training:
            return item_seq, item_seq_len
        
        batch_size, max_seq_length = item_seq.shape
        device = item_seq.device
        
        # Create output tensors
        augmented_seq = item_seq.clone()
        augmented_len = item_seq_len.clone()
        
        for i in range(batch_size):
            seq_len = item_seq_len[i].item()
            
            # Skip if sequence is too short
            if seq_len <= self.min_length:
                continue
            
            # Calculate crop length
            crop_length = max(self.min_length, int(seq_len * self.crop_ratio))
            crop_length = min(crop_length, seq_len)
            
            # Randomly select start position
            max_start = seq_len - crop_length + 1
            start_pos = torch.randint(0, max_start, (1,), device=device).item()
            end_pos = start_pos + crop_length
            
            # Apply cropping
            cropped_items = item_seq[i, start_pos:end_pos]
            augmented_seq[i, :crop_length] = cropped_items
            augmented_seq[i, crop_length:] = 0  # Pad with zeros
            augmented_len[i] = crop_length
        
        return augmented_seq, augmented_len


class MaskAugmentation(nn.Module):
    """Randomly mask items in sequences to a special token.
    
    This augmentation randomly replaces items with a mask token, which helps
    the model learn to predict masked items from context.
    """
    
    def __init__(self, mask_ratio, mask_token=0):
        """Initialize mask augmentation.
        
        Args:
            mask_ratio (float): The ratio of items to mask (0.0 to 1.0)
            mask_token (int): The token ID used for masking
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
    
    def forward(self, item_seq, item_seq_len):
        """Apply mask augmentation to sequences.
        
        Args:
            item_seq (torch.Tensor): Input sequences with shape (batch_size, max_seq_length)
            item_seq_len (torch.Tensor): Actual sequence lengths with shape (batch_size,)
            
        Returns:
            torch.Tensor: Masked sequences with shape (batch_size, max_seq_length)
            torch.Tensor: Sequence lengths (unchanged) with shape (batch_size,)
            
        Example:
            Input:  [1, 3, 5, 4, 8, 6, 0, 0] with length 6
            Output: [1, 3, 0, 4, 0, 6, 0, 0] with length 6 (mask_ratio=0.33)
        """
        if not self.training:
            return item_seq, item_seq_len
        
        batch_size, max_seq_length = item_seq.shape
        device = item_seq.device
        
        # Create output tensor
        augmented_seq = item_seq.clone()
        
        for i in range(batch_size):
            seq_len = item_seq_len[i].item()
            
            # Skip if sequence is too short
            if seq_len <= 1:
                continue
            
            # Create mask for valid positions (non-zero items)
            valid_mask = (item_seq[i, :seq_len] != 0)
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                continue
            
            # Calculate number of items to mask
            num_to_mask = max(1, int(len(valid_indices) * self.mask_ratio))
            num_to_mask = min(num_to_mask, len(valid_indices))
            
            # Randomly select positions to mask
            mask_indices = torch.randperm(len(valid_indices), device=device)[:num_to_mask]
            positions_to_mask = valid_indices[mask_indices]
            
            # Apply masking
            augmented_seq[i, positions_to_mask] = self.mask_token
        
        return augmented_seq, item_seq_len


class ReorderAugmentation(nn.Module):
    """Randomly reorder consecutive subsequences in sequences.
    
    This augmentation randomly selects a continuous subsequence and shuffles
    the order of items within that subsequence.
    """
    
    def __init__(self, reorder_ratio, min_window_size=2, max_window_size=5):
        """Initialize reorder augmentation.
        
        Args:
            reorder_ratio (float): The ratio of sequences to apply reordering
            min_window_size (int): Minimum size of the reorder window
            max_window_size (int): Maximum size of the reorder window
        """
        super().__init__()
        self.reorder_ratio = reorder_ratio
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
    
    def forward(self, item_seq, item_seq_len):
        """Apply reorder augmentation to sequences.
        
        Args:
            item_seq (torch.Tensor): Input sequences with shape (batch_size, max_seq_length)
            item_seq_len (torch.Tensor): Actual sequence lengths with shape (batch_size,)
            
        Returns:
            torch.Tensor: Reordered sequences with shape (batch_size, max_seq_length)
            torch.Tensor: Sequence lengths (unchanged) with shape (batch_size,)
            
        Example:
            Input:  [1, 3, 5, 4, 8, 6, 0, 0] with length 6
            Output: [1, 3, 8, 4, 5, 6, 0, 0] with length 6 (reorder window [5,4,8])
        """
        if not self.training:
            return item_seq, item_seq_len
        
        batch_size, max_seq_length = item_seq.shape
        device = item_seq.device
        
        # Create output tensor
        augmented_seq = item_seq.clone()
        
        for i in range(batch_size):
            seq_len = item_seq_len[i].item()
            
            # Skip if sequence is too short
            if seq_len <= self.min_window_size:
                continue
            
            # Randomly decide whether to apply reordering
            if torch.rand(1, device=device).item() > self.reorder_ratio:
                continue
            
            # Find valid positions (non-zero items)
            valid_mask = (item_seq[i, :seq_len] != 0)
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) < self.min_window_size:
                continue
            
            # Determine window size
            max_possible_window = min(len(valid_indices), self.max_window_size)
            window_size = torch.randint(self.min_window_size, int(max_possible_window) + 1, (1,), device=device).item()
            
            # Randomly select window start position
            max_start = len(valid_indices) - window_size + 1
            start_idx = torch.randint(0, int(max_start), (1,), device=device).item()
            end_idx = start_idx + window_size
            
            # Get window indices in the original sequence
            window_indices = valid_indices[start_idx:end_idx]
            
            # Get items in the window
            window_items = item_seq[i, window_indices]
            
            # Shuffle the window items
            shuffled_indices = torch.randperm(int(window_size), device=device)
            shuffled_items = window_items[shuffled_indices]
            
            # Apply reordering
            augmented_seq[i, window_indices] = shuffled_items
        
        return augmented_seq, item_seq_len


class AugmentationPipeline(nn.Module):
    """Pipeline for applying multiple data augmentation strategies.
    
    This module combines multiple augmentation strategies and applies them
    to generate multiple augmented views of the input sequences.
    """
    
    def __init__(self, augmentation_configs):
        """Initialize augmentation pipeline.
        
        Args:
            augmentation_configs (list): List of dictionaries containing augmentation configurations.
                Each dict should have 'type' and corresponding parameters.
                Example: [
                    {'type': 'mask', 'mask_ratio': 0.2},
                    {'type': 'crop', 'crop_ratio': 0.8},
                    {'type': 'reorder', 'reorder_ratio': 0.3}
                ]
        """
        super().__init__()
        self.augmentations = nn.ModuleList()
        self._build_augmentations(augmentation_configs)
    
    def _build_augmentations(self, augmentation_configs):
        """Build augmentation modules from configurations."""
        for config in augmentation_configs:
            aug_type = config['type']
            
            if aug_type == 'mask':
                mask_ratio = config.get('mask_ratio', 0.2)
                mask_token = config.get('mask_token', 0)
                self.augmentations.append(MaskAugmentation(mask_ratio, mask_token))
            
            elif aug_type == 'crop':
                crop_ratio = config.get('crop_ratio', 0.8)
                min_length = config.get('min_length', 3)
                self.augmentations.append(CropAugmentation(crop_ratio, min_length))
            
            elif aug_type == 'reorder':
                reorder_ratio = config.get('reorder_ratio', 0.3)
                min_window_size = config.get('min_window_size', 2)
                max_window_size = config.get('max_window_size', 5)
                self.augmentations.append(ReorderAugmentation(reorder_ratio, min_window_size, max_window_size))
            
            else:
                raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    def forward(self, item_seq, item_seq_len):
        """Apply all augmentation strategies to generate multiple views.
        
        Args:
            item_seq (torch.Tensor): Input sequences with shape (batch_size, max_seq_length)
            item_seq_len (torch.Tensor): Actual sequence lengths with shape (batch_size,)
            
        Returns:
            list: List of (augmented_seq, augmented_len) tuples for each augmentation strategy
        """
        if not self.training:
            return [(item_seq, item_seq_len)]
        
        augmented_views = []
        for augmentation in self.augmentations:
            aug_seq, aug_len = augmentation(item_seq, item_seq_len)
            augmented_views.append((aug_seq, aug_len))
        
        return augmented_views
    