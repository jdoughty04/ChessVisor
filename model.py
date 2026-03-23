import torch
import torch.nn as nn
import torchvision.models as models

class AddCoords(nn.Module):
    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape (batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        # Create meshgrid
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        # Normalize to [-1, 1]
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        # Reshape to (Batch, 1, X, Y)
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        # Concatenate
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        return ret

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.1):
        super(FeedForward, self).__init__()
        if hidden_dim is None:
            hidden_dim = out_dim * 4
        
        self.w1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.w2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        self.drop2 = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Identity()
        if in_dim != out_dim:
            self.residual = nn.Conv2d(in_dim, out_dim, kernel_size=1)
            
    def forward(self, x):
        res = self.residual(x)
        x = self.w1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.w2(x)
        x = self.drop2(x)
        return x + res

class MultiHeadSpatialAttention(nn.Module):
    """Multi-head attention for spatial features with multiple layers."""
    def __init__(self, in_channels, num_heads=4, num_layers=4, dropout=0.1):
        super(MultiHeadSpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'q_proj': nn.Linear(in_channels, in_channels),
                'k_proj': nn.Linear(in_channels, in_channels),
                'v_proj': nn.Linear(in_channels, in_channels),
                'out_proj': nn.Linear(in_channels, in_channels),
                'norm1': nn.LayerNorm(in_channels),
                'norm2': nn.LayerNorm(in_channels),
                'ffn': nn.Sequential(
                    nn.Linear(in_channels, in_channels * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_channels * 4, in_channels),
                    nn.Dropout(dropout)
                ),
                'dropout': nn.Dropout(dropout)
            })
            self.layers.append(layer)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) spatial features
        Returns:
            (B, H*W, H*W) attention logits for arrow start->end
        """
        B, C, H, W = x.shape
        N = H * W  # number of spatial positions (64 for 8x8)
        
        # Reshape to sequence: (B, N, C)
        x = x.flatten(2).permute(0, 2, 1)
        
        # Store attention weights from last layer for arrow prediction
        attn_weights = None
        
        for layer in self.layers:
            # Multi-head self-attention with residual
            residual = x
            x = layer['norm1'](x)
            
            # Project Q, K, V
            q = layer['q_proj'](x)  # (B, N, C)
            k = layer['k_proj'](x)
            v = layer['v_proj'](x)
            
            # Reshape for multi-head: (B, num_heads, N, head_dim)
            q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
            attn_weights = attn.mean(dim=1)  # Average across heads for arrow output (B, N, N)
            attn = torch.softmax(attn, dim=-1)
            attn = layer['dropout'](attn)
            
            # Apply attention to values
            out = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)
            out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
            out = layer['out_proj'](out)
            out = layer['dropout'](out)
            x = residual + out
            
            # Feed-forward with residual
            residual = x
            x = layer['norm2'](x)
            x = residual + layer['ffn'](x)
        
        return attn_weights  # (B, N, N) = (B, 64, 64) for 8x8 board


class ChessNet(nn.Module):
    def __init__(self, num_classes=13, pretrained=True):
        """
        num_classes: 13 (Empty + 6 White + 6 Black)
        pretrained: Whether to use ImageNet weights for backbone
        """
        super(ChessNet, self).__init__()
        
        # Load ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # We need the layers before global average pooling
        # ResNet18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        # Backbone Split to access Layer 2 (High Res)
        # ResNet18:
        # layer1: 64 channels (64x64)
        # layer2: 128 channels (32x32) -> Use this for edges
        # layer3: 256 channels (16x16)
        # layer4: 512 channels (8x8)
        
        self.stm = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2
        )
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Map 512 channels (from ResNet18 layer4) to 256 channels
        # Size: 8x8
        self.bottleneck = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        
        # Downsample Layer 2 (128ch, 32x32) -> (128ch, 8x8)
        # We use AdaptiveAvgPool or Conv with stride.
        self.downsample_l2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 32->16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16->8
            nn.ReLU()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Convolutional Head for Pieces
        # Input: 256 channels (8x8) -> Output: num_classes channels (8x8)
        self.conv_pieces = nn.Conv2d(256, num_classes, kernel_size=1, padding=0)
        
        # Convolutional Head for Highlights
        # Input: 256 channels (8x8) -> Output: 3 channels (8x8)
        self.conv_highlights = nn.Conv2d(256, 3, kernel_size=1, padding=0)
        
        # Head for Arrows (Multi-Head Attention)
        # Input: High Level (256) + Low Level (128) + Coords (2) = 386 channels
        self.add_coords = AddCoords()
        
        # FFNs for features before fusion
        self.ffn_high = FeedForward(256, 64)
        self.ffn_low = FeedForward(128, 64)
        
        # FFN for fusion (64 + 64 + 2 = 130 -> 64)
        self.ffn_fusion = FeedForward(130, 64)
        
        # Multi-head attention with 4 heads and 4 layers
        # Input is now 64 channels from ffn_fusion
        self.arrow_attention = MultiHeadSpatialAttention(
            in_channels=64, num_heads=4, num_layers=4, dropout=0.1
        )
        
        # Head for Perspective
        # Output: 1 (logit for is_flipped)
        self.fc_perspective = nn.Linear(256 * 8 * 8, 1)


        
        self.num_classes = num_classes
        
    def forward(self, x):
        # x: (B, 3, 256, 256)
        
        # Low Level Features (Layer 2)
        l2 = self.stm(x) # (B, 128, 32, 32)
        
        # High Level Features (Layer 4)
        x = self.layer3(l2)
        x = self.layer4(x) # (B, 512, 8, 8)
        
        # Bottleneck
        feat_high = self.bottleneck(x) # (B, 256, 8, 8)
        
        # Process Low Level for fusion
        feat_low = self.downsample_l2(l2) # (B, 128, 8, 8)
        
        # Arrows (Multi-Head Attention)
        
        # Apply dropout to features before heads (for pieces/highlights/perspective)
        feat_high_drop = self.dropout(feat_high)
        
        # Pieces
        p = self.conv_pieces(feat_high_drop) # (B, num_classes, 8, 8)
        p = p.flatten(2).permute(0, 2, 1) # (B, 64, num_classes)
        
        # Highlights
        h = self.conv_highlights(feat_high_drop) # (B, 3, 8, 8)
        h = h.flatten(2).permute(0, 2, 1) # (B, 64, 3)
        
        # Flatten for remaining FC heads (Perspective)
        x_flat = feat_high_drop.flatten(1)
        
        # Apply FFNs to features
        feat_high_ffn = self.ffn_high(feat_high) # (B, 64, 8, 8)
        feat_low_ffn = self.ffn_low(feat_low)    # (B, 64, 8, 8)
        
        # Concatenate for Arrows (Spatial)
        # (B, 128, 8, 8)
        feat_arrow = torch.cat([feat_high_ffn, feat_low_ffn], dim=1)
        
        # Add Coordinate Channels
        # (B, 130, 8, 8)
        feat_arrow = self.add_coords(feat_arrow)
        
        # Fuse with FFN
        # (B, 64, 8, 8)
        feat_arrow = self.ffn_fusion(feat_arrow)
        
        # Apply multi-head attention with multiple layers
        # Returns attention logits: (B, 64, 64)
        a = self.arrow_attention(feat_arrow)
        
        # Perspective
        flip = self.fc_perspective(x_flat) # (B, 1)

        return p, h, a, flip
