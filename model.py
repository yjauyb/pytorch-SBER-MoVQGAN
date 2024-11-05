import torch
from torch import nn
from blocks import Encoder, MOVQDecoder, VectorQuantizer

class MOVQ(nn.Module):
    # use 67M config
    def __init__(self,            
        num_vocab_embed = 16384,
        dim_vocab_embed = 4, 
        u_channels = 4,
        image_resolution = 256, 
        image_in_channels = 3,
        end_decode_channels = 3, 
        image_embed_channels = 128,
        channel_mult = [1,2,2,4],
        num_res_blocks = 2, 
        attn_at_layer_resolutions = (32,)      
        ):
        """
        Args:
            num_vocab_embed: vocabular size
            dim_vocab_embed: vocabular embedded dim, the dim at bottle neck
            u_channels: the channel for the mid blocks of u-net.
            image_resolution: input image resolution
            image_in_channels: image input channels
            end_decode_channels: output channel of decode, 3 is the RGB channels
            image_embed_channels: image embedding channel, the number of channel after the first conv embedding
            channel_mult: the factor to multiply the ch to get the channel in each layer
            num_res_blocks: number of ResnetBlock in each layer in encoder. The decoder has one more block, that is num_res_blocks+1
            attn_at_layer_resolutions: when layer size H,W is equale to attn_resolutions, single head attention is performed on H*W dimention over channel C 
            
        """
        super().__init__()

        self.encoder = Encoder(u_channels=u_channels, 
                               image_resolution=image_resolution, 
                               image_in_channels=image_in_channels, 
                            image_embed_channels=image_embed_channels, 
                            channel_mult=channel_mult, 
                            num_res_blocks=num_res_blocks, 
                            attn_at_layer_resolutions=attn_at_layer_resolutions)
        self.decoder = MOVQDecoder(dim_vocab_embed=dim_vocab_embed,
                                u_channels=u_channels, 
                                image_resolution=image_resolution,
                                image_embed_channels=image_embed_channels,
                                end_decode_channels=end_decode_channels, 
                                channel_mult=channel_mult, 
                                num_res_blocks=num_res_blocks,
                                attn_at_layer_resolutions=attn_at_layer_resolutions)        
        
        self.quantize = VectorQuantizer(num_vocab_embed=num_vocab_embed, 
                                        dim_vocab_embed=dim_vocab_embed)
        self.quant_conv = torch.nn.Conv2d(u_channels, dim_vocab_embed, 1)
        self.post_quant_conv = torch.nn.Conv2d(dim_vocab_embed, u_channels, 1) 

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_conv(x)
        x, emb_loss, info = self.quantize(x)
        return x, emb_loss, info 
    

    def decode(self, emb):
        feature = self.post_quant_conv(emb)
        feature = self.decoder(feature, emb)
        return feature
    def forward(self, x):
        x, emb_loss, _ = self.encode(x)
        x = self.decode(x)
        return x, emb_loss 