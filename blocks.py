import torch
import torch.nn as nn
import numpy as np

class SpatialNorm(nn.Module):
    def __init__(self, feature_channels, control_channels, norm_layer=nn.GroupNorm,  num_groups=32, eps=1e-6, affine=True):
        super().__init__()
        self.norm_layer = norm_layer(num_channels=feature_channels, num_groups= num_groups, eps=eps, affine=affine)        
        self.conv_y = nn.Conv2d(control_channels, feature_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(control_channels, feature_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, feature, control):
        """
        feature, shape (B, C, H, W)
        control, shape (B,C',H', W')        
        """
        f_size = feature.shape[-2:] 
        control = torch.nn.functional.interpolate(control, size=f_size, mode="nearest")        
        feature = self.norm_layer(feature)
        # incorporate information from zq into the feature map feature in a normalized way. feature * scale + bias
        feature = feature * self.conv_y(control) + self.conv_b(control)
        return feature


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, control_channels = None,
                 dropout = 0.0, norm_layer=nn.GroupNorm):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        if norm_layer is nn.GroupNorm: # input: (N, C, H, W) -> (N, G, C//G, H, W), Mean, variance is calculated across (C//G, H, W) dimensions for each group
            self.spatialnorm = False
            self.norm1 = norm_layer(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        elif norm_layer is SpatialNorm:
            self.spatialnorm = True
            self.norm1 = norm_layer(in_channels, control_channels, norm_layer=nn.GroupNorm, num_groups=32, eps=1e-6, affine=True)       
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
       
        if self.spatialnorm:
            self.norm2 = norm_layer(out_channels, control_channels, norm_layer=nn.GroupNorm, num_groups=32, eps=1e-6, affine=True)
        else:
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, padding=0)

    def forward(self, feature, control = None):
        x = feature
        if self.spatialnorm:
            x = self.norm1(x, control)
        else:
            x = self.norm1(x)
        x = nn.functional.silu(x)
        x = self.conv1(x)
        if self.spatialnorm:
            x = self.norm2(x, control)
        else:
            x = self.norm2(x)
        x = nn.functional.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            feature = self.nin_shortcut(feature)
        return feature+x
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels, 
                 control_channels = 4, # vocabular embedded dim, the dim at bottle neck 
                 norm_layer: nn.Module = nn.GroupNorm):
        super().__init__()
        self.in_channels = in_channels
        if norm_layer is nn.GroupNorm: # input: (N, C, H, W) -> (N, G, C//G, H, W), Mean, variance is calculated across (C//G, H, W) dimensions for each group
            self.spatialnorm = False
            self.norm = norm_layer(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        elif norm_layer is SpatialNorm:
            self.spatialnorm = True
            self.norm = norm_layer(in_channels, control_channels, norm_layer=nn.GroupNorm, num_groups=32, eps=1e-6, affine=True)
        
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, feature, control = None):
        x = feature
        if self.spatialnorm:
            x = self.norm(x, control)
        else:
            x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)       
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w).permute(0,2,1)   # b,hw,c        
        k = k.reshape(b,c,h*w) # b,c,hw        
        x = torch.bmm(q*(int(c)**(-0.5)),k)
        x = torch.nn.functional.softmax(x, dim=2).permute(0,2,1) # b,hw,hw 
        x = torch.bmm(v.reshape(b,c,h*w),x)  # b, c,hw 
        x = x.reshape(b,c,h,w)       

        x = self.proj_out(x)

        return feature+x
    

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    in_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=0)
    def forward(self, x):
        # no asymmetric padding in torch conv, to give H/2, W/2 output
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x) # H/2, W/2       
        return x


class Encoder(nn.Module):
    def __init__(self,             
            u_channels = 4,
            image_resolution = 256,
            image_in_channels = 3,            
            image_embed_channels = 128, 
            channel_mult = [1,2,2,4],
            num_res_blocks = 2, 
            attn_at_layer_resolutions = (32,),             
            double_u_channels = False,
            dropout = 0.0):        
        """
        Args:            
            u_channels: the channel for the mid blocks of u-net..
            image_resolution: input image resolution
            image_in_channels: image input channels            
            image_embed_channels: image embedding channel, the number of channel after the first conv embedding
            channel_mult: the factor to multiply the ch to get the channel in each layer
            num_res_blocks: number of ResnetBlock in each layer in encoder. The decoder has one more block, that is num_res_blocks+1
            attn_at_layer_resolutions: when layer size H,W is equale to attn_resolutions, single head attention is performed on H*W dimention over channel C 
            double_u_channels: if make u_channles * 2.
            dropout: dropout ratio.
        """

        super().__init__()

        self.num_layers = len(channel_mult)
        self.num_res_blocks = num_res_blocks        
        
        # first conv to image
        self.conv_in = torch.nn.Conv2d(image_in_channels,
                                       image_embed_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = image_resolution # tracking out dimention H, W after each layer
        in_ch_mult = (1,)+tuple(channel_mult) # used as multiplication factor for get in_channels of each layer
        self.down = nn.ModuleList()
        for i_level in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = image_embed_channels*in_ch_mult[i_level]
            block_out = image_embed_channels*channel_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         norm_layer=nn.GroupNorm,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_at_layer_resolutions: # single head attention will be added when H, W == resolution = 32 here
                    attn.append(AttnBlock(block_in, norm_layer=nn.GroupNorm))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_layers-1:
                down.downsample = Downsample(block_in) # conv2D down sample to W/2, H/2
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       norm_layer=nn.GroupNorm,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in, norm_layer=nn.GroupNorm)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       norm_layer=nn.GroupNorm,
                                       dropout=dropout)
      
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*u_channels if double_u_channels else u_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x): 
        # downsampling
        x = self.conv_in(x)
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
                if len(self.down[i_level].attn) > 0:
                    x = self.down[i_level].attn[i_block](x)                
            if i_level != self.num_layers-1:
                x = self.down[i_level].downsample(x)
        # middle       
        x = self.mid.block_1(x) # mid res block
        x = self.mid.attn_1(x) # nud attention block
        x = self.mid.block_2(x) # mid res block
        # end
        x = self.norm_out(x) # GroupNorm
        x = nn.functional.silu(x)
        x = self.conv_out(x) 
        return x
    

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()        
        self.conv = nn.Conv2d(in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest") # H*2, W*2
        x = self.conv(x)
        return x


class MOVQDecoder(nn.Module):
    def __init__(self,
                 dim_vocab_embed=4, 
                 u_channels = 4,
                 image_resolution = 256,                 
                 image_embed_channels =128, 
                 end_decode_channels=3, 
                 channel_mult=(1,2,2,4), 
                 num_res_blocks = 2,
                 attn_at_layer_resolutions= (32,), 
                 dropout=0.0,
                ):
        
        """
        Args:            
            dim_vocab_embed: vocabular embedded dim, the dim at bottle neck
            u_channels: the channel for the mid blocks of u-net..
            image_resolution: input image resolution.
            image_embed_channels: image embedding channel, the number of channel after the first conv embedding.
            end_decode_channels: output channel of decode, 3 is the RGB channels.            
            channel_mult: the factor to multiply the ch to get the channel in each layer.
            num_res_blocks: number of ResnetBlock in each layer in encoder. The decoder has one more block, that is num_res_blocks+1.
            attn_at_layer_resolutions: when layer size H,W is equale to attn_resolutions, single head attention is performed on H*W dimention over channel C.
        """         
        super().__init__()
        self.image_embed_channels = image_embed_channels        
        self.num_layers = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.image_resolution = image_resolution        
        block_in = image_embed_channels*channel_mult[self.num_layers-1]
        curr_res = image_resolution // 2**(self.num_layers-1)
        self.z_shape = (1,u_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(u_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,                                       
                                       dropout=dropout,
                                       control_channels=dim_vocab_embed,
                                       norm_layer=SpatialNorm)
        self.mid.attn_1 = AttnBlock(block_in, dim_vocab_embed, norm_layer=SpatialNorm)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,                                       
                                       dropout=dropout,
                                       control_channels=dim_vocab_embed,
                                       norm_layer=SpatialNorm)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_layers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = image_embed_channels*channel_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,                                         
                                         dropout=dropout,
                                         control_channels=dim_vocab_embed,
                                         norm_layer=SpatialNorm))
                block_in = block_out
                if curr_res in attn_at_layer_resolutions:
                    attn.append(AttnBlock(block_in, dim_vocab_embed, norm_layer=SpatialNorm)) # H, W, 32, 32
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end       
        self.norm_out = SpatialNorm(block_in,dim_vocab_embed, norm_layer=nn.GroupNorm, num_groups=32, eps=1e-6, affine=True)        
        self.conv_out = torch.nn.Conv2d(block_in,
                                        end_decode_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x:torch.Tensor, control:torch.Tensor):
        """
        Args:
            z (tensor): B,C H, W
            control (tensor) 
        
        """
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = x.shape        
        # z to block_in
        x = self.conv_in(x)

        # middle
        x = self.mid.block_1(x, control)
        x = self.mid.attn_1(x, control)
        x = self.mid.block_2(x, control)

        # upsampling
        for i_level in reversed(range(self.num_layers)):
            for i_block in range(self.num_res_blocks+1):
                x = self.up[i_level].block[i_block](x, control)
                if len(self.up[i_level].attn) > 0:
                    x = self.up[i_level].attn[i_block](x, control)
            if i_level != 0:
                x = self.up[i_level].upsample(x)
                        
        x = self.norm_out(x, control)
        x = nn.functional.silu(x)
        x = self.conv_out(x)
        return x



class VectorQuantizer(nn.Module):
   
    def __init__(self, num_vocab_embed=16384, dim_vocab_embed=4, embed_loss_coeff=0.25):
        super().__init__()
        """
        Args:
            num_vocab_embed: vocabular size
            dim_vocab_embed: number of channles for embed the vocab.
            embed_loss_coeff: the coefficient of contribution of embedding loss to vocabular embedding loss.        
        """
        self.num_vocab_embed = num_vocab_embed
        self.end_encode_channels = dim_vocab_embed
        self.embed_loss_coeff  = embed_loss_coeff   
        self.embedding = nn.Embedding(self.num_vocab_embed, self.end_encode_channels)
        self.embedding.weight.data.uniform_(-1.0 / self.num_vocab_embed, 1.0 / self.num_vocab_embed)  

    def forward(self, x:torch.Tensor):
       
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        x = x.permute(0,2,3,1).contiguous() # b c h w -> b h w c
        z_flattened = x.view(-1, self.end_encode_channels) # b*h*w, c
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # calculate outer difference between (z-e)**2, the following code is not necssary fast when use torch.compile,
        # should balance copy and calculation
        # embedding.weight shape n_e, c
        encoding_index = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.transpose(-1,-2)) # d shape(b*h*w, n_e)
            # torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))   
        
        encoding_index = torch.argmin(encoding_index, dim=1)
        embed = self.embedding(encoding_index).view(x.shape)       

        # compute loss for embedding
        loss = torch.mean((embed.detach()-x)**2) + self.embed_loss_coeff  * \
                   torch.mean((embed - x.detach()) ** 2)

        # preserve gradients
        embed = x + (embed - x).detach()

        # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        embed =  embed.permute(0, 3, 1, 2)       

        return embed, loss, encoding_index

   
    




