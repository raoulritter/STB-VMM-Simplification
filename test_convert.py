import torch

from models.model import STBVMM


model = STBVMM(img_size=384, patch_size=1, in_chans=3,
                embed_dim=192, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                use_checkpoint=False, img_range=1., resi_connection='1conv',
                manipulator_num_resblk=1).to("cpu")



checkpoint = torch.load('ckpt/ckpt_e10.pth.tar')
# print(checkpoint.keys())

print(checkpoint['state_dict'])

model.load_state_dict(checkpoint['state_dict'], strict= False)
# Get the keys in the checkpoint's state_dict
checkpoint_keys = set(checkpoint['state_dict'].keys())

# Get the keys in the current model's state_dict
model_keys = set(model.state_dict().keys())

# Find the difference between the keys
keys_only_in_checkpoint = checkpoint_keys - model_keys
keys_only_in_model = model_keys - checkpoint_keys

# Print the results
print("Keys only in the checkpoint's state_dict:")
print(keys_only_in_checkpoint)


