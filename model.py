import torch,timm,pdb
import torch.nn as nn
import torch.nn.functional as F

from aspp import build_aspp
from srm import setup_srm_layer

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        # Reduce channels from 512 to 256
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # Reduce channels from 256 to 128
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # Reduce channels from 128 to 2
        self.conv3 = nn.Conv2d(128, 2, kernel_size=3, padding=1)
        # Upsample to the desired spatial dimensions
        self.upsample = nn.Upsample(size=(1000, 1000), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.upsample(x)
        return x
    
class HRFNet(nn.Module):
    def __init__(self, inplanes= 1024):
        super(HRFNet, self).__init__()

        self.rgb_org_encoder = timm.create_model('resnet50', pretrained= True, features_only=True, out_indices=[4])
        self.rgb_downsampled_encoder = timm.create_model('mobilenetv2_100', pretrained= True, features_only=True, out_indices=[2])

        self.srm_org_encoder = timm.create_model('resnet50', pretrained= True, features_only=True, out_indices=[4])
        self.srm_downsampled_encoder = timm.create_model('mobilenetv2_100', pretrained= True, features_only=True, out_indices=[2])

        self.conv_srm = setup_srm_layer()
        self.aspp = build_aspp(inplanes = inplanes*2, outplanes = 512 )

        self.fuse1_conv = nn.Conv2d(in_channels=2080, out_channels=1024, kernel_size=1)
        self.fuse2_conv = nn.Conv2d(in_channels=2080, out_channels=1024, kernel_size=1)

        self.docoder = decoder()


    def forward(self, x):

        srmed = self.conv_srm(x)
        rgb_downsampled =F.interpolate(x, size=(224,224), mode='nearest')
        srmed_downsampled = F.interpolate(srmed, size=(224,224), mode='nearest')

        rgb_encoded = self.rgb_org_encoder(x)[0] #torch.Size([1, 2048, 32, 32])
        rgb_downsampled_encoded = self.rgb_downsampled_encoder(rgb_downsampled)[0] #torch.Size([1, 32, 28, 28])
        rgb_encoded = F.interpolate(rgb_encoded, size=rgb_downsampled_encoded.size(2), mode='nearest') #torch.Size([1, 2048, 28, 28])

        fused1 = torch.cat((rgb_encoded,rgb_downsampled_encoded), 1) #torch.Size([1, 2080, 28, 28])
        fused1_conv = self.fuse1_conv(fused1)
        

        srm_encoded = self.srm_org_encoder(srmed)[0]
        srmed_downsampled_encoded = self.srm_downsampled_encoder (srmed_downsampled ) [0]
        srm_encoded = F.interpolate(srm_encoded, size=srmed_downsampled_encoded.size(2), mode='nearest')

        fused2 = torch.cat((srm_encoded,srmed_downsampled_encoded ), 1) #torch.Size([1, 2080, 28, 28])
        fused2_conv = self.fuse2_conv(fused2)

        final_fusion = torch.cat((fused1_conv, fused2_conv), 1)
        
        aspped= self.aspp(final_fusion)
        out = self.docoder(aspped) 

        return out





