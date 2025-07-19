import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_imagenet(x):
    """Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x

def make_mlp(dims, activation=nn.ReLU, batch_norm=False):
    """Create an MLP for SDF decoder etc."""
    assert len(dims) >= 2
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Conv1d(dims[i], dims[i + 1], kernel_size=1, bias=True))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        if activation:
            layers.append(activation())
    return nn.Sequential(*layers)

class VGG16WithFeatures(nn.Module):
    """修改版VGG16，可以返回中间层特征"""
    def __init__(self, pretrained=True, num_classes=1000, in_channels=3):
        super(VGG16WithFeatures, self).__init__()
        
        # 加载预训练VGG16
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # 如果输入通道不是3，修改第一层
        if in_channels != 3:
            first_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            if pretrained:
                # 初始化新通道的权重
                with torch.no_grad():
                    first_conv.weight[:, :3] = vgg16.features[0].weight
                    if in_channels > 3:
                        first_conv.weight[:, 3:] = 0
            vgg16.features[0] = first_conv
            
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier
        
        # 修改分类器输出
        if num_classes != 1000:
            self.classifier[-1] = nn.Linear(4096, num_classes)
            
        # 用于存储特征的钩子
        self.feature_outputs = []
        self.hooks = []
        
        # 为特定层添加前向钩子
        feature_layers = [3, 8, 15, 22, 29]  # 对应VGG16中的某些卷积层后
        for i in feature_layers:
            self.hooks.append(
                self.features[i].register_forward_hook(self._feature_hook)
            )
    
    def _feature_hook(self, module, input, output):
        self.feature_outputs.append(output)
    
    def forward(self, x):
        self.feature_outputs = []
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, self.feature_outputs

class DISNEncoder(nn.Module):
    def __init__(self,
                 image_size=None,
                 use_pretrained_image_encoder=True,
                 local_feature_size=137,
                 image_encoding_dim=1000,
                 normalize=True,
                 resize_local_feature=True,
                 resize_input_shape=True,
                 in_channels=3):
        super().__init__()

        self.image_size = image_size
        self.local_feature_size = local_feature_size
        self.use_pretrained_image_encoder = use_pretrained_image_encoder
        self.resize_local_feature = resize_local_feature
        self.resize_input_shape = resize_input_shape
        self.image_encoder = VGG16WithFeatures(
            pretrained=use_pretrained_image_encoder,
            num_classes=image_encoding_dim,
            in_channels=in_channels)
        self.normalize = normalize

    def forward(self, images):
        """
        Args:
            images: (batch_size, channels, width, height). value in range [0, 1]

        Returns:
            Tuple: (global_features, encoder_features)
            global_features: the feature for each image (batch_size, num_feats).
            encoder_features: a tuple of output from the image encoder's intermediate layers.
        """
        # 调整图像大小
        if self.resize_input_shape and self.image_size is not None and (
                images.shape[2] != self.image_size or
                images.shape[3] != self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear')

        # 编码图像
        if self.normalize:
            images = normalize_imagenet(images)

        global_features, encoder_outputs = self.image_encoder(images)
        
        # 调整特征图大小
        if self.resize_local_feature:
            resized_outputs = [
                F.interpolate(
                    output,
                    size=(self.local_feature_size, self.local_feature_size),
                    mode='bilinear')
                for output in encoder_outputs
            ]
        else:
            resized_outputs = encoder_outputs
            
        # 将全局特征添加到输出列表中
        resized_outputs.insert(0, global_features)
        return resized_outputs

class SDFGlobalDecoder(nn.Module):
    def __init__(self, out_features, batch_norm=False):
        super().__init__()
        self.mlp1 = make_mlp([3, 64, 256, 512], batch_norm=batch_norm)
        self.mlp2 = make_mlp([1512, 512, 256], batch_norm=batch_norm)
        self.mlp3 = make_mlp([256, out_features], activation=None, batch_norm=False)

    def forward(self, query_points, global_features):
        """
        Args:
            query_points: (batch_size, num_points, 3)
            global_features: (batch_size, num_feats) 或 (batch_size, 1, num_feats)
        """
        batch_size, num_points, _ = query_points.shape
        
        # 确保global_features格式正确
        if global_features.dim() == 2:
            global_features = global_features.unsqueeze(1).expand(-1, num_points, -1)
            
        x = self.mlp1(query_points.permute(0, 2, 1))
        x = torch.cat((x, global_features.permute(0, 2, 1)), axis=1)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x.permute(0, 2, 1)

class SDFLocalDecoder(nn.Module):
    def __init__(self, out_features, batch_norm=False):
        super().__init__()
        self.mlp1 = make_mlp([3, 64, 512], batch_norm=batch_norm)
        self.mlp2 = make_mlp([1984, 512, 256], batch_norm=batch_norm)  # 1984 = 512 + 1472 (假设local_features尺寸)
        self.mlp3 = make_mlp([256, out_features], activation=None, batch_norm=False)

    def forward(self, query_points, local_features):
        """
        Args:
            query_points: (batch_size, num_points, 3)
            local_features: (batch_size, num_points, num_feats)
        """
        x = self.mlp1(query_points.permute(0, 2, 1))
        x = torch.cat((x, local_features.permute(0, 2, 1)), axis=1)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x.permute(0, 2, 1)

class DISNDecoder(nn.Module):
    def __init__(self, out_features, batch_norm=False):
        super().__init__()
        self.sdf_global_decoder = SDFGlobalDecoder(out_features, batch_norm=batch_norm)
        self.sdf_local_decoder = SDFLocalDecoder(out_features, batch_norm=batch_norm)

    def forward(self, encoded_features, camera_matrix=None, tmp=None):
        """原始DISNDecoder的forward方法"""
        # 从编码特征中提取各部分
        global_features = encoded_features[:, :, :1000]
        local_features = encoded_features[:, :, 1000:-3]
        query_points = encoded_features[:, :, -3:]

        # 解码SDF
        global_pred = self.sdf_global_decoder(query_points, global_features)
        local_pred = self.sdf_local_decoder(query_points, local_features)
        pred = global_pred + local_pred

        return pred

class DISN(nn.Module):
    """
    DISN (Deep Implicit Surface Network)模型，整合了图像特征提取和位置解码功能
    """
    def __init__(self, 
                 output_dim=3, 
                 batch_norm=False, 
                 image_size=64,
                 local_feature_size=64,
                 image_encoding_dim=1000,
                 resize_input_shape=True,
                 resize_local_feature=True,
                 in_channels=3):
        super().__init__()
        
        # 创建两个编码器实例，按照DeformableTetNetwork中的实现
        encoder_1 = DISNEncoder(
            image_size=image_size,
            local_feature_size=local_feature_size,
            image_encoding_dim=image_encoding_dim,
            resize_input_shape=resize_input_shape,
            resize_local_feature=resize_local_feature,
            in_channels=in_channels
        )
        
        encoder_2 = DISNEncoder(
            image_size=image_size,
            local_feature_size=local_feature_size,
            image_encoding_dim=image_encoding_dim,
            resize_input_shape=resize_input_shape,
            resize_local_feature=resize_local_feature,
            in_channels=in_channels
        )
        
        # 将两个编码器放入ModuleList中
        self.encoder = nn.ModuleList([encoder_1, encoder_2])
        
        # 解码器
        self.decoder = DISNDecoder(output_dim, batch_norm=batch_norm)
        
    def encode_images(self, images):
        """
        使用两个编码器分别提取图像特征，不合并结果
        返回两个编码器的特征列表
        """
        if images is None:
            return None
            
        img_feat_1 = self.encoder[0](images)
        img_feat_2 = self.encoder[1](images)
        
        # 返回两个特征列表，与原始代码保持一致
        return [img_feat_1, img_feat_2]
    
    def sample_features(self, pos, features, cam_pos=None, cam_rot=None, cam_proj=None):
        """
        采样特征，对应原始代码中的sample_f方法
        """
        # 这里简化实现，未来可以根据需要扩展
        # 在位置解码中，我们只使用第一个编码器的特征
        if features is None:
            return None
        
        # 返回第一个编码器的特征（即encoding[0]）
        return features[0]
        
    def prepare_input_features(self, pos, encoded_features):
        """
        准备输入给Decoder的特征张量
        """
        batch_size, num_points, _ = pos.shape
        
        if encoded_features is None:
            # 如果没有图像特征，创建默认的全0特征
            global_features = torch.zeros(batch_size, num_points, 1000, device=pos.device)
            local_features = torch.zeros(batch_size, num_points, 1472, device=pos.device)
        else:
            # 提取全局特征并复制到每个点
            global_feat = encoded_features[0]
            if global_feat.dim() == 2:
                global_features = global_feat.unsqueeze(1).expand(-1, num_points, -1)
            else:
                global_features = global_feat
                
            # 合并所有局部特征
            if len(encoded_features) > 1:
                # 将编码器提取的不同层次的特征拼接到一起
                # 注意：这是个简化实现
                local_features = torch.zeros(batch_size, num_points, 1472, device=pos.device)
            else:
                local_features = torch.zeros(batch_size, num_points, 1472, device=pos.device)
                
        # 按照原始代码格式拼接：global_features + local_features + points
        combined_features = torch.cat([global_features, local_features, pos], dim=2)
        return combined_features
        
    def decode_pos(self, pos, z, c, init_pos_mask=None, cam_pos=None, cam_rot=None, cam_proj=None):
        """
        位置解码函数，对应原始代码中的decode_pos方法
        """
        # 采样特征
        pos_feature = self.sample_features(pos, c, cam_pos=cam_pos, cam_rot=cam_rot, cam_proj=cam_proj)
        
        # 拼接特征和位置
        if pos_feature is not None:
            pos_feature = torch.cat([pos_feature[0][...,None], pos], dim=1)
            
            # 准备解码器输入 
            pos_feature = self.prepare_input_features(pos, [pos_feature])
            
            # 解码位置
            pos_delta = self.decoder(pos_feature)
            pos_delta = pos_delta * 0.1  # 按照原始代码缩放
        else:
            # 如果没有特征，返回零变形
            pos_delta = torch.zeros_like(pos)
            
        # 处理位置掩码
        if init_pos_mask is not None:
            pos_delta = pos_delta * init_pos_mask
            
        # 计算最终位置
        final_pos = pos + pos_delta
        
        return pos_delta, final_pos, pos_delta
        
    def forward(self, pos, image=None, cam_matrix=None, init_pos_mask=None):
        """
        模型前向传播
        
        参数:
            pos: 输入位置，形状为 [B, n_p, 3] 或 [n_p, 3]
            image: 输入图像，形状为 [B, C, H, W]
            cam_matrix: 相机矩阵，[B, 4, 4]
            init_pos_mask: 初始位置掩码
            
        返回:
            变形后的位置
        """
        # 确保输入批次维度
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)  # [n_p, 3] -> [1, n_p, 3]
            if init_pos_mask is not None:
                init_pos_mask = init_pos_mask.unsqueeze(0)
        
        # 标准化为 [B, n_p, 3] 格式
        if pos.shape[1] == 3 and pos.shape[2] != 3:
            # 转为 [B, n_p, 3]
            pos = pos.permute(0, 2, 1)
            input_format = "3_first"
        else:
            input_format = "3_last"
        
        if image.dim() == 3 and image is not None:
            image = image.unsqueeze(0)
        
        if image.shape[3] == 3:
            image = image.permute(3,1,2)

        # 提取图像特征
        encoded_features = None
        if image is not None:
            encoded_features = self.encode_images(image)
        
        # 解码位置
        _, final_pos, _ = self.decode_pos(
            pos, None, encoded_features, 
            init_pos_mask=init_pos_mask,
            cam_pos=None, cam_rot=None, cam_proj=cam_matrix
        )
        
        # 恢复原始格式
        if input_format == "3_first":
            final_pos = final_pos.permute(0, 2, 1)
        
        # 如果输入是单个点云，移除批次维度
        if pos.shape[0] == 1 and pos.dim() == 3:
            final_pos = final_pos.squeeze(0)
        
        return final_pos

def test_disn():
    """
    测试DISN模型，验证位置变形功能是否正常工作
    """
    print("开始DISN测试...")
    
    # 设置随机种子以便结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建随机点云 [N, 3]
    N = 1000
    points = torch.rand(N, 3) * 2 - 1  # 范围在 [-1, 1]
    print(f"生成的点云形状: {points.shape}")
    
    # 创建随机图像 [C, H, W]
    C, H, W = 3, 64, 64
    image = torch.rand(C, H, W)
    print(f"生成的图像形状: {image.shape}")
    
    # 创建随机相机矩阵 [4, 4]
    cam_matrix = torch.eye(4)  # 简单的单位矩阵作为相机矩阵
    
    # 将数据移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = points.to(device)
    image = image.to(device)
    cam_matrix = cam_matrix.to(device)
    
    print(f"使用设备: {device}")
    
    try:
        # 直接测试DISN模型
        print("\n测试DISN模型（带图像）...")
        model = DISN().to(device)
        with torch.no_grad():
            # 使用图像
            output_with_image = model(points, image, cam_matrix=None)
            print(f"带图像输出形状: {output_with_image.shape}")
            
            # 不使用图像
            print("\n测试DISN模型（不带图像）...")
            output_no_image = model(points, None, None)
            print(f"不带图像输出形状: {output_no_image.shape}")
        
        # 检查输出是否包含NaN
        if torch.isnan(output_with_image).any() or torch.isnan(output_no_image).any():
            print("警告: 输出包含NaN值")
        else:
            print("输出正常，没有NaN值")
            
        # 计算变形的量
        displacement_with_image = torch.norm(output_with_image - points, dim=1).mean().item()
        displacement_no_image = torch.norm(output_no_image - points, dim=1).mean().item()
        print(f"使用图像的平均变形量: {displacement_with_image:.6f}")
        print(f"不使用图像的平均变形量: {displacement_no_image:.6f}")
        
        print("\nDISN测试成功完成!")
        return True
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_disn()


