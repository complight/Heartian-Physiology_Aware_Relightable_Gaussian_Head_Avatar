import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

def create_backbone(backbone_name, pretrained=True,checkpoint_path=None):
    backbone = timm.create_model(backbone_name, 
                        pretrained=pretrained,checkpoint_path=checkpoint_path,
                        features_only=True)
    feature_dim = backbone.feature_info[-1]['num_chs']
    return backbone, feature_dim

class ExpressionEncoder(nn.Module):
    def __init__(self, n_exp=50) -> None:
        super().__init__()
        
        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100',pretrained=True,
                                                    )
        
        
        self.expression_layers = nn.Sequential( 
            nn.Linear(feature_dim, n_exp+2+3) # num expressions + jaw + eyelid
        )

        self.n_exp = n_exp
        self.init_weights()


    def init_weights(self):
        self.expression_layers[-1].weight.data *= 0.1
        self.expression_layers[-1].bias.data *= 0.1


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)


        parameters = self.expression_layers(features).reshape(img.size(0), -1)

        outputs = {}

        outputs['expression_params'] = parameters[...,:self.n_exp]
        outputs['eyelid_params'] = torch.clamp(parameters[...,self.n_exp:self.n_exp+2], 0, 1)
        outputs['jaw_params'] = torch.cat([F.relu(parameters[...,self.n_exp+2].unsqueeze(-1)), 
                                           torch.clamp(parameters[...,self.n_exp+3:self.n_exp+5], -.2, .2)], dim=-1)
        outputs["image_feature"]=features
        return outputs
    
class FlameParamsNetSmirk(nn.Module):
    def __init__(self, exp_dim=50):
        super(FlameParamsNetSmirk, self).__init__()
        
        
        self.model_path="./assets/smirk/pretrained_models/SMIRK_em1.pt"
        self.expression_encoder = ExpressionEncoder(n_exp=exp_dim)
        self.exp_dim=exp_dim
        self.load_initial_state() 

    
    def forward(self, img):
        return self.expression_encoder(img)
        
    def load_initial_state(self):
        checkpoint = torch.load(self.model_path)
        checkpoint_expression = {k.replace('smirk_encoder.expression_encoder.', ''): v for k, v in checkpoint.items() \
                                         if 'smirk_encoder.expression_encoder' in k}
        checkpoint_expression_encoder={k.replace('encoder.', ''): v for k, v in checkpoint_expression.items() \
                                         if 'encoder' in k}
        checkpoint_expression_mlp={k.replace('expression_layers.', ''): v for k, v in checkpoint_expression.items() \
                                         if 'expression_layers' in k}
        self.expression_encoder.encoder.load_state_dict(checkpoint_expression_encoder)
        if self.exp_dim==50:
            self.expression_encoder.expression_layers.load_state_dict(checkpoint_expression_mlp)
    
    def reload(self,state=0,ckpt_path=None):
        if state==0:
            self._state_dict=self.expression_encoder.state_dict()
            if ckpt_path is not None:
                ckpt_state=torch.load(ckpt_path)
                self.load_state_dict(ckpt_state)
            else:        
                self.load_initial_state()
        else:
            self.expression_encoder.load_state_dict(self._state_dict)
