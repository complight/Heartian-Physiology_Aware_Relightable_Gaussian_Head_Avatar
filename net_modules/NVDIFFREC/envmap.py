

import os.path
import os,sys,glob,imageio
from PIL import Image
import numpy as np
import torch,torchvision
import nvdiffrast.torch as dr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import util
import renderutils as ru



def fresnelSchlick(cosTheta,roughness,F0=torch.tensor(0.04)):
    
    return F0 + (torch.max((1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0)
def fresnelSchlick_2(cosTheta,roughness,F0=torch.tensor(0.04)):
    
    FMi = ((-5.55473) * cosTheta - 6.98316) * cosTheta
    return F0 + (torch.max((1.0 - roughness), F0) - F0) * pow(2, FMi)

class EnvironmentMap(torch.nn.Module):
    def __init__(self, diffuse_dim,specular_dim,max_mipmap_level=4,device="cuda:0"):
        super(EnvironmentMap, self).__init__()
        self.diffuse_dim=int(diffuse_dim)
        self.specular_dim=int(specular_dim)
        self.diffuse_map = torch.nn.Parameter(torch.ones((6,self.diffuse_dim,self.diffuse_dim,3),device=device)*0.5, requires_grad=True)
        self.register_parameter('diffuse_map', self.diffuse_map)
        self.specular_map = torch.nn.Parameter(torch.ones((6,self.specular_dim,self.specular_dim,3),device=device)*0.5, requires_grad=True)
        self.register_parameter('specular_map', self.specular_map)
        self.diffuse = ru.diffuse_cubemap(self.diffuse_map)
        self.specular=ru.specular_cubemap(self.specular_map,1.0)

        self.F0=0.04
        
        self.roughness_scale=1.0
        self.reflectance_scale=0.0
        self._FG_LUT = torch.as_tensor(np.fromfile('./net_modules/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
        self._FG_LUT.requires_grad = False
        self.activation=torch.nn.Sigmoid()
        self.max_mipmap_level=max_mipmap_level
        self.use_fresnel=True
        
        self.cache_mipmap=None
        self._eval=False
    def update(self):
        self.diffuse = ru.diffuse_cubemap(self.diffuse_map)
        self.specular=ru.specular_cubemap(self.specular_map,1.0)
        
    def set_reflectance_scale(self,scale):
        self.reflectance_scale=scale
        
    def set_roughness_scale(self,scale):
        self.roughness_scale=scale

    def set_eval(self,state):
        self._eval=state

    def shading(self,albedo,roughness,reflectance,normal,viewdir,alpha=None,bgcolor=0.0):
        #viewdir point to camera
        #refldir point to light

        #edting roughness(decrease) and reflectance(increase)
        roughness=roughness*self.roughness_scale
        reflectance=reflectance+(1-reflectance)*self.reflectance_scale

        NoV=torch.sum(normal* viewdir, dim=-1, keepdim=True).clamp_(1e-6, 1)
        NoV=NoV.detach()
        refldir=2*normal*NoV-viewdir
        F = fresnelSchlick_2(NoV, roughness[0,...,None],reflectance[0,...,None])
        kS = F[None,:,:,0]
        
        irradiance =dr.texture(self.diffuse[None, ...], normal[None].contiguous(), filter_mode='linear', boundary_mode='cube')[0].permute(2,0,1)
        spec = dr.texture(self.specular[None, ...], refldir[None].contiguous(),mip_level_bias=roughness*self.max_mipmap_level,  filter_mode='linear-mipmap-linear',max_mip_level=self.max_mipmap_level, boundary_mode='cube')[0].permute(2,0,1)

        irradiance,spec=self.activation(irradiance),self.activation(spec)

        fg_uv = torch.cat((NoV, roughness[0,...,None]), dim=-1)
        envBRDF = dr.texture(self._FG_LUT, fg_uv[None], filter_mode='linear', boundary_mode='clamp')

        diffuse = irradiance * albedo
        spec_int=(kS * envBRDF[...,0] + envBRDF[...,1])
        specular = spec * spec_int
        kD = 1.0
        shade_result=(kD * diffuse + specular)
        if alpha is not None:
            shade_result=shade_result*alpha+bgcolor[:,None,None]*(1-alpha)
        result={
            "shading":shade_result,"diffuse":diffuse,
            "specular":spec,"irradiance":irradiance,
            "KS":kS,"specular_intensity":spec_int,"I_specular":specular,
        }

        return result
    
class EnvironmentMap_relight(torch.nn.Module):
    def __init__(self, lalong_dir,init_res=1024,mip_map_count=7,down_sample=1.0,device="cuda:0"):
        super(EnvironmentMap_relight, self).__init__()
        diffuse_path=os.path.join(lalong_dir,"diffuse.tga")
        
        self.device=device
        self.rotate_y=0
        print("Loading prefilter environment maps...")
        self.F0=0.04
        self.metalness=0.0
        self.reflectance_scale=0.0
        self.roughness_scale=1.0
        
        self._FG_LUT = torch.as_tensor(np.fromfile('./net_modules/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
        self._FG_LUT.requires_grad = False

        diffuse_lalong_map=torch.tensor(np.array(Image.open(diffuse_path)),device=device)/255
        if down_sample>1:
            down_h,down_w=max(int(diffuse_lalong_map.shape[0]/down_sample),8),max(int(diffuse_lalong_map.shape[1]/down_sample),16)
            diffuse_lalong_map= torch.nn.functional.interpolate(diffuse_lalong_map.permute(2, 0, 1).unsqueeze(0), size=(down_h,down_w), mode='bilinear', align_corners=False)
            diffuse_lalong_map = diffuse_lalong_map.squeeze(0).permute(1, 2, 0).contiguous()
        
        w,h=init_res,init_res
        self.origin_lalong_map=None
        hdr_file=glob.glob(os.path.join(lalong_dir,f'*.hdr'), recursive=True)
        if len(hdr_file) !=0:
            self.origin_lalong_map=torch.tensor(util.load_image(hdr_file[0]),device=device,dtype=torch.float32)
            torchvision.utils.save_image(self.origin_lalong_map.permute(2,0,1), os.path.join(lalong_dir, os.path.basename(lalong_dir).split(".")[0] + ".png"))
        specular_lalong_map=[]
        for i in range(mip_map_count):
            w=int(h)
            h=int(w/2)
            specular_path=diffuse_path=os.path.join(lalong_dir,f"specular_{i}_{w}x{h}.tga")
            lalong_map=torch.tensor(np.array(Image.open(specular_path)),device=device)/255
            if down_sample>1:
                down_h,down_w=int(lalong_map.shape[0]/down_sample),int(lalong_map.shape[1]/down_sample)
                if down_h<4:
                    break
                lalong_map= torch.nn.functional.interpolate(lalong_map.permute(2, 0, 1).unsqueeze(0), size=(down_h,down_w), mode='bilinear', align_corners=False)
                lalong_map = lalong_map.squeeze(0).permute(1, 2, 0).contiguous()
            specular_lalong_map.append(lalong_map)
        self.mip_map_count=len(specular_lalong_map)
        print("Loading complete.")

        self.specular_lalong_map=specular_lalong_map
        self.device=device
        with torch.no_grad():
            self.diffuse_map=util.latlong_to_cubemap(diffuse_lalong_map,[diffuse_lalong_map.shape[0],diffuse_lalong_map.shape[0]],device=device)
            self.diffuse = ru.diffuse_cubemap(self.diffuse_map)
            self.specular_map=[]
            self.specular=[]
            for idx,latlong_map in enumerate(specular_lalong_map):
                self.specular_map.append(util.latlong_to_cubemap(latlong_map, [latlong_map.shape[0],latlong_map.shape[0]],device=device))
                self.specular.append(ru.specular_cubemap(self.specular_map[-1], (idx+1)/mip_map_count))

    def get_mip(self, roughness):
        return torch.tensor(roughness*(self.mip_map_count-1),device=self.device).contiguous()
    def set_rotate_y(self,degree):
        self.rotate_y=degree
    def set_reflectance_scale(self,scale):
        self.reflectance_scale=scale
        self.metalness=1.0*(scale)
    def set_roughness_scale(self,scale):
        self.roughness_scale=scale
    def get_back_map(self,rotate_y,res=[512,512]):
        lalong_map=self.specular_lalong_map[0]
        if self.origin_lalong_map is not  None:
            lalong_map=self.origin_lalong_map
        return util.latlong_to_backmap(lalong_map,res,rotate_y,self.device)[0].permute(2,0,1)

    @torch.no_grad()
    def shading(self,albedo,roughness,reflectance,normal,viewdir,alpha=None,bgcolor=0.0):
        #viewdir point to camera
        #refldir point to light
        # rotate_y: Rotate envmap along the y-axis by an angle equivalent to 
                    # rotating the normal and reflection vectors in opposite angle

        #edting roughness(decrease) and reflectance(increase)
        roughness=roughness*self.roughness_scale
        reflectance=reflectance+(1-reflectance)*self.reflectance_scale
        
        NoV=torch.sum(normal* viewdir, dim=-1, keepdim=True).clamp_(1e-6, 1)
        refldir=2*normal*NoV-viewdir
        if self.rotate_y!=0:
            mtx = util.rotate_y(-self.rotate_y/180*torch.pi,device=self.device)[None]
            refldir = ru.xfm_vectors(refldir.view(1, refldir.shape[0] * refldir.shape[1], refldir.shape[2]), mtx).view(*refldir.shape)
            normal  = ru.xfm_vectors(normal.view(1, normal.shape[0] * normal.shape[1], normal.shape[2]), mtx).view(*normal.shape)

        F = fresnelSchlick_2(NoV, roughness[0,...,None],reflectance[0,...,None])
        kS = F[None,:,:,0]
        kD = 1.0*(1-self.metalness)
        irradiance =dr.texture(self.diffuse[None, ...], normal[None].contiguous(), filter_mode='linear', boundary_mode='cube')[0].permute(2,0,1)
        
        miplevel = self.get_mip(roughness)
        spec = dr.texture(self.specular[0][None, ...], refldir[None].contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel, \
                          filter_mode='linear-mipmap-linear', boundary_mode='cube')[0].permute(2,0,1)
        if alpha is not None:
            spec=spec*alpha
            irradiance=irradiance*alpha
            kS=kS*alpha
        fg_uv = torch.cat((NoV, roughness[0,...,None]), dim=-1)
        envBRDF = dr.texture(self._FG_LUT, fg_uv[None], filter_mode='linear', boundary_mode='clamp')
        
        diffuse = irradiance * albedo
        spec_int= (kS * envBRDF[...,0] + envBRDF[...,1])
        specular = spec *spec_int
        shade_result=(kD * diffuse + specular)
        if alpha is not None:
            shade_result=shade_result*alpha+bgcolor[:,None,None]*(1-alpha)
        result={
            "shading":shade_result,"diffuse":diffuse,
            "specular":spec,"irradiance":irradiance,
            "KS":kS,"specular_intensity":spec_int,"I_specular":specular,
        }
        return result
    
from argparse import ArgumentParser
if __name__=="__main__":
    parser = ArgumentParser(description="Test loading envmap")
    parser.add_argument("--map_path", type=str)
    args=parser.parse_args()
    Envmap=EnvironmentMap_relight(args.map_path)