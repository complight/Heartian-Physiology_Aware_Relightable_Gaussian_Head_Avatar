import torch
import numpy as np
from roma import quat_xyzw_to_wxyz,rotmat_to_unitquat

def calculate_point_normals(mesh):
    verts=mesh["v_template"]
    triangles=mesh["triangles"]
    i0 = triangles[..., 0].long()
    i1 = triangles[..., 1].long()
    i2 = triangles[..., 2].long()
    v0 = verts[ i0, :]
    v1 = verts[ i1, :]
    v2 = verts[ i2, :]
    #a0_=v1 - v0
    a01 = safe_normalize(v0 - v1)
    a02 = safe_normalize(v0 - v2)
    a12 = safe_normalize(v1 - v2)
    n0=safe_normalize( torch.cross(-a01, -a02, dim=-1))
    n1 = safe_normalize( torch.cross(-a12, a01, dim=-1))
    n2 = safe_normalize(torch.cross(a02, a12, dim=-1)) 
    normals = torch.zeros(verts.shape[0], 3).to(verts.device)
    normals.index_add_(0, triangles[:, 0].to(verts.device).long(),n0)
    normals.index_add_(0, triangles[:, 1].to(verts.device).long(),n1)
    normals.index_add_(0, triangles[:, 2].to(verts.device).long(),n2)
    normals = torch.nn.functional.normalize(normals, eps=1e-6, dim=1)
    return normals


def calculate_mesh_normals(mesh):
    xyz=mesh["v_template"]
    triangles=mesh["triangles"]
    triangle_normals=torch.zeros([triangles.shape[0],triangles.shape[1],3])
    #calculate normal of every verties
    normals_dict={}
    for tri_idx in range(triangles.shape[0]):
        point_1=xyz[triangles[tri_idx][0]]
        point_2=xyz[triangles[tri_idx][1]]
        point_3=xyz[triangles[tri_idx][2]]
        normal_1=torch.cross(point_2-point_1,point_3-point_1)
        normal_1=normal_1/torch.linalg.norm(normal_1)
        
        normal_2=torch.cross(point_3-point_2,point_1-point_2)
        normal_2=normal_2/torch.linalg.norm(normal_2)
            
        normal_3=torch.cross(point_1-point_3,point_2-point_3)
        normal_3=normal_3/torch.linalg.norm(normal_3)
        
        triangle_normals[tri_idx,0,:]=normal_1
        triangle_normals[tri_idx,1,:]=normal_2
        triangle_normals[tri_idx,2,:]=normal_3
        for i in range(3):
            if triangles[tri_idx][i].item() not in normals_dict.keys():
                normals_dict[triangles[tri_idx][i].item()]=[]

        normals_dict[triangles[tri_idx][0].item()].append(normal_1)
        normals_dict[triangles[tri_idx][1].item()].append(normal_2)
        normals_dict[triangles[tri_idx][2].item()].append(normal_3)
    pts_normals=torch.zeros_like(xyz,device=xyz.device)
    for p_idx in range(xyz.shape[0]):
        pts_normals[p_idx]=torch.stack(normals_dict[p_idx]).sum(dim=0)
        pts_normals[p_idx]=pts_normals[p_idx]/torch.linalg.norm(pts_normals[p_idx])
    #pts_normals=torch.tensor(pts_normals,device=xyz.device)
    return triangle_normals,pts_normals


def sample_initial_points_from_mesh(mesh,num_pts_sample_fmesh=5,along_normal_scale=0.1):
    #sampling on and beyond mesh suface
    triangles=mesh["triangles"]
    xyz=mesh["v_template"]
    shape_dirs,expression_dirs,pose_dirs,lbs_weights,r_eyelid,l_eyelid=\
    mesh["shape_dirs"],mesh["expression_dirs"],mesh["pose_dirs"],mesh["lbs_weights"],mesh["r_eyelid"],mesh["l_eyelid"]
    
    triangle_normals,pts_normals=calculate_mesh_normals(mesh)
    num_mesh=triangles.shape[0]
    xyz_weight_mesh=torch.rand((num_mesh,num_pts_sample_fmesh, 3))
    xyz_weight_normal=torch.rand((num_mesh,num_pts_sample_fmesh, 1))*along_normal_scale
    #sampling on mesh suface
    xyz_weight_normal[:,:int(num_pts_sample_fmesh/2),:]=0.0
    
    xyz_weight_mesh=xyz_weight_mesh/xyz_weight_mesh.sum(dim=2).reshape(num_mesh,num_pts_sample_fmesh,1)
    triangles_xyz=xyz[triangles,:]
    triangles_shape_dirs=shape_dirs[triangles,:]
    triangles_expression_dirs=expression_dirs[triangles,:]
    triangles_pose_dirs=pose_dirs[triangles,:]
    triangles_lbs_weights=lbs_weights[triangles,:]
    triangle_r_eyelid=r_eyelid[:,triangles,:]
    triangle_l_eyelid=l_eyelid[:,triangles,:]
    
    xyz_sampled=xyz_weight_mesh@triangles_xyz
    
    shape_dirs_sampled=torch.einsum('bjk,bklh->bjlh',xyz_weight_mesh,triangles_shape_dirs)\
        .reshape(num_mesh*num_pts_sample_fmesh,shape_dirs.shape[1],shape_dirs.shape[2])
    expression_dirs_sampled=torch.einsum('bjk,bklh->bjlh',xyz_weight_mesh,triangles_expression_dirs)\
        .reshape(num_mesh*num_pts_sample_fmesh,expression_dirs.shape[1],expression_dirs.shape[2])
    pose_dirs_sampled=torch.einsum('bjk,bklh->bjlh',xyz_weight_mesh,triangles_pose_dirs)\
        .reshape(num_mesh*num_pts_sample_fmesh,pose_dirs.shape[1],pose_dirs.shape[2])
    
    lbs_weights_sampled=(xyz_weight_mesh@triangles_lbs_weights)\
        .reshape(num_mesh*num_pts_sample_fmesh,lbs_weights.shape[1])
    r_eyelid_sampled=(xyz_weight_mesh@triangle_r_eyelid[0,:,:,:])\
        .reshape(num_mesh*num_pts_sample_fmesh,r_eyelid.shape[-1])
    l_eyelid_sampled=(xyz_weight_mesh@triangle_l_eyelid[0,:,:,:])\
        .reshape(num_mesh*num_pts_sample_fmesh,l_eyelid.shape[-1])
        
    normal_sampled=xyz_weight_mesh@triangle_normals
    xyz_sampled=xyz_sampled+normal_sampled*xyz_weight_normal

    xyz_sampled=xyz_sampled.reshape(-1,3)
    
    return xyz_sampled,shape_dirs_sampled,expression_dirs_sampled,pose_dirs_sampled,lbs_weights_sampled,\
            r_eyelid_sampled,l_eyelid_sampled,normal_sampled.reshape(num_mesh*num_pts_sample_fmesh,3)

def calculate_mesh_avg_edge_length(mesh):
    xyz=mesh["v_template"]
    triangles=mesh["triangles"]
    triangle_edge_lengths=torch.zeros(triangles.shape[0],3)
    for tri_idx in range(triangles.shape[0]):
        point_1=xyz[triangles[tri_idx][0]]
        point_2=xyz[triangles[tri_idx][1]]
        point_3=xyz[triangles[tri_idx][2]]
        triangle_edge_lengths[tri_idx,0]=torch.norm(point_1-point_2,p=2)
        triangle_edge_lengths[tri_idx,1]=torch.norm(point_1-point_3,p=2)
        triangle_edge_lengths[tri_idx,2]=torch.norm(point_2-point_3,p=2)
    return triangle_edge_lengths.mean()


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) 
    # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

def calculate_triangles_orientation(verts, triangles, return_scale=False):
    i0 = triangles[..., 0].long()
    i1 = triangles[..., 1].long()
    i2 = triangles[..., 2].long()

    v0 = verts[ i0, :]
    v1 = verts[ i1, :]
    v2 = verts[ i2, :]

    #a0_=v1 - v0
    a0 = safe_normalize(v1 - v0)
    
    a1 = safe_normalize( torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  
    # will have artifacts without negation

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)
    scale= None
    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
        #scale=torch.sqrt(length(a1_corss))
    return orientation, scale

def calculate_vertex_adjacency(vertexes,triangles,adj_num=6):
    
    num_vertex=vertexes.shape[0]
    adjacency_list = [[] for _ in range(num_vertex)]

    for triangle in triangles:
        for i in range(3):
            vertex_idex = triangle[i]
            for j in range(3):
                if j != i and triangle[j] not in adjacency_list[vertex_idex]:
                    adjacency_list[vertex_idex].append(triangle[j])

    adjacency_tensor = [torch.tensor(adjacent_vertices) for adjacent_vertices in adjacency_list]
    adjacency_tensor=torch.arange(num_vertex,device=triangles.device).unsqueeze(1).repeat(1,adj_num)
    for i in range(num_vertex):
        if len(adjacency_list[i])>adj_num:
            adjacency_list[i]=adjacency_list[i][:adj_num]
        adjacency_tensor[i,:len(adjacency_list[i])]=torch.tensor(adjacency_list[i],device=triangles.device,dtype=torch.long)
    return adjacency_tensor


def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    v_normals = torch.zeros_like(verts)

    for i in range(face_normals.shape[0]):
        v_normals[i0[i]] += face_normals[i]
        v_normals[i1[i]] += face_normals[i]
        v_normals[i2[i]] += face_normals[i]
        
    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals

def init_vertex_rot(vertexes,triangles):
     triangles_orient,_=calculate_triangles_orientation(vertexes, triangles, return_scale=False)
     triangles_orient_quad=quat_xyzw_to_wxyz(rotmat_to_unitquat(triangles_orient))
     vertexes_orient=torch.zeros([vertexes.shape[0],4],device=vertexes.device)
     vertexes_orient_count=torch.zeros([vertexes.shape[0],1],device=vertexes.device)
     for tri_idx in range(triangles.shape[0]):
         for i in range(3):
             vertexes_orient[triangles[tri_idx][i]]+=triangles_orient_quad[tri_idx]
             vertexes_orient_count[triangles[tri_idx][i]]+=1
             
     vertexes_orient=vertexes_orient/vertexes_orient_count
     
     return vertexes_orient