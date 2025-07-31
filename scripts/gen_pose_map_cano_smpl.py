

import tqdm
import numpy  as np
import torch
from os.path import join
import os
import sys
sys.path.append('../')
from submodules  import smplx

from scipy.spatial.transform import Rotation as R
import trimesh
from utils.general_utils import load_masks, load_barycentric_coords, gen_lbs_weight_from_ori
from arguments import smplx_cpose_param, smpl_cpose_param
import cv2
def render_posmap(v_minimal, faces, uvs, faces_uvs, img_size=32):
    '''
    v_minimal: vertices of the minimally-clothed SMPL body mesh
    faces: faces (triangles) of the minimally-clothed SMPL body mesh
    uvs: the uv coordinate of vertices of the SMPL body model
    faces_uvs: the faces (triangles) on the UV map of the SMPL body model
    '''
    from posmap_generator.lib.renderer.gl.pos_render import PosRender

    # instantiate renderer
    rndr = PosRender(width=img_size, height=img_size)

    # set mesh data on GPU
    rndr.set_mesh(v_minimal, faces, uvs, faces_uvs)

    # render
    rndr.display()

    # retrieve the rendered buffer
    uv_pos = rndr.get_color(0)
    uv_mask = uv_pos[:, :, 3]
    uv_pos = uv_pos[:, :, :3]

    uv_mask = uv_mask.reshape(-1)
    uv_pos = uv_pos.reshape(-1, 3)

    rendered_pos = uv_pos[uv_mask != 0.0]

    uv_pos = uv_pos.reshape(img_size, img_size, 3)

    # get face_id (triangle_id) per pixel
    face_id = uv_mask[uv_mask != 0].astype(np.int32) - 1

    assert len(face_id) == len(rendered_pos)

    return uv_pos, uv_mask, face_id


def compute_barycentric_coords(pt, tri_uv):
    v0 = tri_uv[1] - tri_uv[0]
    v1 = tri_uv[2] - tri_uv[0]
    v2 = np.array(pt) - tri_uv[0]

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-8:
        return np.array([-1, -1, -1])

    inv_denom = 1.0 / denom
    w1 = (d11 * d20 - d01 * d21) * inv_denom
    w2 = (d00 * d21 - d01 * d20) * inv_denom
    w0 = 1.0 - w1 - w2
    return np.array([w0, w1, w2])

def render_posmap_cpu(vertices, faces, uvs, faces_uvs, img_size=512):
    uv_coords = uvs.copy()
    uv_coords[:, 0] *= (img_size - 1)
    uv_coords[:, 1] *= (img_size - 1)
    uv_coords[:, 1] = (img_size - 1) - uv_coords[:, 1]

    posmap = np.zeros((img_size, img_size, 3), dtype=np.float32)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    for f_idx, face in tqdm.tqdm(enumerate(faces), total=len(faces), desc="Rendering UV map"):
        tri_uv_idx = faces_uvs[f_idx]
        tri_vert_idx = face
        tri_uv = uv_coords[tri_uv_idx]
        tri_pos = vertices[tri_vert_idx]

        contour = np.round(tri_uv).astype(np.int32)
        triangle_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillConvexPoly(triangle_mask, contour, 1)

        ys, xs = np.where(triangle_mask == 1)
        if len(xs) == 0:
            continue

        for x, y in zip(xs, ys):
            bc = compute_barycentric_coords((x, y), tri_uv)
            if np.any(bc < -1e-4):
                continue
            posmap[y, x] = bc[0]*tri_pos[0] + bc[1]*tri_pos[1] + bc[2]*tri_pos[2]
            mask[y, x] = 1

    return posmap, mask

def save_obj(data_path, name):
    smpl_data = torch.load( data_path + '/smpl_parms.pth')
    smpl_model = smplx.SMPL(model_path ='../assets/smpl_files/smpl',batch_size = 1)
    cano_dir = os.path.join(data_path,)


    cano_smpl = smpl_model.forward(betas=smpl_data['beta'],
                            global_orient=smpl_cpose_param[:, :3],
                            transl = torch.tensor([[0, 0.30, 0]]),
                            # global_orient=cpose_param[:, :3],
                            body_pose=smpl_cpose_param[:, 3:],
                            )

    ori_vertices = cano_smpl.vertices.detach().cpu().numpy().squeeze()
    joint_mat = cano_smpl.A
    print(joint_mat.shape)
    torch.save(joint_mat ,join(cano_dir, 'smpl_cano_joint_mat.pth'))


    mesh = trimesh.Trimesh(ori_vertices, smpl_model.faces, process=False)
    mesh.export('%s/%s.obj' % (cano_dir, 'cano_smpl'))



def save_npz(data_path, res=128):
    from posmap_generator.lib.renderer.mesh import load_obj_mesh
    verts, faces, uvs, faces_uvs = load_obj_mesh(uv_template_fn, with_texture=True)
    start_obj_num = 0
    result = {}
    body_mesh = trimesh.load('%s/%s.obj'%(data_path, 'cano_smpl'), process=False)

    if res==128:
        posmap128, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=128)
        result['posmap128'] = posmap128   
    elif res == 256:
    
        posmap256, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=256)
        result['posmap256'] = posmap256

    else:
        posmap512, _= render_posmap_cpu(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=512)
        result['posmap512'] = posmap512

    save_fn = join(data_path, 'query_posemap_%s_%s.npz'% (str(res), 'cano_smpl'))
    np.savez(save_fn, **result)



if __name__ == '__main__':
    smplx_parm_path = '/kaggle/working/GaussianAvatar/content/gs_3d_ava/train' # path to the folder that include smpl params
    parms_name = 'smpl_parms.pth'
    uv_template_fn = '../assets/template_mesh_smpl_uv.obj'
    assets_path = '/kaggle/working/GaussianAvatar/assets'    # path to the folder that include 'assets'

    print('saving obj...')
    save_obj(smplx_parm_path, parms_name)

    print('saving pose_map 512 ...')
    save_npz(smplx_parm_path, 512)