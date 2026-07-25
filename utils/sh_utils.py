#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5


Cp0 = 0.28209479177387814
Cp1 = 0.4886025119029199
Cp2 = [
    1.0925484305920792,
    1.0925484305920792,
    0.31539156525252005,
    1.0925484305920792,
    0.5462742152960396
]
Cp3 = [
    0.5900435899266435,
    2.890611442640554,
    0.4570457994644658,
    0.3731763325901154,
    0.4570457994644658,
    1.445305721320277,
    0.5900435899266435
]
Cp4 = [
    2.5033429417967046,
    1.7701307697799304,
    0.9461746957575601,
    0.6690465435572892,
    0.10578554691520431,
    0.6690465435572892,
    0.47308734787878004,
    1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh_p(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = Cp0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result +
                Cp1 * y * sh[..., 1] +
                Cp1 * z * sh[..., 2] +
                Cp1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    Cp2[0] * xy * sh[..., 4] +
                    Cp2[1] * yz * sh[..., 5] +
                    Cp2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    Cp2[3] * xz * sh[..., 7] +
                    Cp2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                Cp3[0] * y * (3 * xx - yy) * sh[..., 9] +
                Cp3[1] * xy * z * sh[..., 10] +
                Cp3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                Cp3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                Cp3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                Cp3[5] * z * (xx - yy) * sh[..., 14] +
                Cp3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            Cp4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            Cp4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            Cp4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            Cp4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            Cp4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            Cp4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            Cp4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            Cp4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


CTs_2=torch.tensor([ 0.8862269254527579,
    # L1, m-1
    1.0233267079464883,
    # L1, m
    1.0233267079464883,
    1.0233267079464883,
    # L2, m-2
    0.8580855308097834,
    # N2_1 (y*z)
    0.8580855308097834,
    # N2_0 (2*z^2 - x^2 - y^2))
    0.24770795610037571,
    # N2_1 (z*x)
    0.8580855308097834,
    # N2_2 (x^2-y^2)
    0.8580855308097834]).float()

def eval_sh_v2(deg, sh, dirs):
    assert deg==2
    sh_v=sh.permute((0,2,1))
    result = torch.stack([
                dirs[ 0,...] * 0. + 1,
                # N1 - y
                dirs[ 1,...],
                # N1 - z
                dirs[ 2,...],
                # N1 - x
                dirs[ 0,...],
                # N2_1 - x*y
                dirs[ 0,...] * dirs[ 1,...],
                # N2_1 - y*z
                dirs[ 1,...] * dirs[ 2,...],
                # N2_0 - 2*(z^2-x^2-y^2)
                (2 *dirs[ 2,...] ** 2 - dirs[ 0,...] ** 2 - dirs[ 1,...] ** 2),
                # N2_1 (z*x)
                dirs[ 2,...] * dirs[ 0,...],
                # N2_2 (x^2-y^2)
                dirs[ 0,...] ** 2 - dirs[ 1,...] ** 2
            ], 0)  # [ 9, h, w]
    result= result*CTs_2[:,None,None].to(result.device)# [9, h, w]  sh_v[bz,9,3]
    shading=torch.sum(sh_v[:,:,:,None,None]*result[None,:,None,:,:], 1) # [bz, 9, 3, h, w]
    return shading.squeeze()

CTs_3=torch.tensor([ 0.28209479177387814,#1
    # L1, m-1
    0.4886025119029199,#y
    # L1, m
    0.4886025119029199,#x
    0.4886025119029199,#z
    # L2, m-2 (x*y)
    1.0925484305920792,
    # N2_1 (y*z)
    1.0925484305920792,
    # N2_0 (2*z^2 - x^2-y^2) (3*z^2 - 1)
    0.31539156525251999,
    # N2_1 (z*x)
    1.0925484305920792,
    # N2_2 (x^2-y^2)
    0.5462742152960396]).float()

def eval_sh_v3(deg, sh, dirs):
    assert deg==2
    sh_v=sh.permute((0,2,1))
    result = torch.stack([
                dirs[ 0,...] * 0. + 1,
                # N1 - y
                dirs[ 1,...],
                # N1 - z
                dirs[ 2,...],
                # N1 - x
                dirs[ 0,...],
                # N2_1 - x*y
                dirs[ 0,...] * dirs[ 1,...],
                # N2_1 - y*z
                dirs[ 1,...] * dirs[ 2,...],
                # N2_0 - (3*z^2-1)
                 (3*dirs[ 2,...]**2-1),
                # N2_1 (z*x)
                dirs[ 2,...] * dirs[ 0,...],
                # N2_2 (x^2-y^2)
                dirs[ 0,...] ** 2 - dirs[ 1,...] ** 2
            ], 0)  # [ 9, h, w]
    result= result*CTs_3[:,None,None].to(result.device)# [9, h, w]  sh_v[bz,9,3]
    shading=torch.sum(sh_v[:,:,:,None,None]*result[None,:,None,:,:], 1) # [bz, 9, 3, h, w]
    return shading.squeeze()

def sinc(x):
    """Supporting sinc function
    """
    output = np.sin(x)/x
    output[np.isnan(output)] = 1.
    return output
# perform spherical harmonics projection based on Cartesian Coords SH basis(CTs_3)
def SH_proj_v2(func,coords,width):
	# func and coords have shape (npix, 3[rgb]/[xyz])
	c1 = 0.8862269254527579/np.pi
	c2 = 1.0233267079464883/(2 * np.pi / 3.)
	c3 = 0.8580855308097834 /(np.pi / 4.)
	c4 = 0.24770795610037571/(np.pi / 4.)
	c5 = 0.8580855308097834/(np.pi /2.)

	x = coords[:,0,np.newaxis]
	y = coords[:,1,np.newaxis]
	z = coords[:,2,np.newaxis]
	theta = np.arccos(z)
	domega = 4*np.pi**2/width**2 *np.sinc(theta/np.pi)# sinc(theta)
    
	coeffs = []
	coeffs.append(np.sum(func * c1 * domega, axis=0))
	coeffs.append(np.sum(func * c2*y * domega, axis=0))
	coeffs.append(np.sum(func * c2*z * domega, axis=0))
	coeffs.append(np.sum(func * c2*x * domega, axis=0))
	coeffs.append(np.sum(func * c3*x*y * domega, axis=0))
	coeffs.append(np.sum(func * c3*y*z * domega, axis=0))
	coeffs.append(np.sum(func * c4*(2*z*z-x*x-y*y) * domega, axis=0))#2*(z^2 - x^2 - y^2)
	coeffs.append(np.sum(func * c3*x*z * domega, axis=0))
	coeffs.append(np.sum(func * c5*(x*x-y*y) * domega, axis=0))

	coeffs = np.stack(coeffs,axis=0)
	return coeffs    

def SH_proj_v3(func,coords,width):
	# func and coords have shape (npix, 3[rgb]/[xyz])
	c1 = 0.28209479177387814
	c2 = 0.4886025119029199
	c3 = 1.0925484305920792
	c4 = 0.31539156525251999
	c5 = 0.5462742152960396

	x = coords[:,0,np.newaxis]
	y = coords[:,1,np.newaxis]
	z = coords[:,2,np.newaxis]
	theta = np.arccos(z)
	domega = 4*np.pi**2/width**2 * np.sinc(theta/np.pi)#sinc(theta)

	coeffs = []
	coeffs.append(np.sum(func * c1 * domega, axis=0))
	coeffs.append(np.sum(func * c2*y * domega, axis=0))
	coeffs.append(np.sum(func * c2*z * domega, axis=0))
	coeffs.append(np.sum(func * c2*x * domega, axis=0))
	coeffs.append(np.sum(func * c3*x*y * domega, axis=0))
	coeffs.append(np.sum(func * c3*y*z * domega, axis=0))
	coeffs.append(np.sum(func * c4*(3*z*z-1) * domega, axis=0))
	coeffs.append(np.sum(func * c3*x*z * domega, axis=0))
	coeffs.append(np.sum(func * c5*(x*x-y*y) * domega, axis=0))

	coeffs = np.stack(coeffs,axis=0)
	return coeffs    

import pyshtools as pysh
def rotateSH(sh_input, rot_angle, rot_axis):
    ls = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])
    ms = np.array([0, -1, 0, 1, -2, -1, 0, 1, 2])
    mneg_mask = (ms < 0).astype(np.int)

    # (csphase=1 => exclude)
    __OUR_SH_CSPHASE = 1
    __OUR_SH_NORM = 'ortho'

    def _to_pysh(oldsh):
        # old: [l0,m0; l1,m-1; l1,m0; l1,m1; l2,m-2, l2,m-1, l2,m0, l2,m1, l2,m2]
        # newsh = np.zeros(2, 3, 3)

        newsh = []
        for c in range(3):
            curr_sh = pysh.SHCoeffs.from_zeros(lmax=2, normalization=__OUR_SH_NORM, csphase=__OUR_SH_CSPHASE)
            values = oldsh[:, c]
            # print("&&&")
            # print(values)
            curr_sh.set_coeffs(values, ls, ms)
            # print(curr_sh.coeffs)
            # print("&&&")
            # L0,m0
            # curr_sh.set_coeffs( oldsh[0, c],  )

            newsh.append(curr_sh)

        return newsh

    def _from_pysh(shlist):

        out = []
        for c in range(3):
            # convert back to orthonormalized
            # _sh = shlist[c].convert('ortho')
            _sh_orth = shlist[c].to_array(normalization=__OUR_SH_NORM, csphase=__OUR_SH_CSPHASE)
            # print("///")
            # print(_sh_orth)
            _sh_1D = _sh_orth[mneg_mask, ls, np.abs(ms)].ravel()
            # print(_sh_1D)
            out.append(_sh_1D[:, None])

        out = np.concatenate(out, axis=1)
        return out

    # convert to pysh format
    _sh = _to_pysh(sh_input)

    # rotate each channel
    rotM = pysh.rotate.djpi2(2)
    rot = [0, 0, 0]
    rot[rot_axis] = rot_angle * np.pi / 180.
    rot_sh = []

    for c in range(3):
        _sh_for_rot = _sh[c].convert('4pi', csphase=1)  # needed for rotation
        _sh_rot = pysh.rotate.SHRotateRealCoef(_sh_for_rot.coeffs, rot, rotM)
        _sh_rot = pysh.SHCoeffs.from_array(_sh_rot, normalization='4pi', csphase=1)
        rot_sh.append(_sh_rot)

    rot_sh = _from_pysh(rot_sh)
    return rot_sh#torch.as_tensor(rot_sh)