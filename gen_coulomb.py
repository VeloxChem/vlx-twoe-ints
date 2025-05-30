#  This file is part of vlx-twoe-ints.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  Copyright 2024-2025 Xin Li
#
#  Redistribution and use in source and binary forms, with or without modification,
#  are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software without
#     specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from eriwriter import write_eri_code
import sys

angmoms = 'spd'

ab_list = []
header_only = False

for arg in sys.argv[1:]:
    if arg == 'header':
        header_only = True
    else:
        assert len(arg) == 2
        assert arg[0] in angmoms
        assert arg[1] in angmoms
        ab_list.append(arg)

cd_list = []
for idx_c, c in enumerate(angmoms):
    for idx_d, d in enumerate(angmoms):
        if idx_c <= idx_d:
            cd_list.append(f'{c}{d}')

eri_list = []
for ab in ab_list:
    for cd in cd_list:
        eri_list.append(f'{ab}{cd}')

for eri_abcd in eri_list:

    assert len(eri_abcd) == 4

    a,b,c,d = eri_abcd.lower()
    assert a in angmoms
    assert b in angmoms
    assert c in angmoms
    assert d in angmoms

    has_d_orbital = ('d' in eri_abcd.lower())

    denom = { 's': '', 'p': ' / 3', 'd': ' / 6' }
    angmom = { 's': 0, 'p': 1, 'd': 2 }
    angmom_sum = angmom[a] + angmom[b] + angmom[c] + angmom[d]

    print('__global__ void __launch_bounds__(TILE_SIZE_J)')
    print(f'computeCoulombFock{eri_abcd.upper()}(double*         mat_J,')
    for x in angmoms:
        if x in eri_abcd.lower():
            print(f'                       const double*   {x}_prim_info,')
            print(f'                       const uint32_t  {x}_prim_count,')
    print(f'                       const double*   {c}{d}_mat_D,')
    print(f'                       const double*   {a}{b}_mat_Q_local,')
    print(f'                       const double*   {c}{d}_mat_Q,')
    print(f'                       const uint32_t* {a}{b}_first_inds_local,')
    print(f'                       const uint32_t* {a}{b}_second_inds_local,')
    print(f'                       const double*   {a}{b}_pair_data_local,')
    print(f'                       const uint32_t  {a}{b}_prim_pair_count_local,')
    print(f'                       const uint32_t* {c}{d}_first_inds,')
    print(f'                       const uint32_t* {c}{d}_second_inds,')
    print(f'                       const double*   {c}{d}_pair_data,')
    print(f'                       const uint32_t  {c}{d}_prim_pair_count,')
    print(f'                       const double*   boys_func_table,')
    print(f'                       const double*   boys_func_ft,')
    print(f'                       const double    eri_threshold)', end='')

    if header_only:
        print(';')
        print()
        continue
    else:
        print()

    print('{')
    print('    // each thread row scans over [ij|??] and sum up to a primitive J matrix element')
    print('    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015')
    print()
    print('    __shared__ double   ERIs[TILE_DIM][TILE_DIM + 1];')

    if has_d_orbital:
        print('    __shared__ uint32_t d_cart_inds[6][2];')

    if angmom_sum >= 2:
        print('    __shared__ double   delta[3][3];')

    print()
    print('    const uint32_t ij = blockDim.x * blockIdx.x + threadIdx.x;')
    print()
    print('    double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;')

    uint_vars = ['i', 'j']
    double_vars = []
    if a == 'p' or a == 'd':
        uint_vars.append('a0')
        double_vars.append('PA_0')
    if a == 'd':
        uint_vars.append('a1')
        double_vars.append('PA_1')
    if b == 'p' or b == 'd':
        uint_vars.append('b0')
        double_vars.append('PB_0')
    if b == 'd':
        uint_vars.append('b1')
        double_vars.append('PB_1')
    if double_vars:
        print(f'    double {", ".join(double_vars)};')
    print(f'    uint32_t {", ".join(uint_vars)};')
    print()

    if has_d_orbital or angmom_sum >= 2:
        print('    if ((threadIdx.y == 0) && (threadIdx.x == 0))')
        print('    {')

    if has_d_orbital:
        print('        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;')
        print('        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;')
        print('        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;')
        print('        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;')
        print('        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;')
        print('        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;')
        print()

    if angmom_sum >= 2:
        print('        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;')
        print('        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;')
        print('        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;')
        print()

    if has_d_orbital or angmom_sum >= 2:
        print('    }')
        print()

    print('    ERIs[threadIdx.y][threadIdx.x] = 0.0;')
    print()
    print('    __syncthreads();')
    print()
    print(f'    if (ij < {a}{b}_prim_pair_count_local)')
    print(f'    {{')
    print(f'        i = {a}{b}_first_inds_local[ij];')
    print(f'        j = {a}{b}_second_inds_local[ij];')
    print()
    print(f'        a_i = {a}_prim_info[i{denom[a]} + {a}_prim_count * 0];')
    print()
    print(f'        r_i[0] = {a}_prim_info[i{denom[a]} + {a}_prim_count * 2];')
    print(f'        r_i[1] = {a}_prim_info[i{denom[a]} + {a}_prim_count * 3];')
    print(f'        r_i[2] = {a}_prim_info[i{denom[a]} + {a}_prim_count * 4];')
    print()
    print(f'        a_j = {b}_prim_info[j{denom[b]} + {b}_prim_count * 0];')
    print()
    print(f'        r_j[0] = {b}_prim_info[j{denom[b]} + {b}_prim_count * 2];')
    print(f'        r_j[1] = {b}_prim_info[j{denom[b]} + {b}_prim_count * 3];')
    print(f'        r_j[2] = {b}_prim_info[j{denom[b]} + {b}_prim_count * 4];')
    print()
    print(f'        S1 = a_i + a_j;')
    print(f'        inv_S1 = 1.0 / S1;')
    print()
    print(f'        S_ij_00 = {a}{b}_pair_data_local[ij];')
    print()
    for x,m,n in zip([a,b], 'ab', 'ij'):
        if x == 'p':
            print(f'        {m}0 = {n} % 3;')
        elif x == 'd':
            print(f'        {m}0 = d_cart_inds[{n} % 6][0];')
            print(f'        {m}1 = d_cart_inds[{n} % 6][1];')
    print()
    if a == 'p' or a == 'd':
        print(f'        PA_0 = (a_j  * inv_S1) * (r_j[a0] - r_i[a0]);')
    if a == 'd':
        print(f'        PA_1 = (a_j  * inv_S1) * (r_j[a1] - r_i[a1]);')
    if b == 'p' or b == 'd':
        print(f'        PB_0 = (-a_i * inv_S1) * (r_j[b0] - r_i[b0]);')
    if b == 'd':
        print(f'        PB_1 = (-a_i * inv_S1) * (r_j[b1] - r_i[b1]);')
    print()
    print(f'    }}')
    print()
    print(f'    for (uint32_t m = 0; m < ({c}{d}_prim_pair_count + TILE_DIM - 1) / TILE_DIM; m++)')
    print('    {')
    print('        const uint32_t kl = m * TILE_DIM + threadIdx.y;')
    print()
    print(f'        if ((ij >= {a}{b}_prim_pair_count_local) || (kl >= {c}{d}_prim_pair_count) || (fabs({a}{b}_mat_Q_local[ij] * {c}{d}_mat_Q[kl] * {c}{d}_mat_D[kl]) <= eri_threshold))')
    print('        {')
    print('            break;')
    print('        }')
    print()
    print(f'        const auto k = {c}{d}_first_inds[kl];')
    print(f'        const auto l = {c}{d}_second_inds[kl];')
    print()
    print(f'        const auto a_k = {c}_prim_info[k{denom[c]} + {c}_prim_count * 0];')
    print()
    print(f'        const double r_k[3] = {{{c}_prim_info[k{denom[c]} + {c}_prim_count * 2],')
    print(f'                               {c}_prim_info[k{denom[c]} + {c}_prim_count * 3],')
    print(f'                               {c}_prim_info[k{denom[c]} + {c}_prim_count * 4]}};')
    print()
    print(f'        const auto a_l = {d}_prim_info[l{denom[d]} + {d}_prim_count * 0];')
    print()
    print(f'        const double r_l[3] = {{{d}_prim_info[l{denom[d]} + {d}_prim_count * 2],')
    print(f'                               {d}_prim_info[l{denom[d]} + {d}_prim_count * 3],')
    print(f'                               {d}_prim_info[l{denom[d]} + {d}_prim_count * 4]}};')
    print()
    print(f'        const auto S_kl_00 = {c}{d}_pair_data[kl];')
    print()
    for x,m,n in zip([c,d], 'cd', 'kl'):
        if x == 'p':
            print(f'        const auto {m}0 = {n} % 3;')
        elif x == 'd':
            print(f'        const auto {m}0 = d_cart_inds[{n} % 6][0];')
            print(f'        const auto {m}1 = d_cart_inds[{n} % 6][1];')
    print()

    if a in 'sp' and b in 'sp' and c in 'sp' and d in 'sp':
        print('        // Electron. J. Theor. Chem., Vol. 2, 66-70 (1997)')

    print(f'        // J. Chem. Phys. 84, 3963-3974 (1986)')
    print()
    print(f'        const auto S2 = a_k + a_l;')
    print()
    print(f'        const auto inv_S2 = 1.0 / S2;')
    print(f'        const auto inv_S4 = 1.0 / (S1 + S2);')
    print()
    print(f'        const double PQ[3] = {{(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,')
    print(f'                              (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,')
    print(f'                              (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1}};')
    print()
    print(f'        const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];')
    print()
    print(f'        const auto Lambda = sqrt(4.0 * S1 * S2 * MATH_CONST_INV_PI * inv_S4);')
    print()
    print(f'        double F{angmom_sum}_t[{angmom_sum+1}];')
    print()
    print(f'        gpu::computeBoysFunction(F{angmom_sum}_t, S1 * S2 * inv_S4 * r2_PQ, {angmom_sum}, boys_func_table, boys_func_ft);')
    print()
    if c == 'p' or c == 'd':
        print(f'        const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);')
    if c == 'd':
        print(f'        const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1];')
    if d == 'p' or d == 'd':
        print(f'        const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);')
    if d == 'd':
        print(f'        const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);')
    print()
    print(f'        const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (')

    print(write_eri_code(eri_abcd.lower(), indent=16))

    print(f'                );')
    print()
    print(f'        // NOTE: doubling for off-diagonal elements of D due to k<=>l symmetry')
    if c != d:
        print(f'        ERIs[threadIdx.y][threadIdx.x] += eri_ijkl * {c}{d}_mat_D[kl] * 2.0;')
    else:
        print(f'        //       (static_cast<double>(k != l) + 1.0) == (k == l ? 1.0 : 2.0)')
        print(f'        ERIs[threadIdx.y][threadIdx.x] += eri_ijkl * {c}{d}_mat_D[kl] * (static_cast<double>(k != l) + 1.0);')
    print()
    print('    }')
    print()
    print('    __syncthreads();')
    print()
    print(f'    if ((threadIdx.y == 0) && (ij < {a}{b}_prim_pair_count_local))')
    print('    {')
    print('        double J_ij = 0.0;')
    print()
    print('        for (uint32_t n = 0; n < TILE_DIM; n++)')
    print('        {')
    print('            J_ij += ERIs[n][threadIdx.x];')
    print('        }')
    print()
    print('        mat_J[ij] += J_ij;')
    print('    }')
    print('}')
    print()
