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

ac_list = []
header_only = False

for arg in sys.argv[1:]:
    if arg == 'header':
        header_only = True
    else:
        assert len(arg) == 2
        assert arg[0] in angmoms
        assert arg[1] in angmoms
        ac_list.append(arg)

bd_list = []
for b in angmoms:
    for d in angmoms:
        bd_list.append(f'{b}{d}')

eri_list = []
for a, c in ac_list:
    for b, d in bd_list:
        eri_list.append(f'{a}{b}{c}{d}')

for eri_abcd in eri_list:

    assert len(eri_abcd) == 4

    a,b,c,d = eri_abcd.lower()
    assert a in angmoms
    assert b in angmoms
    assert c in angmoms
    assert d in angmoms

    has_d_orbital = ('d' in eri_abcd.lower())

    multi = { 's': '', 'p': ' * 3', 'd': ' * 6' }
    denom = { 's': '', 'p': ' / 3', 'd': ' / 6' }
    rem = { 's': '', 'p': ' % 3', 'd': ' % 6' }
    angmom = { 's': 0, 'p': 1, 'd': 2 }
    angmom_sum = angmom[a] + angmom[b] + angmom[c] + angmom[d]

    print('__global__ void __launch_bounds__(TILE_SIZE_K)')
    print(f'computeExchangeFock{eri_abcd.upper()}(double*         mat_K,')

    print(f'                        const uint32_t* pair_inds_i_for_K_{a}{c},')
    print(f'                        const uint32_t* pair_inds_k_for_K_{a}{c},')
    print(f'                        const uint32_t  pair_inds_count_for_K_{a}{c},')

    for x in angmoms:
        if x in eri_abcd.lower():
            print(f'                        const double*   {x}_prim_info,')
            print(f'                        const uint32_t* {x}_prim_aoinds,')
            print(f'                        const uint32_t  {x}_prim_count,')

    print(f'                        const double    {b}{d}_max_D,')
    print(f'                        const double*   mat_D_full_AO,')
    print(f'                        const uint32_t  naos,')

    print(f'                        const double*   Q_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        print(f'                        const double*   Q_K_{c}{d},')

    print(f'                        const uint32_t* D_inds_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        print(f'                        const uint32_t* D_inds_K_{c}{d},')

    print(f'                        const uint32_t* pair_displs_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        print(f'                        const uint32_t* pair_displs_K_{c}{d},')

    print(f'                        const uint32_t* pair_counts_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        print(f'                        const uint32_t* pair_counts_K_{c}{d},')

    print(f'                        const double*   pair_data_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        print(f'                        const double*   pair_data_K_{c}{d},')

    print('                        const double*   boys_func_table,')
    print('                        const double*   boys_func_ft,')
    print('                        const double    omega,')
    print('                        const double    eri_threshold)', end='')

    if header_only:
        print(';')
        print('')
        continue
    else:
        print('')

    print('{')
    print('    // each thread block scans over [i?|k?] and sum up to a primitive K matrix element')
    print('    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015')
    print('')
    print('    __shared__ double   ERIs[TILE_DIM_Y_K][TILE_DIM_X_K + 1];')
    print('    __shared__ uint32_t i, k, count_i, count_k, displ_i, displ_k;')
    print('    __shared__ double   a_i, r_i[3], a_k, r_k[3];')

    if has_d_orbital:
        print('    __shared__ uint32_t d_cart_inds[6][2];')

    if angmom_sum >= 2:
        print('    __shared__ double   delta[3][3];')

    print('')
    print('    const uint32_t ik = blockIdx.x;')
    print('')
    print(f'    // we make sure that ik < pair_inds_count_for_K_{a}{c} when calling the kernel')
    print('')
    print('    if ((threadIdx.y == 0) && (threadIdx.x == 0))')
    print('    {')

    if has_d_orbital:
        print('        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;')
        print('        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;')
        print('        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;')
        print('        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;')
        print('        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;')
        print('        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;')
        print('')

    if angmom_sum >= 2:
        print('        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;')
        print('        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;')
        print('        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;')
        print('')

    print(f'        i = pair_inds_i_for_K_{a}{c}[ik];')
    print(f'        k = pair_inds_k_for_K_{a}{c}[ik];')
    print('')
    print(f'        count_i = pair_counts_K_{a}{b}[i];')
    print(f'        count_k = pair_counts_K_{c}{d}[k];')
    print('')
    print(f'        displ_i = pair_displs_K_{a}{b}[i];')
    print(f'        displ_k = pair_displs_K_{c}{d}[k];')
    print('')
    print(f'        a_i = {a}_prim_info[i{denom[a]} + {a}_prim_count * 0];')
    print('')
    print(f'        r_i[0] = {a}_prim_info[i{denom[a]} + {a}_prim_count * 2];')
    print(f'        r_i[1] = {a}_prim_info[i{denom[a]} + {a}_prim_count * 3];')
    print(f'        r_i[2] = {a}_prim_info[i{denom[a]} + {a}_prim_count * 4];')
    print('')
    print(f'        a_k = {c}_prim_info[k{denom[c]} + {c}_prim_count * 0];')
    print('')
    print(f'        r_k[0] = {c}_prim_info[k{denom[c]} + {c}_prim_count * 2];')
    print(f'        r_k[1] = {c}_prim_info[k{denom[c]} + {c}_prim_count * 3];')
    print(f'        r_k[2] = {c}_prim_info[k{denom[c]} + {c}_prim_count * 4];')

    print('    }')
    print('')
    print('    ERIs[threadIdx.y][threadIdx.x] = 0.0;')
    print('')
    print('    __syncthreads();')
    print('')

    print(f'    for (uint32_t m = 0; m < (count_i + TILE_DIM_Y_K - 1) / TILE_DIM_Y_K; m++)')
    print(f'    {{')
    print(f'        const uint32_t j = m * TILE_DIM_Y_K + threadIdx.y;')
    print('')
    print(f'        // sync threads before starting a new scan')
    print(f'        __syncthreads();')
    print('')
    print(f'        double Q_ij, a_j, r_j[3], S_ij_00, S1, inv_S1;')
    print(f'        uint32_t j_prim, j_cgto;')
    print('')
    print(f'        if (j < count_i)')
    print(f'        {{')
    print(f'            Q_ij   = Q_K_{a}{b}[displ_i + j];')
    print('')
    print(f'            j_prim = D_inds_K_{a}{b}[displ_i + j];')
    print('')

    if b == 's':
        print(f'            j_cgto = {b}_prim_aoinds[j_prim{denom[b]}];')
    else:
        print(f'            j_cgto = {b}_prim_aoinds[(j_prim{denom[b]}) + {b}_prim_count * (j_prim{rem[b]})];')
    print('')

    print(f'            a_j = {b}_prim_info[j_prim{denom[b]} + {b}_prim_count * 0];')
    print('')
    print(f'            r_j[0] = {b}_prim_info[j_prim{denom[b]} + {b}_prim_count * 2];')
    print(f'            r_j[1] = {b}_prim_info[j_prim{denom[b]} + {b}_prim_count * 3];')
    print(f'            r_j[2] = {b}_prim_info[j_prim{denom[b]} + {b}_prim_count * 4];')
    print('')
    print(f'            S1 = a_i + a_j;')
    print(f'            inv_S1 = 1.0 / S1;')
    print('')
    print(f'            S_ij_00 = pair_data_K_{a}{b}[displ_i + j];')
    print(f'        }}')
    print('')
    print(f'        for (uint32_t n = 0; n < (count_k + TILE_DIM_X_K - 1) / TILE_DIM_X_K; n++)')
    print('        {')
    print('            const uint32_t l = n * TILE_DIM_X_K + threadIdx.x;')
    print('')
    print(f'            // Q_kl == Q_K_{c}{d}[displ_k + l]')
    print(f'            if ((j >= count_i) || (l >= count_k) || (fabs(Q_ij * Q_K_{c}{d}[displ_k + l] * {b}{d}_max_D) <= eri_threshold))')
    print('            {')
    print('                break;')
    print('            }')
    print('')
    print(f'            const auto Q_kl = Q_K_{c}{d}[displ_k + l];')
    print('')
    print(f'            const auto l_prim = D_inds_K_{c}{d}[displ_k + l];')
    print('')

    if d == 's':
        print(f'            const auto l_cgto = {d}_prim_aoinds[l_prim{denom[d]}];')
    else:
        print(f'            const auto l_cgto = {d}_prim_aoinds[(l_prim{denom[d]}) + {d}_prim_count * (l_prim{rem[d]})];')
    print('')

    print(f'            const auto a_l = {d}_prim_info[l_prim{denom[d]} + {d}_prim_count * 0];')
    print('')
    print(f'            const double r_l[3] = {{{d}_prim_info[l_prim{denom[d]} + {d}_prim_count * 2],')
    print(f'                                   {d}_prim_info[l_prim{denom[d]} + {d}_prim_count * 3],')
    print(f'                                   {d}_prim_info[l_prim{denom[d]} + {d}_prim_count * 4]}};')
    print('')
    print(f'            const auto S_kl_00 = pair_data_K_{c}{d}[displ_k + l];')
    print('')

    for x,m,n in zip([a,b,c,d], 'abcd', ['i', 'j_prim', 'k', 'l_prim']):
        if x == 'p':
            print(f'            const auto {m}0 = {n} % 3;')
        elif x == 'd':
            print(f'            const auto {m}0 = d_cart_inds[{n} % 6][0];')
            print(f'            const auto {m}1 = d_cart_inds[{n} % 6][1];')
    print('')

    if a in 'sp' and b in 'sp' and c in 'sp' and d in 'sp':
        print(f'            // Electron. J. Theor. Chem., Vol. 2, 66-70 (1997)')

    print(f'            // J. Chem. Phys. 84, 3963-3974 (1986)')
    print('')
    print(f'            const auto S2 = a_k + a_l;')
    print('')
    print(f'            const auto inv_S2 = 1.0 / S2;')
    print(f'            const auto inv_S4 = 1.0 / (S1 + S2);')
    print('')
    print(f'            const double PQ[3] = {{(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,')
    print(f'                                  (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,')
    print(f'                                  (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1}};')
    print('')
    print(f'            const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];')
    print('')
    print(f'            const auto rho = S1 * S2 * inv_S4;')
    print('')
    print(f'            double d2 = 1.0;')
    print('')
    print(f'            if (omega != 0.0) d2 = omega * omega / (rho + omega * omega);')
    print('')
    print(f'            const auto Lambda = sqrt(4.0 * rho * d2 * MATH_CONST_INV_PI);')
    print('')
    print(f'            double F{angmom_sum}_t[{angmom_sum+1}];')
    print('')
    print(f'            gpu::computeBoysFunction(F{angmom_sum}_t, rho * d2 * r2_PQ, {angmom_sum}, boys_func_table, boys_func_ft);')
    print('')

    if angmom_sum > 0:
        print(f'            if (omega != 0.0)')
        print(f'            {{')
    for am in range(1, angmom_sum+1):
        print(f'                F{angmom_sum}_t[{am}] *= ' + ' * '.join(['d2'] * am) + ';')
    if angmom_sum > 0:
        print(f'            }}')
        print('')

    if a == 'p' or a == 'd':
        print(f'            const auto PA_0 = (a_j  * inv_S1) * (r_j[a0] - r_i[a0]);')
    if a == 'd':
        print(f'            const auto PA_1 = (a_j  * inv_S1) * (r_j[a1] - r_i[a1]);')
    if b == 'p' or b == 'd':
        print(f'            const auto PB_0 = (-a_i * inv_S1) * (r_j[b0] - r_i[b0]);')
    if b == 'd':
        print(f'            const auto PB_1 = (-a_i * inv_S1) * (r_j[b1] - r_i[b1]);')
    if c == 'p' or c == 'd':
        print(f'            const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);')
    if c == 'd':
        print(f'            const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);')
    if d == 'p' or d == 'd':
        print(f'            const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);')
    if d == 'd':
        print(f'            const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);')
    print('')
    print(f'            const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (')

    print(write_eri_code(eri_abcd.lower(), indent=20))

    print('                    );')
    print('')
    print('            ERIs[threadIdx.y][threadIdx.x] += eri_ijkl * mat_D_full_AO[j_cgto * naos + l_cgto];')
    print('        }')
    print('    }')
    print('')
    print('    __syncthreads();')
    print('')
    print('    if ((threadIdx.y == 0) && (threadIdx.x == 0))')
    print('    {')
    print('        double K_ik = 0.0;')
    print('')
    print('        for (uint32_t y = 0; y < TILE_DIM_Y_K; y++)')
    print('        {')
    print('            for (uint32_t x = 0; x < TILE_DIM_X_K; x++)')
    print('            {')
    print('                K_ik += ERIs[y][x];')
    print('            }')
    print('        }')
    print('')
    print('        mat_K[ik] += K_ik;')
    print('    }')
    print('}')
    print('')
