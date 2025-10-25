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


def write_exchange_fock_code(eri_name, header_only=False):

    assert isinstance(eri_name, str)

    assert len(eri_name) == 4

    angmoms = 'spd'
    for x in eri_name:
        assert x in angmoms

    a,b,c,d = eri_name.lower()

    has_d_orbital = ('d' in eri_name.lower())

    # multi = { 's': '', 'p': ' * 3', 'd': ' * 6' }
    denom = { 's': '', 'p': ' / 3', 'd': ' / 6' }
    rem = { 's': '', 'p': ' % 3', 'd': ' % 6' }
    angmom = { 's': 0, 'p': 1, 'd': 2 }
    angmom_sum = angmom[a] + angmom[b] + angmom[c] + angmom[d]

    code = []

    code.append('__global__ void __launch_bounds__(TILE_SIZE_K)')
    code.append(f'computeExchangeFock{eri_name.upper()}(double*         mat_K,')

    code.append(f'                        const uint32_t* pair_inds_i_for_K_{a}{c},')
    code.append(f'                        const uint32_t* pair_inds_k_for_K_{a}{c},')
    code.append(f'                        const uint32_t  pair_inds_count_for_K_{a}{c},')

    for x in angmoms:
        if x in eri_name.lower():
            code.append(f'                        const double*   {x}_prim_info,')
            code.append(f'                        const uint32_t* {x}_prim_aoinds,')
            code.append(f'                        const uint32_t  {x}_prim_count,')

    code.append(f'                        const double    {b}{d}_max_D,')
    code.append(f'                        const double*   mat_D_full_AO,')
    code.append(f'                        const uint32_t  naos,')

    code.append(f'                        const double*   Q_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        code.append(f'                        const double*   Q_K_{c}{d},')

    code.append(f'                        const uint32_t* D_inds_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        code.append(f'                        const uint32_t* D_inds_K_{c}{d},')

    code.append(f'                        const uint32_t* pair_displs_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        code.append(f'                        const uint32_t* pair_displs_K_{c}{d},')

    code.append(f'                        const uint32_t* pair_counts_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        code.append(f'                        const uint32_t* pair_counts_K_{c}{d},')

    code.append(f'                        const double*   pair_data_K_{a}{b},')
    if f'{c}{d}' != f'{a}{b}':
        code.append(f'                        const double*   pair_data_K_{c}{d},')

    code.append('                        const double*   boys_func_table,')
    code.append('                        const double*   boys_func_ft,')
    code.append('                        const double    omega,')

    code_str = '                        const double    eri_threshold)'

    if header_only:
        code.append(code_str + ';')
        code.append('')

        return code

    code.append(code_str)

    code.append('{')
    code.append('    // each thread block scans over [i?|k?] and sum up to a primitive K matrix element')
    code.append('    // J. Chem. Theory Comput. 2009, 5, 4, 1004-1015')
    code.append('')
    code.append('    __shared__ double   ERIs[TILE_DIM_Y_K][TILE_DIM_X_K + 1];')
    code.append('    __shared__ uint32_t i, k, count_i, count_k, displ_i, displ_k;')
    code.append('    __shared__ double   a_i, r_i[3], a_k, r_k[3];')

    if has_d_orbital:
        code.append('    __shared__ uint32_t d_cart_inds[6][2];')

    if angmom_sum >= 2:
        code.append('    __shared__ double   delta[3][3];')

    code.append('')
    code.append('    const uint32_t ik = blockIdx.x;')
    code.append('')
    code.append(f'    // we make sure that ik < pair_inds_count_for_K_{a}{c} when calling the kernel')
    code.append('')
    code.append('    if ((threadIdx.y == 0) && (threadIdx.x == 0))')
    code.append('    {')

    if has_d_orbital:
        code.append('        d_cart_inds[0][0] = 0; d_cart_inds[0][1] = 0;')
        code.append('        d_cart_inds[1][0] = 0; d_cart_inds[1][1] = 1;')
        code.append('        d_cart_inds[2][0] = 0; d_cart_inds[2][1] = 2;')
        code.append('        d_cart_inds[3][0] = 1; d_cart_inds[3][1] = 1;')
        code.append('        d_cart_inds[4][0] = 1; d_cart_inds[4][1] = 2;')
        code.append('        d_cart_inds[5][0] = 2; d_cart_inds[5][1] = 2;')
        code.append('')

    if angmom_sum >= 2:
        code.append('        delta[0][0] = 1.0; delta[0][1] = 0.0; delta[0][2] = 0.0;')
        code.append('        delta[1][0] = 0.0; delta[1][1] = 1.0; delta[1][2] = 0.0;')
        code.append('        delta[2][0] = 0.0; delta[2][1] = 0.0; delta[2][2] = 1.0;')
        code.append('')

    code.append(f'        i = pair_inds_i_for_K_{a}{c}[ik];')
    code.append(f'        k = pair_inds_k_for_K_{a}{c}[ik];')
    code.append('')
    code.append(f'        count_i = pair_counts_K_{a}{b}[i];')
    code.append(f'        count_k = pair_counts_K_{c}{d}[k];')
    code.append('')
    code.append(f'        displ_i = pair_displs_K_{a}{b}[i];')
    code.append(f'        displ_k = pair_displs_K_{c}{d}[k];')
    code.append('')
    code.append(f'        a_i = {a}_prim_info[i{denom[a]} + {a}_prim_count * 0];')
    code.append('')
    code.append(f'        r_i[0] = {a}_prim_info[i{denom[a]} + {a}_prim_count * 2];')
    code.append(f'        r_i[1] = {a}_prim_info[i{denom[a]} + {a}_prim_count * 3];')
    code.append(f'        r_i[2] = {a}_prim_info[i{denom[a]} + {a}_prim_count * 4];')
    code.append('')
    code.append(f'        a_k = {c}_prim_info[k{denom[c]} + {c}_prim_count * 0];')
    code.append('')
    code.append(f'        r_k[0] = {c}_prim_info[k{denom[c]} + {c}_prim_count * 2];')
    code.append(f'        r_k[1] = {c}_prim_info[k{denom[c]} + {c}_prim_count * 3];')
    code.append(f'        r_k[2] = {c}_prim_info[k{denom[c]} + {c}_prim_count * 4];')

    code.append('    }')
    code.append('')
    code.append('    ERIs[threadIdx.y][threadIdx.x] = 0.0;')
    code.append('')
    code.append('    __syncthreads();')
    code.append('')

    code.append(f'    for (uint32_t m = 0; m < (count_i + TILE_DIM_Y_K - 1) / TILE_DIM_Y_K; m++)')
    code.append(f'    {{')
    code.append(f'        const uint32_t j = m * TILE_DIM_Y_K + threadIdx.y;')
    code.append('')
    code.append(f'        // sync threads before starting a new scan')
    code.append(f'        __syncthreads();')
    code.append('')
    code.append(f'        double Q_ij, a_j, r_j[3], S_ij_00, S1, inv_S1;')
    code.append(f'        uint32_t j_prim, j_cgto;')
    code.append('')
    code.append(f'        if (j < count_i)')
    code.append(f'        {{')
    code.append(f'            Q_ij   = Q_K_{a}{b}[displ_i + j];')
    code.append('')
    code.append(f'            j_prim = D_inds_K_{a}{b}[displ_i + j];')
    code.append('')

    if b == 's':
        code.append(f'            j_cgto = {b}_prim_aoinds[j_prim{denom[b]}];')
    else:
        code.append(f'            j_cgto = {b}_prim_aoinds[(j_prim{denom[b]}) + {b}_prim_count * (j_prim{rem[b]})];')
    code.append('')

    code.append(f'            a_j = {b}_prim_info[j_prim{denom[b]} + {b}_prim_count * 0];')
    code.append('')
    code.append(f'            r_j[0] = {b}_prim_info[j_prim{denom[b]} + {b}_prim_count * 2];')
    code.append(f'            r_j[1] = {b}_prim_info[j_prim{denom[b]} + {b}_prim_count * 3];')
    code.append(f'            r_j[2] = {b}_prim_info[j_prim{denom[b]} + {b}_prim_count * 4];')
    code.append('')
    code.append(f'            S1 = a_i + a_j;')
    code.append(f'            inv_S1 = 1.0 / S1;')
    code.append('')
    code.append(f'            S_ij_00 = pair_data_K_{a}{b}[displ_i + j];')
    code.append(f'        }}')
    code.append('')
    code.append(f'        for (uint32_t n = 0; n < (count_k + TILE_DIM_X_K - 1) / TILE_DIM_X_K; n++)')
    code.append('        {')
    code.append('            const uint32_t l = n * TILE_DIM_X_K + threadIdx.x;')
    code.append('')
    code.append(f'            // Q_kl == Q_K_{c}{d}[displ_k + l]')
    code.append(f'            if ((j >= count_i) || (l >= count_k) || (fabs(Q_ij * Q_K_{c}{d}[displ_k + l] * {b}{d}_max_D) <= eri_threshold))')
    code.append('            {')
    code.append('                break;')
    code.append('            }')
    code.append('')
    code.append(f'            // const auto Q_kl = Q_K_{c}{d}[displ_k + l];')
    code.append('')
    code.append(f'            const auto l_prim = D_inds_K_{c}{d}[displ_k + l];')
    code.append('')

    if d == 's':
        code.append(f'            const auto l_cgto = {d}_prim_aoinds[l_prim{denom[d]}];')
    else:
        code.append(f'            const auto l_cgto = {d}_prim_aoinds[(l_prim{denom[d]}) + {d}_prim_count * (l_prim{rem[d]})];')
    code.append('')

    code.append(f'            const auto a_l = {d}_prim_info[l_prim{denom[d]} + {d}_prim_count * 0];')
    code.append('')
    code.append(f'            const double r_l[3] = {{{d}_prim_info[l_prim{denom[d]} + {d}_prim_count * 2],')
    code.append(f'                                   {d}_prim_info[l_prim{denom[d]} + {d}_prim_count * 3],')
    code.append(f'                                   {d}_prim_info[l_prim{denom[d]} + {d}_prim_count * 4]}};')
    code.append('')
    code.append(f'            const auto S_kl_00 = pair_data_K_{c}{d}[displ_k + l];')
    code.append('')

    for x,m,n in zip([a,b,c,d], 'abcd', ['i', 'j_prim', 'k', 'l_prim']):
        if x == 'p':
            code.append(f'            const auto {m}0 = {n} % 3;')
        elif x == 'd':
            code.append(f'            const auto {m}0 = d_cart_inds[{n} % 6][0];')
            code.append(f'            const auto {m}1 = d_cart_inds[{n} % 6][1];')
    code.append('')

    if a in 'sp' and b in 'sp' and c in 'sp' and d in 'sp':
        code.append(f'            // Electron. J. Theor. Chem., Vol. 2, 66-70 (1997)')

    code.append(f'            // J. Chem. Phys. 84, 3963-3974 (1986)')
    code.append('')
    code.append(f'            const auto S2 = a_k + a_l;')
    code.append('')
    code.append(f'            const auto inv_S2 = 1.0 / S2;')
    code.append(f'            const auto inv_S4 = 1.0 / (S1 + S2);')
    code.append('')
    code.append(f'            const double PQ[3] = {{(a_k * r_k[0] + a_l * r_l[0]) * inv_S2 - (a_i * r_i[0] + a_j * r_j[0]) * inv_S1,')
    code.append(f'                                  (a_k * r_k[1] + a_l * r_l[1]) * inv_S2 - (a_i * r_i[1] + a_j * r_j[1]) * inv_S1,')
    code.append(f'                                  (a_k * r_k[2] + a_l * r_l[2]) * inv_S2 - (a_i * r_i[2] + a_j * r_j[2]) * inv_S1}};')
    code.append('')
    code.append(f'            const auto r2_PQ = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];')
    code.append('')
    code.append(f'            const auto rho = S1 * S2 * inv_S4;')
    code.append('')
    code.append(f'            double d2 = 1.0;')
    code.append('')
    code.append(f'            if (omega != 0.0) d2 = omega * omega / (rho + omega * omega);')
    code.append('')
    code.append(f'            const auto Lambda = sqrt(4.0 * rho * d2 * MATH_CONST_INV_PI);')
    code.append('')
    code.append(f'            double F{angmom_sum}_t[{angmom_sum+1}];')
    code.append('')
    code.append(f'            gpu::computeBoysFunction(F{angmom_sum}_t, rho * d2 * r2_PQ, {angmom_sum}, boys_func_table, boys_func_ft);')
    code.append('')

    if angmom_sum > 0:
        code.append(f'            if (omega != 0.0)')
        code.append(f'            {{')
    for am in range(1, angmom_sum+1):
        code.append(f'                F{angmom_sum}_t[{am}] *= ' + ' * '.join(['d2'] * am) + ';')
    if angmom_sum > 0:
        code.append(f'            }}')
        code.append('')

    if a == 'p' or a == 'd':
        code.append(f'            const auto PA_0 = (a_j  * inv_S1) * (r_j[a0] - r_i[a0]);')
    if a == 'd':
        code.append(f'            const auto PA_1 = (a_j  * inv_S1) * (r_j[a1] - r_i[a1]);')
    if b == 'p' or b == 'd':
        code.append(f'            const auto PB_0 = (-a_i * inv_S1) * (r_j[b0] - r_i[b0]);')
    if b == 'd':
        code.append(f'            const auto PB_1 = (-a_i * inv_S1) * (r_j[b1] - r_i[b1]);')
    if c == 'p' or c == 'd':
        code.append(f'            const auto QC_0 = (a_l * inv_S2) * (r_l[c0] - r_k[c0]);')
    if c == 'd':
        code.append(f'            const auto QC_1 = (a_l * inv_S2) * (r_l[c1] - r_k[c1]);')
    if d == 'p' or d == 'd':
        code.append(f'            const auto QD_0 = (-a_k * inv_S2) * (r_l[d0] - r_k[d0]);')
    if d == 'd':
        code.append(f'            const auto QD_1 = (-a_k * inv_S2) * (r_l[d1] - r_k[d1]);')
    code.append('')
    code.append(f'            const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * (')

    #code.append(write_eri_code(eri_name.lower(), indent=20))
    code += write_eri_code(eri_name.lower(), indent=20).splitlines()
    code.append('')

    code.append('                    );')
    code.append('')
    code.append('            ERIs[threadIdx.y][threadIdx.x] += eri_ijkl * mat_D_full_AO[j_cgto * naos + l_cgto];')
    code.append('        }')
    code.append('    }')
    code.append('')
    code.append('    __syncthreads();')
    code.append('')
    code.append('    if ((threadIdx.y == 0) && (threadIdx.x == 0))')
    code.append('    {')
    code.append('        double K_ik = 0.0;')
    code.append('')
    code.append('        for (uint32_t y = 0; y < TILE_DIM_Y_K; y++)')
    code.append('        {')
    code.append('            for (uint32_t x = 0; x < TILE_DIM_X_K; x++)')
    code.append('            {')
    code.append('                K_ik += ERIs[y][x];')
    code.append('            }')
    code.append('        }')
    code.append('')
    code.append('        mat_K[ik] += K_ik;')
    code.append('    }')
    code.append('}')
    code.append('')

    return code
