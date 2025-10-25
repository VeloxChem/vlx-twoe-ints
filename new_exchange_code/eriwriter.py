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

from intsutils import apply_hrr_b, apply_hrr_d, apply_vrr_c, apply_vrr_a, simplify_coef
from intswriter import write_integrals
from eri import ERI


def write_eri_code(eri_string, indent=20, flag='', gradient=None):

    assert len(eri_string) == 4

    list_a = []
    list_b = []
    list_c = []
    list_d = []

    if eri_string.lower()[0] == 'p':
        list_a = ['a0']
    elif eri_string.lower()[0] == 'd':
        list_a = ['a0','a1']

    if eri_string.lower()[1] == 'p':
        list_b = ['b0']
    elif eri_string.lower()[1] == 'd':
        list_b = ['b0','b1']

    if eri_string.lower()[2] == 'p':
        list_c = ['c0']
    elif eri_string.lower()[2] == 'd':
        list_c = ['c0','c1']

    if eri_string.lower()[3] == 'p':
        list_d = ['d0']
    elif eri_string.lower()[3] == 'd':
        list_d = ['d0','d1']

    #flip_ab, flip_cd = True, True
    #a = ERI(list_b, list_a, list_d, list_c)

    flip_ab, flip_cd = False, False
    a = ERI(list_a, list_b, list_c, list_d)

    # apply HRR on d

    if gradient == 'a':
        coefs, eris = a.apply_gradient_a()
        coefs, eris = apply_hrr_d(coefs, eris)

    elif gradient == 'b':
        coefs, eris = a.apply_gradient_b()
        coefs, eris = apply_hrr_d(coefs, eris)

    elif gradient == 'c':
        coefs, eris = a.apply_gradient_c()
        coefs, eris = apply_hrr_d(coefs, eris)

    elif gradient == 'd':
        coefs, eris = a.apply_gradient_d()
        coefs, eris = apply_hrr_d(coefs, eris)

    elif gradient is None:
        coefs, eris = a.apply_hrr_d()

    # apply HRR on b

    coefs, eris = apply_hrr_b(coefs, eris)

    # apply VRR on c

    coefs, eris = apply_vrr_c(coefs, eris)

    # apply VRR on a

    coefs, eris = apply_vrr_a(coefs, eris)

    # simplify coefficients

    final_list = []
    for ind, (c, e) in enumerate(zip(coefs, eris)):
        final_list.append((simplify_coef(c), ind, e))

    coefs, eris = [], []
    for (c, ind, e) in sorted(final_list):
        coefs.append(c)
        eris.append(e)

    # find out maximum order of Boys function

    max_bf_order = 0
    for c, e in zip(coefs, eris):
        if e.bf_order > max_bf_order:
            max_bf_order = e.bf_order

    # reorganize terms and finalize the result

    final_ints_lines = write_integrals(coefs, eris, indent=indent, flag=flag, flip_ab=flip_ab, flip_cd=flip_cd)

    if flag == 'split':
        return final_ints_lines

    # finalize the result

    final_ints_str = ''

    for line in final_ints_lines:

        if '( S S | S S )^' in line:
            assert '( S S | S S )^' in line
            bf_order = int(line.split('( S S | S S )^')[1].split()[0])
            old_keyword = f'( S S | S S )^{bf_order}'
            new_keyword = f'F{max_bf_order}_t[' + str(bf_order) + '] * ('
            line = line.replace(old_keyword, new_keyword)
            if flag == 'split':
                line = line.replace(f'+ F{max_bf_order}_t', f'F{max_bf_order}_t')

        if line.endswith('__nonewline__'):
            final_ints_str += line.split('__nonewline__')[0]
        else:
            final_ints_str += line + '\n'

    final_ints_str = final_ints_str.replace(r'( + ', '(')
    final_ints_str = final_ints_str.replace(r'(1.0 * (-1.0))', '(-1.0)')
    final_ints_str = final_ints_str.replace(r'(1.0 * (-2.0))', '(-2.0)')
    final_ints_str = final_ints_str.replace(r'(1.0 * (-3.0))', '(-3.0)')
    final_ints_str = final_ints_str.replace(r'(1.0 * (-4.0))', '(-4.0)')

    final_ints_str = final_ints_str.replace(r'PA_g', 'PA_x')
    final_ints_str = final_ints_str.replace(r'PB_g', 'PB_x')
    final_ints_str = final_ints_str.replace(r'QC_g', 'QC_x')
    final_ints_str = final_ints_str.replace(r'QD_g', 'QD_x')
    final_ints_str = final_ints_str.replace(r'PQ_g', 'PQ[grad_cart_ind]')

    return final_ints_str
