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

def write_integrals(coefs, eris, indent=16, flag='', flip_ab=True, flip_cd=True):

    final_ints_lines = []

    split_ints_dict = {}

    integrals = {}

    for c, e in zip(coefs, eris):

        integral_key = str(e)

        if integral_key not in integrals:
            integrals[integral_key] = {}

        deltas = []
        PQs = []
        coefs = []

        for t in c.split(' * '):
            assert '*' not in t
            if t.startswith('delta_'):
                deltas.append(t)
            elif (t.startswith('P') or t.startswith('Q') or t.startswith('A')
                  or t.startswith('B') or t.startswith('C')
                  or t.startswith('D')):
                PQs.append(t)
            else:
                coefs.append(t)

        delta_str = ' * '.join(deltas)
        PQ_str = ' * '.join(PQs)
        coef_str = ' * '.join(coefs)

        if not coef_str:
            coef_str = '1'

        if not delta_str:
            delta_str = '1'
        else:
            for ix, x in enumerate(['a0', 'a1', 'a2', 'b0', 'b1', 'b2']):
                for iy, y in enumerate(['a0', 'a1', 'a2', 'b0', 'b1', 'b2']):
                    if ix > iy:
                        delta_str = delta_str.replace(f'delta_{x}_{y}',
                                                      f'delta_{y}_{x}')

        if not PQ_str:
            PQ_str = '1'

        if coef_str.startswith('-'):
            coef_str = coef_str[1:]

            coef_str = coef_str.replace('S1', '(a_i + a_j)')
            coef_str = coef_str.replace('S2', '(a_k + a_l)')
            coef_str = coef_str.replace('S4', '(a_i + a_j + a_k + a_l)')

            coef_str = coef_str.replace('1 * ', '')

            coef_str = coef_str.replace('(a_i + a_j)', 'S1')
            coef_str = coef_str.replace('(a_k + a_l)', 'S2')
            coef_str = coef_str.replace('(a_i + a_j + a_k + a_l)', 'S4')

            PQ_str = '-' + PQ_str

        if coef_str not in integrals[integral_key]:
            integrals[integral_key][coef_str] = {}

        if delta_str not in integrals[integral_key][coef_str]:
            integrals[integral_key][coef_str][delta_str] = []

        integrals[integral_key][coef_str][delta_str].append(PQ_str)

    has_bf_order = False
    for integral_key in sorted(list(integrals.keys())):
        if '( S S | S S )^' in integral_key:
            has_bf_order = True
            break

    if has_bf_order:
        max_bf_order = 0
        for integral_key in sorted(list(integrals.keys())):
            assert '( S S | S S )^' in integral_key
            bf_order = int(integral_key.split('( S S | S S )^')[1])
            max_bf_order = max(max_bf_order, bf_order)

    if flag == 'bf_order':
        assert has_bf_order
        return max_bf_order

    indentation = indent

    for integral_idx, integral_key in enumerate(sorted(list(
            integrals.keys()))):
        if has_bf_order:
            assert '( S S | S S )^' in integral_key

        bf_order = int(integral_key.split('( S S | S S )^')[1])

        #plus_sign = '' if integral_idx == 0 else '+ '
        plus_sign = '+ '

        final_ints_lines.append('')
        final_ints_lines.append(' ' * indentation +
                                f'{plus_sign}{integral_key}')
        final_ints_lines.append('')

        for coef_idx, coef_str in enumerate(
                list(integrals[integral_key].keys())):

            if flag == 'split':
                final_ints_key = (bf_order, coef_idx)
                final_ints_lines = []

            indentation += 4

            new_coef_str = coef_str
            new_coef_str = new_coef_str.replace('S1', '(a_i + a_j)')
            new_coef_str = new_coef_str.replace('S2', '(a_k + a_l)')
            new_coef_str = new_coef_str.replace('S4',
                                                '(a_i + a_j + a_k + a_l)')

            new_coef_str = new_coef_str.replace('1 ', '1.0 ')
            new_coef_str = new_coef_str.replace('2 ', '2.0 ')

            new_coef_str = new_coef_str.replace('(a_i + a_j)', 'S1')
            new_coef_str = new_coef_str.replace('(a_k + a_l)', 'S2')
            new_coef_str = new_coef_str.replace('(a_i + a_j + a_k + a_l)', 'S4')

            plus_sign = '' if coef_idx == 0 else '+ '
            if coef_str.startswith('-'):
                plus_sign = ''

            coef_output = '(' if coef_str == '1' else f'{new_coef_str} * ('
            coef_output = ' ' * indentation + f'{plus_sign}{coef_output}'

            delta_output_list = []
            PQ_output_list = []

            for delta_idx, delta_str in enumerate(
                    list(integrals[integral_key][coef_str].keys())):
                indentation += 4

                new_delta_str = delta_str
                new_delta_str = new_delta_str.replace('delta_', 'delta[')
                new_delta_str = new_delta_str.replace('_', '][')
                new_delta_str = new_delta_str.replace(' * ', '] * ')
                new_delta_str = new_delta_str + ']'

                plus_sign = '' if delta_idx == 0 else '+ '

                delta_output = '' if delta_str == '1' else f'{new_delta_str}'
                delta_output = ' ' * indentation + f'{plus_sign}{delta_output}'
                delta_output_list.append(delta_output)

                PQ_output = ''

                PQ_all_digits = all([
                    (x.isdigit()
                     or (x.startswith('-') and x.lstrip('-').isdigit()))
                    for x in integrals[integral_key][coef_str][delta_str]
                ])

                if PQ_all_digits:
                    PQ_sum = sum([
                        int(x)
                        for x in integrals[integral_key][coef_str][delta_str]
                    ])
                    if PQ_sum != 1:
                        if PQ_sum < 0:
                            PQ_output += f' * ({PQ_sum:.1f})'
                        else:
                            PQ_output += f' * {PQ_sum:.1f}'
                    else:
                        if delta_str == '1':
                            PQ_output += '1.0'

                else:
                    if delta_str != '1':
                        PQ_output += ' * ('

                    PQ_dict = {}

                    for PQ_str in integrals[integral_key][coef_str][delta_str]:
                        if PQ_str.startswith('-'):
                            PQ_str_abs = PQ_str[1:]
                        else:
                            PQ_str_abs = PQ_str

                        if PQ_str_abs not in PQ_dict:
                            PQ_dict[PQ_str_abs] = 0

                        if PQ_str.startswith('-'):
                            PQ_dict[PQ_str_abs] -= 1
                        else:
                            PQ_dict[PQ_str_abs] += 1

                    for PQ_idx, PQ_str in enumerate(list(PQ_dict.keys())):
                        if PQ_idx == 0:
                            plus_sign = ''
                        elif delta_str == '1':
                            plus_sign = '\n' + ' ' * indentation + '+ '
                        else:
                            plus_sign = ' + '

                        new_PQ_str = PQ_str

                        if flip_ab:
                            new_PQ_str = new_PQ_str.replace('PA', 'P_A_')
                            new_PQ_str = new_PQ_str.replace('PB', 'PA')
                            new_PQ_str = new_PQ_str.replace('P_A_', 'PB')
                            new_PQ_str = new_PQ_str.replace('AB', 'BA')

                        if flip_cd:
                            new_PQ_str = new_PQ_str.replace('QC', 'Q_C_')
                            new_PQ_str = new_PQ_str.replace('QD', 'QC')
                            new_PQ_str = new_PQ_str.replace('Q_C_', 'QD')
                            new_PQ_str = new_PQ_str.replace('CD', 'DC')

                        for x in [0, 1, 2]:
                            new_PQ_str = new_PQ_str.replace(
                                f'PA_a{x}', f'PA_{x}')
                            new_PQ_str = new_PQ_str.replace(
                                f'PB_b{x}', f'PB_{x}')
                            new_PQ_str = new_PQ_str.replace(
                                f'QC_c{x}', f'QC_{x}')
                            new_PQ_str = new_PQ_str.replace(
                                f'QD_d{x}', f'QD_{x}')

                        for x in 'abcd':
                            for y in [0, 1, 2]:
                                new_PQ_str = new_PQ_str.replace(f'PQ_{x}{y}', f'PQ[{x}{y}]')

                        if new_PQ_str == '1':
                            new_PQ_str = '1.0'

                        if PQ_dict[PQ_str] == 0:
                            #PQ_output += f'{plus_sign}0.0'
                            pass
                        else:
                            PQ_output += f'{plus_sign}{new_PQ_str}'
                            if PQ_dict[PQ_str] < 0:
                                PQ_output += f' * ({PQ_dict[PQ_str]:.1f})'
                            elif PQ_dict[PQ_str] != 1:
                                PQ_output += f' * {PQ_dict[PQ_str]:.1f}'

                if not (delta_str == '1' or PQ_all_digits):
                    PQ_output += ')'

                PQ_output_list.append(PQ_output)

                indentation -= 4

            skip_coef_output = True
            for delta_output, PQ_output in zip(delta_output_list,
                                               PQ_output_list):
                if not (PQ_output == r' * (0.0)' or PQ_output == r' * 0.0'):
                    skip_coef_output = False
                    break

            if skip_coef_output:
                if len(integrals[integral_key]) == 1:
                    #final_ints_lines.append(' ' * indentation + '0.0')
                    pass
            else:
                PQ_delta_dict = {}
                for delta_output, PQ_output in zip(delta_output_list,
                                                   PQ_output_list):
                    if not (PQ_output == r' * (0.0)'
                            or PQ_output == r' * 0.0'):
                        if PQ_output not in PQ_delta_dict:
                            PQ_delta_dict[PQ_output] = []
                        new_delta_output = delta_output.lstrip(
                            ' ' * indentation).lstrip('+ ')
                        PQ_delta_dict[PQ_output].append(new_delta_output)

                PQ_common_factor = None
                for PQ_output in PQ_delta_dict:
                    for PQ_prod in PQ_output.split(' + '):
                        found_factor = False
                        factor = None
                        match_factor = False
                        if PQ_prod == ' * (':
                            continue
                        for term in PQ_prod.split(' * '):
                            term = term.strip()
                            term = term.replace('(', '').replace(')', '')
                            if term.lstrip('-').replace('.', '').isdigit():
                                found_factor = True
                                factor = float(term)
                                break
                        if found_factor and factor is not None:
                            if PQ_common_factor is None:
                                PQ_common_factor = factor
                                match_factor = True
                            else:
                                match_factor = (PQ_common_factor == factor)
                        if not match_factor:
                            PQ_common_factor = None
                            break
                    if PQ_common_factor is None:
                        break

                if ' * ' in coef_output:
                    coef_start = ' ' * indentation
                    coef_output = coef_output.lstrip(coef_start)
                    if coef_output.startswith('+ '):
                        coef_start += '+ '
                        coef_output = coef_output.lstrip('+ ')

                    coef_terms = coef_output.split(' * ')
                    coef_coef = 1.0
                    if PQ_common_factor is not None:
                        coef_coef *= PQ_common_factor
                    for coef_term_idx in range(len(coef_terms)):
                        term = coef_terms[coef_term_idx]
                        if ' / ' in term:
                            term_a, term_b = term.split(' / ')
                            if term_a.lstrip('-').replace('.', '').isdigit():
                                coef_coef *= float(term_a)
                                term = '1.0' + ' / ' + term_b
                        else:
                            if term.lstrip('-').replace('.', '').isdigit():
                                coef_coef *= float(term)
                                term = '1.0'
                        coef_terms[coef_term_idx] = term

                    if coef_coef < 0:
                        coef_terms = [f'({coef_coef})'] + coef_terms
                    else:
                        coef_terms = [f'{coef_coef}'] + coef_terms
                    coef_output = coef_start + ' * '.join(coef_terms)
                    coef_output = coef_output.replace(' * 1.0 ', ' ')
                    coef_output = coef_output.replace(' 1.0 * ', ' ')

                else:
                    if PQ_common_factor is not None:
                        if PQ_common_factor < 0:
                            PQ_f = f'({PQ_common_factor}) * '
                        else:
                            PQ_f = f'{PQ_common_factor} * '
                        coef_output = (coef_output[:-1] + f'{PQ_f}' +
                                       coef_output[-1])

                final_ints_lines.append(coef_output)

                for idx, (PQ_output,
                          delta_s) in enumerate(PQ_delta_dict.items()):
                    if len(delta_s) > 1:
                        delta_line = '(' + ' + '.join(delta_s) + ')'
                    elif len(delta_s) == 1:
                        delta_line = ' + '.join(delta_s)
                    else:
                        delta_line = ''
                    delta_line = ('' if idx == 0 else '+ ') + delta_line
                    delta_line = ' ' * (indentation + 4) + delta_line

                    if PQ_common_factor is not None:
                        if PQ_common_factor < 0:
                            PQ_f = f' * ({PQ_common_factor:.1f})'
                        else:
                            PQ_f = f' * {PQ_common_factor:.1f}'
                        PQ_output = PQ_output.replace(PQ_f, '')

                    final_ints_lines.append(delta_line + PQ_output)

                final_ints_lines.append(' ' * indentation + ')')

            final_ints_lines.append('')

            indentation -= 4

            if flag == 'split':
                split_ints_dict[final_ints_key] = list(final_ints_lines)

        final_ints_lines.append(' ' * indentation + ')')

    if flag == 'split':
        return split_ints_dict
    else:
        return final_ints_lines
