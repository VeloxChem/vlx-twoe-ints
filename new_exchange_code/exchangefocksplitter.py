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


def split_exchange_fock_code(eri_name, lines, opt_sizes):

    angmoms = 'spd'

    assert len(eri_name) == 4
    for x in eri_name:
        assert x in angmoms

    # kernel name

    kernel_name = f'computeExchangeFock{eri_name.upper()}'
    kernel_file_name = kernel_name.replace('compute', 'Eri')

    # group code in a dictionary code_dict

    code_dict = {
        'prep': [],
        'fock': [],
        'final': [],
    }

    code_flag = None

    for line_index, line in enumerate(lines):

        if kernel_name in line:
            code_flag = 'prep'
            code_dict[code_flag].append(lines[line_index - 1])

        if code_flag is not None:

            if line.strip().startswith('const double eri_ijkl ='):
                code_flag = 'fock'

            code_dict[code_flag].append(line)

            if '    );' in line:
                code_flag = 'final'

            if line.startswith('}'):
                code_flag = None

    # boys function info

    boys_compute_line = None
    boys_var_name = None

    for key in code_dict:
        for line in code_dict[key]:
            if 'gpu::computeBoysFunction' in line:
                if boys_compute_line is None:
                    boys_compute_line = line.rstrip()
                if boys_var_name is None:
                    boys_var_name = boys_compute_line.split('(')[1].split(',')[0]

    # print('kernel_name', kernel_name)
    # print('boys_compute_line', boys_compute_line)
    # print('boys_var_name', boys_var_name)

    atom_terms_dict = {
        'fock': [],
    }

    for key in ['fock']:

        bf_order = None
        current_term = []

        for line in code_dict[key]:

            if f'{boys_var_name}[' in line:

                if bf_order is not None:
                    if current_term:
                        current_term.append(' ' * 8 + '                        )')
                        atom_terms_dict[key].append((bf_order, list(current_term)))
                        current_term = []

                bf_order = int(line.split('[')[1].split(']')[0])
                # print('bf_order', bf_order)

            if f'{boys_var_name}[' not in line and line.rstrip().endswith('('):

                if bf_order is not None:
                    if current_term:
                        current_term.append(' ' * 8 + '                        )')
                        atom_terms_dict[key].append((bf_order, list(current_term)))
                        current_term = []

            if f'{boys_var_name}[' not in line and line.strip() and line.strip() != ')' and line.strip() != ');':

                if bf_order is not None:
                    current_term.append(line)

            if '  );' in line:

                if current_term:
                    current_term.append(' ' * 8 + '                        )')
                    atom_terms_dict[key].append((bf_order, list(current_term)))
                    current_term = []

                bf_order = None

    # This can be useful in tuning the splitting sizes
    # indices = []
    # for idx in indices:
    #     if idx == 0:
    #         assert opt_sizes[idx] > 1
    #         new_idx = int((0 + opt_sizes[idx]) / 2 + 0.5)
    #     else:
    #         assert opt_sizes[idx] - opt_sizes[idx - 1] > 1
    #         new_idx = int((opt_sizes[idx - 1] + opt_sizes[idx]) / 2 + 0.5)
    #     opt_sizes.append(new_idx)
    # opt_sizes = list(sorted(opt_sizes))

    # print(opt_sizes, len(opt_sizes))

    if opt_sizes[-1] != len(atom_terms_dict['fock']):
        print(eri_name)
        print(opt_sizes)
        print(len(atom_terms_dict['fock']))
        print()

    assert opt_sizes[-1] == len(atom_terms_dict['fock'])

    # print fock code

    max_bf_order = int(boys_var_name.replace('F', '').replace('_t', ''))

    new_code = []

    # === print fock ===

    start_inds = [0] + opt_sizes
    end_inds = list(opt_sizes)

    for idx, (start, end) in enumerate(zip(start_inds, end_inds)):

        local_max_bf_order = max([bf_order for (bf_order, term) in atom_terms_dict['fock'][start:end]])

        for line in code_dict['prep']:

            line = line.replace(kernel_name, f'{kernel_name}{idx}')

            if f'double {boys_var_name}' in line:
                line = line.replace(f'{boys_var_name}[{max_bf_order + 1}]', f'{boys_var_name}[{local_max_bf_order + 1}]')

            if 'gpu::computeBoysFunction' in line:
                line = line.replace(f'r2_PQ, {max_bf_order}, boys_func_table,', f'r2_PQ, {local_max_bf_order}, boys_func_table,')

            if line.rstrip().endswith('d2;'):
                omega_bf_order = int(line.split('[')[1].split(']')[0])
                if omega_bf_order > local_max_bf_order:
                    line = None

            if line is not None:
                new_code.append(line)

        for line in code_dict['fock'][:2]:

            new_code.append(line)

        for (bf_order, term) in atom_terms_dict['fock'][start:end]:

            if term[0].lstrip().startswith('+ '):
                new_code.append(term[0].replace('    + ', f'    + {boys_var_name}[{bf_order}] * '))
            else:
                n_spaces = len(term[0]) - len(term[0].lstrip())
                #plus_sign = '' if bf_order == 0 else '+ '
                plus_sign = '+ '
                new_code.append(' ' * n_spaces + f'{plus_sign}{boys_var_name}[{bf_order}] * ' + term[0].lstrip())

            for line in term[1:]:
                new_code.append(line)

            new_code.append('')

        for line in code_dict['fock'][-2:]:

            new_code.append(line)

        for line in code_dict['final']:

            new_code.append(line)

        new_code.append('')

    return new_code
