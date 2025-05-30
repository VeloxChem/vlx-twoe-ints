import sys

angmoms = 'spd'

eri_name = sys.argv[1]

assert len(eri_name) == 4

assert eri_name[0].lower() in angmoms
assert eri_name[1].lower() in angmoms
assert eri_name[2].lower() in angmoms
assert eri_name[3].lower() in angmoms

# kernel name

kernel_name = f'computeExchangeFock{eri_name.upper()}'
kernel_file_name = kernel_name.replace('compute', 'Eri')

# read in code

with open(f'exchange_fock.hip', 'r') as fh:
    lines = fh.readlines()

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

#print(kernel_name)
#print(boys_compute_line)
#print(boys_var_name)

# header and last line

header = """//
//                                   VELOXCHEM
//              ----------------------------------------------------
//                          An Electronic Structure Code
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  Copyright 2018-2025 VeloxChem developers
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this
//     list of conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software without
//     specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

hip_header = header + f"""
#include <hip/hip_runtime.h>

#include "BoysFuncGPU.hpp"
#include "{kernel_file_name}.hpp"

namespace gpu {{  // gpu namespace

"""

hpp_header = header + f"""
#ifndef Eri{kernel_name.replace('compute', '')}_hpp
#define Eri{kernel_name.replace('compute', '')}_hpp

#include <cstdint>

#include "GpuConstants.hpp"

namespace gpu {{  // gpu namespace

"""

last_line = """
}  // namespace gpu
"""

hip_last_line = last_line
hpp_last_line = last_line + """
#endif
"""

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
                    current_term.append(' ' * 8 + '                        )\n')
                    atom_terms_dict[key].append((bf_order, list(current_term)))
                    current_term = []

            bf_order = int(line.split('[')[1].split(']')[0])

        if f'{boys_var_name}[' not in line and line.rstrip().endswith('('):

            if bf_order is not None:
                if current_term:
                    current_term.append(' ' * 8 + '                        )\n')
                    atom_terms_dict[key].append((bf_order, list(current_term)))
                    current_term = []

        if f'{boys_var_name}[' not in line and line.strip() and line.strip() != ')' and line.strip() != ');':

            if bf_order is not None:
                current_term.append(line)

        if '  );' in line:

            if current_term:
                current_term.append(' ' * 8 + '                        )\n')
                atom_terms_dict[key].append((bf_order, list(current_term)))
                current_term = []

            bf_order = None

#print(len(atom_terms_dict['fock']))

opt_sizes = [len(atom_terms_dict['fock'])]

opt_sizes = [12, 20, 30, 38, 44, 50, 52, 56, 59, 63, 66, 67, 70, 74, 77, 80, 84, 95] 

indices = [5]
for idx in indices:
    if idx == 0:
        assert opt_sizes[idx] > 1
        new_idx = int((0 + opt_sizes[idx]) / 2 + 0.5)
    else:
        assert opt_sizes[idx] - opt_sizes[idx - 1] > 1
        new_idx = int((opt_sizes[idx - 1] + opt_sizes[idx]) / 2 + 0.5)
    opt_sizes.append(new_idx)
opt_sizes = list(sorted(opt_sizes))

print(opt_sizes, len(opt_sizes))

assert opt_sizes[-1] == len(atom_terms_dict['fock'])

# print fock code

max_bf_order = int(boys_var_name.replace('F', '').replace('_t', ''))

with open(f'mytest.hip', 'w') as fh:

    fh.write(hip_header)

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
                    line = ''

            fh.write(line)

        for line in code_dict['fock'][:2]:

            fh.write(line)

        for (bf_order, term) in atom_terms_dict['fock'][start:end]:

            if term[0].lstrip().startswith('+ '):
                fh.write(term[0].replace('    + ', f'    + {boys_var_name}[{bf_order}] * '))
            else:
                n_spaces = len(term[0]) - len(term[0].lstrip())
                plus_sign = '' if bf_order == 0 else '+ '
                fh.write(' ' * n_spaces + f'{plus_sign}{boys_var_name}[{bf_order}] * ' + term[0].lstrip())

            for line in term[1:]:
                fh.write(line)

            fh.write('\n')

        for line in code_dict['fock'][-2:]:

            fh.write(line)

        for line in code_dict['final']:

            fh.write(line)

        fh.write('\n')

    fh.write(hip_last_line)
