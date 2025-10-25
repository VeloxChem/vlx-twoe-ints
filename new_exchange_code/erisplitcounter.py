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

from pathlib import Path


def count_split_twoe_ints(vlxpath, eri_type, eri_name):

    assert isinstance(vlxpath, str)
    assert isinstance(eri_type, str)
    assert isinstance(eri_name, str)

    assert eri_type in ['coulomb', 'exchange']
    assert len(eri_name) == 4
    for x in eri_name:
        assert x in 'spd'

    fpath = Path(f'{vlxpath}/src/gpu/Eri{eri_type.capitalize()}.cu')
    assert fpath.is_file()

    data = {}

    with fpath.open('r') as fh:
        kernel_name = None

        for line in fh:
            if f'compute{eri_type.capitalize()}Fock{eri_name.upper()}' in line:
                kernel_name = line.split('(')[0]
                if kernel_name not in data:
                    data[kernel_name] = 0
                    # print(kernel_name)
                    # print()
            elif f'compute{eri_type.capitalize()}Fock' in line:
                kernel_name = None

            if (line.strip().startswith('F') or line.strip().startswith('+ F')):
                if (r'_t[' in line) and line.strip().endswith(' * ('):
                    if kernel_name is not None:
                        data[kernel_name] += 1
                        # print(line.strip())
                        # print()

    values = list(data.values())
    results = [sum(values[:p + 1]) for p in range(len(values))]

    return results
