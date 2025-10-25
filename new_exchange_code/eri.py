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

from intsutils import (apply_vrr_a_once, apply_vrr_c_once, apply_hrr_b_once,
                       apply_hrr_d_once)


class ERI:

    def __init__(self, cart_a, cart_b, cart_c, cart_d, bf_order=0):

        assert isinstance(bf_order, int)
        assert bf_order >= 0

        self.cart_a = list(cart_a)
        self.cart_b = list(cart_b)
        self.cart_c = list(cart_c)
        self.cart_d = list(cart_d)
        self.bf_order = bf_order

    def __repr__(self):

        label_a = self.get_angmom_str(self.cart_a)
        label_b = self.get_angmom_str(self.cart_b)
        label_c = self.get_angmom_str(self.cart_c)
        label_d = self.get_angmom_str(self.cart_d)

        return f'( {label_a} {label_b} | {label_c} {label_d} )^{self.bf_order}'

    @staticmethod
    def get_angmom_str(cart_comp):

        labels = 'SPDFGHI'
        if len(cart_comp) == 0:
            return labels[len(cart_comp)]
        else:
            return labels[len(cart_comp)] + '_' + '_'.join(cart_comp)

    @property
    def La(self):

        return len(self.cart_a)

    @property
    def Lb(self):

        return len(self.cart_b)

    @property
    def Lc(self):

        return len(self.cart_c)

    @property
    def Ld(self):

        return len(self.cart_d)

    def apply_gradient_a(self):

        # Eq.(5), Obara-Saika JCP 1986

        coef_s = ['2 * a_i']
        eri_s = [
            ERI(self.cart_a + ['g'], self.cart_b, self.cart_c, self.cart_d,
                self.bf_order)
        ]

        for ind_a in range(len(self.cart_a)):

            coef_s.append(f'(-1) * delta_{self.cart_a[ind_a]}_g')

            new_cart_a = list(self.cart_a)
            new_cart_a.pop(ind_a)
            eri_s.append(
                ERI(new_cart_a, self.cart_b, self.cart_c, self.cart_d,
                    self.bf_order)
            )

        return coef_s, eri_s

    def apply_gradient_b(self):

        # Eq.(5), Obara-Saika JCP 1986

        coef_s = ['2 * a_j']
        eri_s = [
            ERI(self.cart_a, self.cart_b + ['g'], self.cart_c, self.cart_d,
                self.bf_order)
        ]

        for ind_b in range(len(self.cart_b)):

            coef_s.append(f'(-1) * delta_{self.cart_b[ind_b]}_g')

            new_cart_b = list(self.cart_b)
            new_cart_b.pop(ind_b)
            eri_s.append(
                ERI(self.cart_a, new_cart_b, self.cart_c, self.cart_d,
                    self.bf_order)
            )

        return coef_s, eri_s

    def apply_gradient_c(self):

        # Eq.(5), Obara-Saika JCP 1986

        coef_s = ['2 * a_k']
        eri_s = [
            ERI(self.cart_a, self.cart_b, self.cart_c + ['g'], self.cart_d,
                self.bf_order)
        ]

        for ind_c in range(len(self.cart_c)):

            coef_s.append(f'(-1) * delta_{self.cart_c[ind_c]}_g')

            new_cart_c = list(self.cart_c)
            new_cart_c.pop(ind_c)
            eri_s.append(
                ERI(self.cart_a, self.cart_b, new_cart_c, self.cart_d,
                    self.bf_order)
            )

        return coef_s, eri_s

    def apply_gradient_d(self):

        # Eq.(5), Obara-Saika JCP 1986

        coef_s = ['2 * a_l']
        eri_s = [
            ERI(self.cart_a, self.cart_b, self.cart_c, self.cart_d + ['g'],
                self.bf_order)
        ]

        for ind_d in range(len(self.cart_d)):

            coef_s.append(f'(-1) * delta_{self.cart_d[ind_d]}_g')

            new_cart_d = list(self.cart_d)
            new_cart_d.pop(ind_d)
            eri_s.append(
                ERI(self.cart_a, self.cart_b, self.cart_c, new_cart_d,
                    self.bf_order)
            )

        return coef_s, eri_s

    def apply_hrr_d_once(self):

        if self.Ld == 0:

            coef_s = ['1']
            eri_s = [
                ERI(self.cart_a, self.cart_b, self.cart_c, self.cart_d,
                    self.bf_order)
            ]

        else:

            # Original RR in Eq.(39), Obara-Saika JCP 1986
            # For HRR on d, see also Eq.(13) in JCTC 2018, 14, 572-587
            # Note that we rewrite CD in terms of QC and QD

            coef_s = [
                '1', f'QD_{self.cart_d[-1]}', f'(-1) * QC_{self.cart_d[-1]}'
            ]
            eri_s = [
                ERI(self.cart_a, self.cart_b, self.cart_c + [self.cart_d[-1]],
                    self.cart_d[:-1], self.bf_order),
                ERI(self.cart_a, self.cart_b, self.cart_c, self.cart_d[:-1],
                    self.bf_order),
                ERI(self.cart_a, self.cart_b, self.cart_c, self.cart_d[:-1],
                    self.bf_order)
            ]

        return coef_s, eri_s

    def apply_hrr_b_once(self):

        if self.Lb == 0:

            coef_s = ['1']
            eri_s = [
                ERI(self.cart_a, self.cart_b, self.cart_c, self.cart_d,
                    self.bf_order)
            ]

        else:

            # Original RR in Eq.(39), Obara-Saika JCP 1986
            # For HRR on b, see also Eq.(12) in JCTC 2018, 14, 572-587
            # Note that we rewrite AB in terms of PA and PB

            coef_s = [
                '1', f'PB_{self.cart_b[-1]}', f'(-1) * PA_{self.cart_b[-1]}'
            ]
            eri_s = [
                ERI(self.cart_a + [self.cart_b[-1]], self.cart_b[:-1],
                    self.cart_c, self.cart_d, self.bf_order),
                ERI(self.cart_a, self.cart_b[:-1], self.cart_c, self.cart_d,
                    self.bf_order),
                ERI(self.cart_a, self.cart_b[:-1], self.cart_c, self.cart_d,
                    self.bf_order)
            ]

        return coef_s, eri_s

    def apply_vrr_c_once(self):

        if self.Lc == 0:

            coef_s = ['1']
            eri_s = [
                ERI(self.cart_a, self.cart_b, self.cart_c, self.cart_d,
                    self.bf_order)
            ]

        else:

            # Original RR in Eq.(39), Obara-Saika JCP 1986
            # For VRR on c, see also Eq.(11) in JCTC 2018, 14, 572-587

            # Note that our PQ differs by a factor of -1
            coef_s = [
                f'QC_{self.cart_c[-1]}',
                f'(-1) * S1/S4 * PQ_{self.cart_c[-1]}',
            ]

            for ind_c in range(len(self.cart_c) - 1):
                coef_s.append(
                    f'1/2 * 1/S2 * delta_{self.cart_c[ind_c]}_{self.cart_c[-1]}'
                )
                coef_s.append(
                    f'(-1) * 1/2 * 1/S2 * S1/S4 * delta_{self.cart_c[ind_c]}_{self.cart_c[-1]}'
                )

            for ind_a in range(len(self.cart_a)):
                coef_s.append(
                    f'1/2 * 1/S4 * delta_{self.cart_a[ind_a]}_{self.cart_c[-1]}'
                )

            eri_s = [
                ERI(self.cart_a, self.cart_b, self.cart_c[:-1], self.cart_d,
                    self.bf_order),
                ERI(self.cart_a, self.cart_b, self.cart_c[:-1], self.cart_d,
                    self.bf_order + 1),
            ]

            for ind_c in range(len(self.cart_c) - 1):
                new_cart_c = list(self.cart_c[:-1])
                new_cart_c.pop(ind_c)
                eri_s.append(
                    ERI(self.cart_a, self.cart_b, new_cart_c, self.cart_d,
                        self.bf_order))
                eri_s.append(
                    ERI(self.cart_a, self.cart_b, new_cart_c, self.cart_d,
                        self.bf_order + 1))

            for ind_a in range(len(self.cart_a)):
                new_cart_a = list(self.cart_a)
                new_cart_a.pop(ind_a)
                eri_s.append(
                    ERI(new_cart_a, self.cart_b, self.cart_c[:-1], self.cart_d,
                        self.bf_order + 1))

        return coef_s, eri_s

    def apply_vrr_a_once(self):

        if self.La == 0:

            coef_s = ['1']
            eri_s = [
                ERI(self.cart_a, self.cart_b, self.cart_c, self.cart_d,
                    self.bf_order)
            ]

        else:

            # Original RR in Eq.(39), Obara-Saika JCP 1986
            # For VRR on a, see also Eq.(10) in JCTC 2018, 14, 572-587

            # Note that our PQ differs by a factor of -1
            coef_s = [
                f'PA_{self.cart_a[-1]}',
                f'S2/S4 * PQ_{self.cart_a[-1]}',
            ]

            for ind_a in range(len(self.cart_a) - 1):
                coef_s.append(
                    f'1/2 * 1/S1 * delta_{self.cart_a[ind_a]}_{self.cart_a[-1]}'
                )
                coef_s.append(
                    f'(-1) * 1/2 * 1/S1 * S2/S4 * delta_{self.cart_a[ind_a]}_{self.cart_a[-1]}'
                )

            for ind_c in range(len(self.cart_c)):
                coef_s.append(
                    f'1/2 * 1/S4 * delta_{self.cart_c[ind_c]}_{self.cart_c[-1]}'
                )

            eri_s = [
                ERI(self.cart_a[:-1], self.cart_b, self.cart_c, self.cart_d,
                    self.bf_order),
                ERI(self.cart_a[:-1], self.cart_b, self.cart_c, self.cart_d,
                    self.bf_order + 1),
            ]

            for ind_a in range(len(self.cart_a) - 1):
                new_cart_a = list(self.cart_a[:-1])
                new_cart_a.pop(ind_a)
                eri_s.append(
                    ERI(new_cart_a, self.cart_b, self.cart_c, self.cart_d,
                        self.bf_order))
                eri_s.append(
                    ERI(new_cart_a, self.cart_b, self.cart_c, self.cart_d,
                        self.bf_order + 1))

            for ind_c in range(len(self.cart_c)):
                new_cart_c = list(self.cart_c)
                new_cart_c.pop(ind_c)
                eri_s.append(
                    ERI(self.cart_a[:-1], self.cart_b, new_cart_c, self.cart_d,
                        self.bf_order + 1))

        return coef_s, eri_s

    def apply_hrr_d(self):

        coefs, eris = self.apply_hrr_d_once()

        while True:

            done = True
            for e in eris:
                if e.Ld > 0:
                    done = False
                    break

            if done:
                break

            coefs, eris = apply_hrr_d_once(coefs, eris)

        return coefs, eris

    def apply_hrr_b(self):

        coefs, eris = self.apply_hrr_b_once()

        while True:

            done = True
            for e in eris:
                if e.Lb > 0:
                    done = False
                    break

            if done:
                break

            coefs, eris = apply_hrr_b_once(coefs, eris)

        return coefs, eris

    def apply_vrr_c(self):

        coefs, eris = self.apply_vrr_c_once()

        while True:

            done = True
            for e in eris:
                if e.Lc > 0:
                    done = False
                    break

            if done:
                break

            coefs, eris = apply_vrr_c_once(coefs, eris)

        return coefs, eris

    def apply_vrr_a(self):

        coefs, eris = self.apply_vrr_a_once()

        while True:

            done = True
            for e in eris:
                if e.La > 0:
                    done = False
                    break

            if done:
                break

            coefs, eris = apply_vrr_a_once(coefs, eris)

        return coefs, eris
