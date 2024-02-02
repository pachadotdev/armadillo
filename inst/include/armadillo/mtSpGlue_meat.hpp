// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

//! \addtogroup mtSpGlue
//! @{

template <typename out_eT, typename T1, typename T2, typename spglue_type>
inline mtSpGlue<out_eT, T1, T2, spglue_type>::mtSpGlue(const T1& in_A, const T2& in_B)
    : A(in_A), B(in_B) {
  arma_extra_debug_sigprint();
}

template <typename out_eT, typename T1, typename T2, typename spglue_type>
inline mtSpGlue<out_eT, T1, T2, spglue_type>::~mtSpGlue() {
  arma_extra_debug_sigprint();
}

template <typename out_eT, typename T1, typename T2, typename spglue_type>
template <typename eT2>
arma_inline bool mtSpGlue<out_eT, T1, T2, spglue_type>::is_alias(
    const SpMat<eT2>& X) const {
  return (A.is_alias(X) || B.is_alias(X));
}

//! @}
