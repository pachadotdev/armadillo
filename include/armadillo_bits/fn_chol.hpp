// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//! \addtogroup fn_chol
//! @{



template<typename T1>
arma_warn_unused
inline
typename enable_if2< is_blas_type<typename T1::elem_type>::value, const Op<T1, op_chol> >::result
chol
  (
  const Base<typename T1::elem_type,T1>& X,
  const char* layout = "upper"
  )
  {
  arma_debug_sigprint();
  
  const char sig = (layout != nullptr) ? layout[0] : char(0);
  
  arma_conform_check( ((sig != 'u') && (sig != 'l')), "chol(): layout must be \"upper\" or \"lower\"" );
  
  return Op<T1, op_chol>(X.get_ref(), ((sig == 'u') ? 0 : 1), 0 );
  }



template<typename T1>
inline
typename enable_if2< is_blas_type<typename T1::elem_type>::value, bool >::result
chol
  (
         Mat<typename T1::elem_type>&    out,
  const Base<typename T1::elem_type,T1>& X,
  const char* layout = "upper"
  )
  {
  arma_debug_sigprint();
  
  const char sig = (layout != nullptr) ? layout[0] : char(0);
  
  arma_conform_check( ((sig != 'u') && (sig != 'l')), "chol(): layout must be \"upper\" or \"lower\"" );
  
  const bool status = op_chol::apply_direct(out, X.get_ref(), ((sig == 'u') ? 0 : 1));
  
  if(status == false)
    {
    out.soft_reset();
    arma_warn(3, "chol(): decomposition failed");
    }
  
  return status;
  }



template<typename T1>
inline
typename enable_if2< is_blas_type<typename T1::elem_type>::value, bool >::result
chol
  (
         Mat<typename T1::elem_type>&    out,
         Mat<uword>&                     P,
  const Base<typename T1::elem_type,T1>& X,
  const char*                            layout = "upper",
  const char*                            P_mode = "matrix"
  )
  {
  arma_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const char sig_layout = (layout != nullptr) ? layout[0] : char(0);
  const char sig_P_mode = (P_mode != nullptr) ? P_mode[0] : char(0);
  
  arma_conform_check( ((sig_layout != 'u') && (sig_layout != 'l')), "chol(): argument 'layout' must be \"upper\" or \"lower\""   );
  arma_conform_check( ((sig_P_mode != 'm') && (sig_P_mode != 'v')), "chol(): argument 'P_mode' must be \"vector\" or \"matrix\"" );
  
  out = X.get_ref();
  
  arma_conform_check( (out.is_square() == false), "chol(): given matrix must be square sized", [&](){ out.soft_reset(); } );
  
  if(out.is_empty())
    {
    P.reset();
    return true;
    }
  
  if((arma_config::check_conform) && (auxlib::rudimentary_sym_check(out) == false))
    {
    if(is_cx<eT>::no )  { arma_warn(1, "chol(): given matrix is not symmetric"); }
    if(is_cx<eT>::yes)  { arma_warn(1, "chol(): given matrix is not hermitian"); }
    }
  
  bool status = false;
  
  if(sig_P_mode == 'v')
    {
    status = auxlib::chol_pivot(out, P, ((sig_layout == 'u') ? 0 : 1));
    }
  else
  if(sig_P_mode == 'm')
    {
    Mat<uword> P_vec;
    
    status = auxlib::chol_pivot(out, P_vec, ((sig_layout == 'u') ? 0 : 1));
    
    if(status)
      {
      // construct P
      
      const uword N = P_vec.n_rows;
      
      P.zeros(N,N);
      
      for(uword i=0; i < N; ++i)  { P.at(P_vec[i], i) = uword(1); }
      }
    }
  
  if(status == false)
    {
    out.soft_reset();
      P.soft_reset();
    arma_warn(3, "chol(): decomposition failed");
    }
  
  return status;
  }



//
// rank-revealing Cholesky decomposition functions
// this is based on
// Efficient estimation of maximum likelihood models with multiple fixed-effects: the R package FENmlm
// https://ideas.repec.org/p/luc/wpaper/18-13.html
//

template<typename T1>
inline
typename enable_if2< is_blas_type<typename T1::elem_type>::value, bool >::result
chol_rank
  (
    Mat<typename T1::elem_type>&           out,
    Col<uword>&                            excluded,
    uword&                                 rank_out,
    const Base<typename T1::elem_type,T1>& X,
    const char*                            layout = "upper",
    const typename T1::elem_type           tol = typename T1::elem_type(1e-12)
  )
  {
    arma_debug_sigprint();
    
    typedef typename T1::elem_type eT;
    
    const char sig = (layout != nullptr) ? layout[0] : char(0);
    
    arma_conform_check( ((sig != 'u') && (sig != 'l')), "chol_rank(): layout must be \"upper\" or \"lower\"" );
    
    const Mat<eT>& A = X.get_ref();
    
    arma_conform_check( (A.is_square() == false), "chol_rank(): given matrix must be square sized" );
    
    const uword N = A.n_rows;
    
    if(A.is_empty())
      {
      out.reset();
      excluded.reset();
      rank_out = 0;
      return true;
      }
    
    if((arma_config::check_conform) && (auxlib::rudimentary_sym_check(A) == false))
      {
      if(is_cx<eT>::no )  { arma_warn(1, "chol_rank(): given matrix is not symmetric"); }
      if(is_cx<eT>::yes)  { arma_warn(1, "chol_rank(): given matrix is not hermitian"); }
      }
    
    out.zeros(N, N);
    excluded.zeros(N);
    rank_out = 0;

    // Preallocate temporaries for excluded-aware path and keep an "alive"
    // index list of non-excluded pivots < j so we avoid scanning every time.
    Row<eT> rest;
    Mat<eT> OutBlock;
    Col<eT> r_head;
    arma::uvec idx_uvec;
    std::vector<uword> alive;
    alive.reserve(N);
    
    // Blocked rank-revealing Cholesky: process panels to enable Level-3 BLAS
    const uword env_block = []() -> uword {
      const char* s = std::getenv("CHOL_BLOCK_SIZE");
      if (!s) return uword(64);
      try { return static_cast<uword>(std::stoul(std::string(s))); } catch(...) { return uword(64); }
    }();

    const uword block = std::min<uword>(std::max<uword>(1, env_block), N);

    // diag_contrib accumulates sum_k out(k, j)^2 from processed panels for j > processed columns
    std::vector<eT> diag_contrib(N, eT(0));

    for (uword p = 0; p < N; p += block)
      {
      const uword end = std::min<uword>(N - 1, p + block - 1);

      // acceptances inside this panel (to be folded into diag_contrib in batch)
      std::vector<uword> panel_accepts;

      for (uword j = p; j <= end; ++j)
        {
        // Compute diagonal element using accumulated diag contributions from previous panels
        eT R_jj = A(j, j) - diag_contrib[j];

        // subtract contributions from earlier accepted pivots in this panel
        for (uword t = 0; t < panel_accepts.size(); ++t)
          {
          const uword idx = panel_accepts[t];
          const eT v = out(idx, j);
          R_jj -= v * v;
          }

        // Check for rank deficiency
        if (std::abs(R_jj) < tol)
          {
          excluded(j) = 1;
          continue;
          }

        R_jj = std::sqrt(R_jj);
        out(j, j) = R_jj;

        if (sig == 'u')
          {
          if (j + 1 < N)
            {
            const uword trailing_start = j + 1;
            const uword trailing_len   = N - trailing_start;

            // start with original A row
            rest = A.row(j).cols(trailing_start, N - 1);

            // subtract contributions from previously completed panels (alive)
            for (uword t = 0; t < alive.size(); ++t)
              {
              const uword ii = alive[t];
              const eT val = out(ii, j);
              if (val == eT(0)) continue;
              rest -= val * out.row(ii).cols(trailing_start, N - 1);
              }

            // subtract contributions from earlier accepts in this panel
            for (uword t = 0; t < panel_accepts.size(); ++t)
              {
              const uword ii = panel_accepts[t];
              const eT val = out(ii, j);
              if (val == eT(0)) continue;
              rest -= val * out.row(ii).cols(trailing_start, N - 1);
              }

            rest /= R_jj;
            out.row(j).cols(trailing_start, N - 1) = rest;
            }
          }
        else
          {
          if (j + 1 < N)
            {
            const uword trailing_start = j + 1;
            const uword trailing_len   = N - trailing_start;

            Col<eT> restc = A.col(j).rows(trailing_start, N - 1);

            for (uword t = 0; t < alive.size(); ++t)
              {
              const uword ii = alive[t];
              const eT val = out(ii, j);
              if (val == eT(0)) continue;
              restc -= out.col(ii).rows(trailing_start, N - 1) * val;
              }

            for (uword t = 0; t < panel_accepts.size(); ++t)
              {
              const uword ii = panel_accepts[t];
              const eT val = out(ii, j);
              if (val == eT(0)) continue;
              restc -= out.col(ii).rows(trailing_start, N - 1) * val;
              }

            restc /= R_jj;
            out.col(j).rows(trailing_start, N - 1) = restc;
            }
          }

        // record acceptance in panel
        panel_accepts.push_back(j);
        }

      // After processing the panel, fold panel contributions into diag_contrib for trailing columns
      const uword trailing_start_after = end + 1;
      if (!panel_accepts.empty() && trailing_start_after < N)
        {
        const uword trailing_len = N - trailing_start_after;
        // build small matrix G (accepted_in_panel x trailing_len)
        Mat<eT> G(panel_accepts.size(), trailing_len);
        for (uword t = 0; t < panel_accepts.size(); ++t)
          {
          const uword ii = panel_accepts[t];
          G.row(t) = out.row(ii).cols(trailing_start_after, N - 1);
          }

        rowvec col_sums = sum(square(G), 0);
        for (uword c = 0; c < trailing_len; ++c)
          {
          diag_contrib[trailing_start_after + c] += col_sums(c);
          }
        }

      // Append panel accepts to alive list (they are now fully processed)
      for (uword t = 0; t < panel_accepts.size(); ++t) alive.push_back(panel_accepts[t]);
      rank_out += panel_accepts.size();
      }

    // Mark excluded diagonal entries with NaN so downstream code can detect exclusion
    for(uword j = 0; j < N; ++j)
      {
      if (excluded(j) != 0)
        {
        out(j, j) = std::numeric_limits<eT>::quiet_NaN();
        }
      }

    return true;
  }



//! @}
