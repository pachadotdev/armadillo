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
  
  // Blocked rank-revealing Cholesky with Level 3 BLAS optimizations
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
    
    // Calculate contribution from previous panels (0 to p-1) to the current active submatrix
    // Matrix update: A(p:N, p:N) -= U(0:p, p:N)' * U(0:p, p:N)
    Mat<eT> BlockUpdate;
    if (p > 0)
      {
      if (sig == 'u')
        {
        const Mat<eT> U_top  = out.submat(0, p, p - 1, N - 1);
        const Mat<eT> U_left = out.submat(0, p, p - 1, end);
        BlockUpdate = U_left.t() * U_top;
        }
      else
        {
        // For lower: A(j, trailing) -= sum over k<p of L(j,k)*L(trailing,k)^T
        // Here we compute updates for columns j=p...end.
        // BlockUpdate(row_idx, col_idx) corresponds to update for L(row, col).
        // Update(r, c) = L(r, 0:p-1) * L(c, 0:p-1)^T
        // We only need this for r >= trailing, c in p...end
        // Because of the loop structure, we compute full block for simplicity or sliced.
        
        const Mat<eT> L_left = out.submat(p, 0, N - 1, p - 1);
        const Mat<eT> L_top  = out.submat(p, 0, end,   p - 1);
        BlockUpdate = L_left * L_top.t();
        }
      }

    std::vector<uword> panel_accepts;
    panel_accepts.reserve(block);

    for (uword j = p; j <= end; ++j)
      {
      // Compute diagonal element
      eT R_jj = A(j, j) - diag_contrib[j];
      
      // Subtract intra-panel contributions to diagonal
      // We rely on zero-padding of excluded rows so we can iterate or vectorise
      if (sig == 'u')
        {
        if (j > p)
          {
          const Col<eT> col_j = out.col(j).rows(p, j - 1);
          R_jj -= dot(col_j, col_j);
          }
        }
      else
        {
        if (j > p)
          {
          const Row<eT> row_j = out.row(j).cols(p, j - 1);
          R_jj -= dot(row_j, row_j);
          }
        }

      // Check for rank deficiency
      if (std::abs(R_jj) < tol)
        {
        excluded(j) = 1;
        continue;
        }

      R_jj = std::sqrt(R_jj);
      out(j, j) = R_jj;
      
      if (j + 1 < N)
        {
        const uword trailing_start = j + 1;
        
        if (sig == 'u')
          {
          // rest = A.row(j).cols(trailing)
          // rest -= PreviousPanelsUpdate
          // rest -= IntraPanelUpdate
          
          if (p > 0)
            {
            // BlockUpdate rows correspond to j in p..end
            // We need row j-p.
            // BlockUpdate cols correspond to p..N-1.
            // We need cols corresponding to trailing_start..N-1.
            // trailing_start is j+1. So offset is (j+1)-p.
            
            // Note: temporary to allow inplace modification
            Row<eT> tmp = A.row(j).cols(trailing_start, N - 1) - BlockUpdate.row(j - p).cols(j + 1 - p, N - 1 - p);
            
            if (j > p)
              {
              // Intra-panel: sum_{k=p}^{j-1} out(k, j) * out(k, trailing)
              // Vector (size j-p) * Matrix (size j-p x trailing_len)
              tmp -= out.col(j).rows(p, j - 1).t() * out.submat(p, trailing_start, j - 1, N - 1);
              }
              
            tmp /= R_jj;
            out.row(j).cols(trailing_start, N - 1) = tmp;
            }
          else
            {
            // p == 0 path (no BlockUpdate)
            Row<eT> tmp = A.row(j).cols(trailing_start, N - 1);
            if (j > p)
              {
              tmp -= out.col(j).rows(p, j - 1).t() * out.submat(p, trailing_start, j - 1, N - 1);
              }
            tmp /= R_jj;
            out.row(j).cols(trailing_start, N - 1) = tmp;
            }
          }
        else // sig == 'l'
          {
          if (p > 0)
            {
            // BlockUpdate size (N-p) x (end-p+1).
            // Col index in BlockUpdate corresponds to j.
            // Row index in BlockUpdate corresponds to global rows p...N-1.
            // We need rows corresponding to trailing_start..N-1.
            // trailing offset: trailing_start - p.
            
            Col<eT> tmp = A.col(j).rows(trailing_start, N - 1) - BlockUpdate.col(j - p).rows(j + 1 - p, N - 1 - p);
            
            if (j > p)
              {
              // Intra-panel: sum_{k=p}^{j-1} out(j, k) * out(trailing, k)
              // Matrix (size trailing_len x j-p) * Vector (size j-p)
              // out(trailing, p...j-1) * out(j, p...j-1).t()
              tmp -= out.submat(trailing_start, p, N - 1, j - 1) * out.row(j).cols(p, j - 1).t();
              }
            
            tmp /= R_jj;
            out.col(j).rows(trailing_start, N - 1) = tmp;
            }
          else
            {
            Col<eT> tmp = A.col(j).rows(trailing_start, N - 1);
            if (j > p)
              {
              tmp -= out.submat(trailing_start, p, N - 1, j - 1) * out.row(j).cols(p, j - 1).t();
              }
            tmp /= R_jj;
            out.col(j).rows(trailing_start, N - 1) = tmp;
            }
          }
        }
      
      panel_accepts.push_back(j);
      }
    
    // Update diagonal contributions for future panels
    const uword trailing_start_after = end + 1;
    if (trailing_start_after < N)
      {
      if (sig == 'u')
        {
        // For upper, we sum squares of columns of the current panel (rows p...end, cols trailing)
        // We need contribution of out(p...end, trailing).
        // A(k,k) -= sum_{i in p..end} out(i, k)^2
        // We sum down the columns i (rows in out).
        // Since out is upper, out(i, k) means row i, col k. i < k.
        // So we take block out.submat(p, trailing_start_after, end, N-1).
        
        const Mat<eT> G = out.submat(p, trailing_start_after, end, N - 1);
        const Row<eT> col_sums = sum(square(G), 0);
        
        for (uword k = 0; k < col_sums.n_elem; ++k)
          {
          diag_contrib[trailing_start_after + k] += col_sums[k];
          }
        }
      else
        {
        // For lower, A(k,k) -= sum_{i in p...end} L(k, i)^2.
        // k is row index (trailing). i is col index (p...end).
        // We take block L(trailing, p...end).
        // We sum squares along rows (dim 1).
        
        const Mat<eT> G = out.submat(trailing_start_after, p, N - 1, end);
        const Col<eT> row_sums = sum(square(G), 1);
        
        for (uword k = 0; k < row_sums.n_elem; ++k)
          {
          diag_contrib[trailing_start_after + k] += row_sums[k];
          }
        }
      }
    
    rank_out += panel_accepts.size();
    }

  // Mark excluded diagonal entries with NaN
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
