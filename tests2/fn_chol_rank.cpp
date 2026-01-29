// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2025 Your Name
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

#include <armadillo>
#include "catch.hpp"
#include "utils.hpp"

using namespace arma;

TEST_CASE("chol_rank_basic", "[chol_rank]")
{
    mat A(5, 5, fill::randu);
    mat X = A.t() * A;

    mat R1 = chol(X);
    mat R2 = chol(X, "lower");

    mat R3;
    bool ok = chol(R3, X);
    REQUIRE(ok == true);
    REQUIRE(approx_equal(R1, R3, "absdiff", 1e-10));

    mat R;
    uvec P_vec;
    umat P_mat;

    chol(R, P_vec, X, "upper", "vector");
    REQUIRE(approx_equal(R, R1, "absdiff", 1e-10));

    chol(R, P_mat, X, "lower", "matrix");
    REQUIRE(approx_equal(R, R2, "absdiff", 1e-10));

    // Test chol_rank() upper
    mat R_rank_upper;
    uvec excluded_upper;
    uword rank_upper = 0;
    bool ok_rank_upper = chol_rank(R_rank_upper, excluded_upper, rank_upper, X, "upper");
    REQUIRE(ok_rank_upper == true);
    REQUIRE(rank_upper == X.n_rows);
    REQUIRE(approx_equal(R1, R_rank_upper, "absdiff", 1e-10));
    REQUIRE(excluded_upper.n_elem == X.n_rows);
    REQUIRE(accu(excluded_upper) == 0);

    // Test chol_rank() lower
    mat R_rank_lower;
    uvec excluded_lower;
    uword rank_lower = 0;
    bool ok_rank_lower = chol_rank(R_rank_lower, excluded_lower, rank_lower, X, "lower");
    REQUIRE(ok_rank_lower == true);
    REQUIRE(rank_lower == X.n_rows);
    REQUIRE(approx_equal(R2, R_rank_lower, "absdiff", 1e-10));
    REQUIRE(excluded_lower.n_elem == X.n_rows);
    REQUIRE(accu(excluded_lower) == 0);
}

TEST_CASE("chol_rank_edge_cases", "[chol_rank]")
{
    // Rank-deficient matrix (last row is zero)
    mat X = eye<mat>(4,4);
    X.row(3).zeros();

    mat R;
    uvec excluded;
    uword rank = 0;
    bool ok = chol_rank(R, excluded, rank, X, "upper");
    REQUIRE(ok == true);
    REQUIRE(rank == 3); // Only 3 nonzero pivots
    REQUIRE(excluded.n_elem == 4);
    REQUIRE(excluded(3) != 0);
    // Off-diagonals in excluded row/col should be zero and diagonal should be NaN
    for (uword i = 0; i < 4; ++i) {
        if (i != 3) {
            REQUIRE(std::abs(R(3, i)) < 1e-10);
            REQUIRE(std::abs(R(i, 3)) < 1e-10);
        }
    }
    REQUIRE(std::isnan(R(3,3)));

    // Empty matrix
    mat X_empty;
    mat R_empty;
    uvec excluded_empty;
    uword rank_empty = 42; // should be reset
    bool ok_empty = chol_rank(R_empty, excluded_empty, rank_empty, X_empty, "upper");
    REQUIRE(ok_empty == true);
    REQUIRE(rank_empty == 0);
    REQUIRE(R_empty.is_empty());
    REQUIRE(excluded_empty.is_empty());

    // Symmetric but not positive-definite
    mat X_sympd = eye<mat>(3,3);
    X_sympd(2,2) = -1.0; // negative eigenvalue
    mat R_sympd;
    uvec excluded_sympd;
    uword rank_sympd = 0;
    bool ok_sympd = chol_rank(R_sympd, excluded_sympd, rank_sympd, X_sympd, "upper");
    REQUIRE(ok_sympd == true);
    REQUIRE(rank_sympd == 2); // Only two positive diagonal elements
    REQUIRE(excluded_sympd.n_elem == 3);
    REQUIRE(excluded_sympd(2) != 0);
    REQUIRE(std::isnan(R_sympd(2,2)));

    // Test with lower layout for rank-deficient
    mat R_lower;
    uvec excluded_lower2;
    uword rank_lower = 0;
    bool ok_lower = chol_rank(R_lower, excluded_lower2, rank_lower, X, "lower");
    REQUIRE(ok_lower == true);
    REQUIRE(rank_lower == 3);
    REQUIRE(excluded_lower2(3) != 0);
    REQUIRE(std::isnan(R_lower(3,3)));
}
