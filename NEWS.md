# 15.0.2 - Mate Taragui

I added an UNOFFICIAL Cholesky decomposition with rank revealing capabilities.

What it does is:

1. Basic rank-revealing Cholesky:

```cpp
template<typename T1>
bool chol_rank(Mat<typename T1::elem_type>& out,
               const Base<typename T1::elem_type,T1>& X,
               uword& rank_out,
               const char* layout = "upper",
               const typename T1::elem_type tol = typename T1::elem_type(1e-12))
```

2. Same as (1) with excluded variables tracking:

```cpp
template<typename T1>
bool chol_rank(Mat<typename T1::elem_type>& out,
               Col<uword>& excluded,
               const Base<typename T1::elem_type,T1>& X,
               uword& rank_out,
               const char* layout = "upper",
               const typename T1::elem_type tol = typename T1::elem_type(1e-12))
```

In other words, this is useful to identify which columns/variables are linearly dependent (e.g., useful for Econometrics' problems of multicollinearity and an `X.t() * X` does does not have full rank). `tol` allows to set a threshold to determine when a diagonal element is considered zero to determine if it counts / does not count in the rank.

Here is how I tested it:

Save this as `./chol_rank_demo.cpp` and compile with `g++ chol_rank_demo.cpp -o chol_rank_demo -std=c++14 -O0 -I ./include -DARMA_DONT_USE_WRAPPER -lopenblas` and run with `./chol_rank_demo`.

```cpp
#include <iostream>
#include <armadillo>

using namespace arma;

int main()
{
    std::cout << "Armadillo rank-revealing Cholesky decomposition demo\n";
    std::cout << "====================================================\n\n";
    
    // Create a test case similar to your use case
    std::cout << "Creating a 5x5 test matrix...\n";
    mat A(5, 5, fill::randu);
    mat X = A.t() * A;  // Positive semi-definite matrix
    
    std::cout << "Original matrix X:\n";
    X.print();
    
    // Test the new chol_rank functions
    std::cout << "\n1. Basic rank-revealing Cholesky (upper triangular):\n";
    mat R1;
    uword rank1;
    bool ok1 = chol_rank(R1, X, rank1);
    std::cout << "Success: " << ok1 << ", Rank: " << rank1 << "\n";
    std::cout << "R1:\n";
    R1.print();
    
    std::cout << "\n2. Rank-revealing Cholesky (lower triangular):\n";
    mat R2;
    uword rank2;
    bool ok2 = chol_rank(R2, X, rank2, "lower");
    std::cout << "Success: " << ok2 << ", Rank: " << rank2 << "\n";
    std::cout << "R2:\n";
    R2.print();
    
    std::cout << "\n3. With excluded variables tracking:\n";
    mat R3;
    Col<uword> excluded;
    uword rank3;
    bool ok3 = chol_rank(R3, excluded, X, rank3);
    std::cout << "Success: " << ok3 << ", Rank: " << rank3 << "\n";
    std::cout << "Excluded variables: ";
    excluded.t().print();
    
    // Test with a rank-deficient matrix
    std::cout << "\n4. Testing with rank-deficient matrix:\n";
    mat B = {{4, 2, 1}, {2, 1, 0.5}, {1, 0.5, 0.25}};  // Rank 1 matrix
    std::cout << "Rank-deficient matrix B:\n";
    B.print();
    
    mat R4;
    Col<uword> excluded4;
    uword rank4;
    bool ok4 = chol_rank(R4, excluded4, B, rank4, "upper", 1e-10);
    std::cout << "Success: " << ok4 << ", Rank: " << rank4 << "/3\n";
    std::cout << "Excluded variables: ";
    excluded4.t().print();
    std::cout << "Decomposition R4:\n";
    R4.print();
    
    std::cout << "\nDemo completed successfully!\n";
    
    return 0;
}
```
