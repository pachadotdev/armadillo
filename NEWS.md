# armadillo 0.2.9

* Removes leontief dependency to test the functions.
* Adds minimal documentation to use OpenBLAS with R and Armadillo.
* New examples, which are clearer and more informative.

# armadillo 0.2.8

* Provides templates to convert to and from sparse matrices.
* Removes `using namespace std;` from headers (see https://www.reddit.com/r/cpp_questions/comments/160eivk/is_using_namespace_std_really_considered_bad/)

# armadillo 0.2.7

* Provides wrappers for `arma::uvec` used to subset vectors.
* Minimal optimizations in R to/from C++ templates.

# armadillo 0.2.6

* Fewer implicit conversions.
* Using balanced parallelization in OpenMP.

# armadillo 0.2.5

* Uses messages that do not generate warnings in the R API (#379d8d6).
* Skips OpenMP on Mac hardware (#13e805b).
* Provides a template to convert vectors to column matrices (#6138a35),
* First version on CRAN.

# armadillo 0.2.0

* Uses OpenMP.
* Sticks to Clang format.

# armadillo 0.1.2

* Improves vendoring (i.e., does the same as `cpp11`)

# armadillo 0.1.1

* Includes more formal tests in the `armadillotest` directory.
* Provides a conversion from complex vector/matrix to a list of double
  vectors/matrices.

# armadillo 0.1

* First public version. Elemental vector/matrix conversion from/to R and C++.
