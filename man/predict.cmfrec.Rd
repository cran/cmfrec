% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/predict.R
\name{predict.cmfrec}
\alias{predict.cmfrec}
\title{Predict entries in the factorized `X` matrix}
\usage{
\method{predict}{cmfrec}(object, user, item = NULL, nthreads = object$info$nthreads, ...)
}
\arguments{
\item{object}{A collective matrix factorization model from this package - see
\link{fit_models} for details.}

\item{user}{The user IDs for which to make predictions. If `X` to which the model
was fit was a `data.frame`, should pass IDs matching to the first column of `X`
(the user indices, should be a character vector),
otherwise should pass row numbers for `X`, with numeration
starting at 1 (should be an integer vector).

If passing a single entry for `user` and `item` has more entries, will
predict all the entries in `item` for that single `user.`

Alternatively, might instead pass a sparse matrix in COO/triplets formats,
for which the \bold{non-missing} entries will be predicted, in which case it
its not necessary to pass `item`.

If passing a sparse matrix, can be from package `Matrix` (class `dgTMatrix` or `ngTMatrix`)
or from package `SparseM` (class `matrix.coo`). If using the package `softImpute`,
its objects of class `incomplete` might be convertable to `Matrix` objects through
e.g. `as(as(X, "TsparseMatrix"), "nMatrix")`.}

\item{item}{The item IDs for which to make predictions - see the documentation
about `user` for details about the indexing.

If passing a single entry for `item` and `user` has more entries, will
predict all the entries in `user` for that single `item`.

If passing a sparse matrix as `user`, `item` will be ignored.}

\item{nthreads}{Number of parallel threads to use.}

\item{...}{Not used.}
}
\value{
A numeric vector with the predicted values at the requested combinations.
If the `user` passed was a sparse matrix, and it was not of class `ngTMatrix`,
will instead return a sparse matrix of the same format, with the non-missing entries
set to the predicted values.
}
\description{
Predict entries in the `X` matrix according to the model
at the combinations [row,column] given by the entries in
`user` and `item` (e.g. passing `user=c(1,2,3), item=c(1,1,1)` will predict
X[1,1], X[2,1], X[3,1]).

Alternatively, might pass a sparse matrix, in which case it will make
predictions for all of its non-missing entries.

Invalid combinations (e.g. rows and columns outside of the range of `X` to
which the model was fit) will be filled with global mean plus biases if applicable
for `CMF_explicit`, and with NAs for the other models.

For example usage, see the main section \link{fit_models}.
}
\seealso{
\link{predict_new} \link{topN}
}
