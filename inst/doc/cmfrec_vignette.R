## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
    collapse = TRUE,
    comment = "#>"
)
options(rmarkdown.html_vignette.check_title = FALSE)

## ---- include = FALSE---------------------------------------------------------
### Don't overload CRAN servers
### https://stackoverflow.com/questions/28961431/computationally-heavy-r-vignettes
is_check <- ("CheckExEnv" %in% search()) || any(c("_R_CHECK_TIMINGS_",
             "_R_CHECK_LICENSE_") %in% names(Sys.getenv()))

## ---- message=FALSE-----------------------------------------------------------
library(cmfrec)
library(Matrix)
library(MatrixExtra)
library(recommenderlab)

data("MovieLense")
X <- as.coo.matrix(MovieLense@data)
str(X)

## -----------------------------------------------------------------------------
subsample_coo_matrix <- function(X, indices) {
    X@i <- X@i[indices]
    X@j <- X@j[indices]
    X@x <- X@x[indices]
    return(X)
}

n_ratings <- length(X@x)
set.seed(123)
ix_train <- sample(n_ratings, floor(0.75 * n_ratings), replace=FALSE)
X_train <- subsample_coo_matrix(X, ix_train)
X_test <- subsample_coo_matrix(X, -ix_train)

## ---- eval=FALSE--------------------------------------------------------------
#  set.seed(1)
#  model.classic <- CMF(X_train, k=25, lambda=0.1, scale_lam=TRUE, verbose=FALSE)

## ---- echo=FALSE--------------------------------------------------------------
### Don't overload CRAN servers
set.seed(1)
if (!is_check) {
    model.classic <- CMF(X_train, k=25, lambda=0.1, scale_lam=TRUE, verbose=FALSE)
} else {
    model.classic <- CMF(X_train, k=5, lambda=0.1, scale_lam=TRUE, verbose=FALSE,
                         niter=2, nthreads=1)
}

## -----------------------------------------------------------------------------
print_rmse <- function(X_test, X_hat, model_name) {
  rmse <- sqrt(mean( (X_test@x - X_hat@x)^2 ))
  cat(sprintf("RMSE for %s is: %.4f\n", model_name, rmse))
}

pred_classic <- predict(model.classic, X_test)
print_rmse(X_test, pred_classic, "classic model")

## -----------------------------------------------------------------------------
model.baseline <- MostPopular(X_train, lambda=10, scale_lam=FALSE)
pred_baseline <- predict(model.baseline, X_test)
print_rmse(X_test, pred_baseline, "non-personalized model")

## ---- eval=FALSE--------------------------------------------------------------
#  set.seed(1)
#  model.improved <- CMF(X_train, k=25, lambda=0.1, scale_lam=TRUE,
#                        add_implicit_features=TRUE, w_main=0.75, w_implicit=0.25,
#                        use_cg=FALSE, niter=30, verbose=FALSE)
#  pred_improved <- predict(model.improved, X_test)
#  print_rmse(X_test, pred_improved, "improved classic model")

## ---- echo=FALSE--------------------------------------------------------------
### Don't overload CRAN servers
set.seed(1)
if (!is_check) {
    model.improved <- CMF(X_train, k=25, lambda=0.1, scale_lam=TRUE,
                          add_implicit_features=TRUE, w_main=0.75, w_implicit=0.25,
                          use_cg=FALSE, niter=30, verbose=FALSE)
} else {
   model.improved <- CMF(X_train, k=5, lambda=0.1, scale_lam=TRUE,
                         add_implicit_features=TRUE, w_main=0.75, w_implicit=0.25,
                         use_cg=FALSE, verbose=FALSE,
                         niter=2, nthreads=1)
}
pred_improved <- predict(model.improved, X_test)
print_rmse(X_test, pred_improved, "improved classic model")

## -----------------------------------------------------------------------------
U <- MovieLenseUser
U$id      <- NULL
U$zipcode <- NULL
U$age2    <- U$age^2
### Note that `cmfrec` does not standardize features beyond mean centering
U$age     <- (U$age - mean(U$age)) / sd(U$age)
U$age2    <- (U$age2 - mean(U$age2)) / sd(U$age2)
U <- model.matrix(~.-1, data=U)

I <- MovieLenseMeta
I$title <- NULL
I$url   <- NULL
I$year  <- ifelse(is.na(I$year), median(I$year, na.rm=TRUE), I$year)
I$year2 <- I$year^2
I$year  <- (I$year - mean(I$year)) / sd(I$year)
I$year2 <- (I$year2 - mean(I$year2)) / sd(I$year2)
I <- as.coo.matrix(I)

cat(dim(U), "\n")
cat(dim(I), "\n")

## ---- eval=FALSE--------------------------------------------------------------
#  set.seed(1)
#  model.w.sideinfo <- CMF(X_train, U=U, I=I, NA_as_zero_item=TRUE,
#                          k=25, lambda=0.1, scale_lam=TRUE,
#                          niter=30, use_cg=FALSE, include_all_X=FALSE,
#                          w_main=0.75, w_user=0.5, w_item=0.5, w_implicit=0.5,
#                          verbose=FALSE)
#  pred_side_info <- predict(model.w.sideinfo, X_test)
#  print_rmse(X_test, pred_side_info, "model with side info")

## ---- echo=FALSE--------------------------------------------------------------
### Don't overload CRAN servers
set.seed(1)
if (!is_check) {
    model.w.sideinfo <- CMF(X_train, U=U, I=I, NA_as_zero_item=TRUE,
                            k=25, lambda=0.1, scale_lam=TRUE,
                            niter=30, use_cg=FALSE, include_all_X=FALSE,
                            w_main=0.75, w_user=0.5, w_item=0.5, w_implicit=0.5,
                            verbose=FALSE)
} else {
    model.w.sideinfo <- CMF(X_train, U=U, I=I, NA_as_zero_item=TRUE,
                            k=5, lambda=0.1, scale_lam=TRUE, scale_lam_sideinfo=TRUE,
                            use_cg=FALSE, include_all_X=FALSE,
                            w_main=0.75, w_user=0.5, w_item=0.5, w_implicit=0.5,
                            verbose=FALSE, niter=2, nthreads=1)
}
pred_side_info <- predict(model.w.sideinfo, X_test)
print_rmse(X_test, pred_side_info, "model with side info")

## -----------------------------------------------------------------------------
library(kableExtra)

calc_rmse <- function(X_test, X_hat) {
    return(sqrt(mean( (X_test@x - X_hat@x)^2 )))
}
results <- data.frame(
    NonPersonalized = calc_rmse(X_test, pred_baseline),
    ClassicalModel = calc_rmse(X_test, pred_classic),
    ClassicPlusImplicit = calc_rmse(X_test, pred_improved),
    CollectiveModel = calc_rmse(X_test, pred_side_info)
)
results <- as.data.frame(t(results))
names(results) <- "RMSE"
results %>%
    kable() %>%
    kable_styling()

## ---- eval=FALSE--------------------------------------------------------------
#  ### Re-fitting the earlier model to all the data,
#  ### this time *without* scaled regularization
#  set.seed(1)
#  model.classic <- CMF(X, k=20, lambda=10, scale_lam=FALSE, verbose=FALSE)
#  set.seed(1)
#  model.w.sideinfo <- CMF(X, U=U, I=I, k=20, lambda=10, scale_lam=FALSE,
#                          w_main=0.75, w_user=0.125, w_item=0.125,
#                          verbose=FALSE)

## ---- echo=FALSE--------------------------------------------------------------
### Don't overload CRAN servers
if (!is_check) {
    set.seed(1)
    model.classic <- CMF(X, k=20, lambda=10, scale_lam=FALSE, verbose=FALSE)
    set.seed(1)
    model.w.sideinfo <- CMF(X, U=U, I=I, k=20, lambda=10, scale_lam=FALSE,
                            w_main=0.75, w_user=0.125, w_item=0.125,
                            verbose=FALSE)
} else {
    set.seed(1)
    model.classic <- CMF(X, k=5, lambda=10, scale_lam=FALSE, verbose=FALSE,
                         niter=2, nthreads=1)
    set.seed(1)
    model.w.sideinfo <- CMF(X, U=U, I=I, k=5, lambda=10, scale_lam=FALSE,
                            w_main=0.75, w_user=0.125, w_item=0.125,
                            verbose=FALSE, niter=2, nthreads=1)
}

## -----------------------------------------------------------------------------
user_to_recommend <- 10
### Note: slicing of 'X' is provided by 'MatrixExtra',
### returning a 'sparseVector' object as required by cmfrec
topN(model.classic, user=user_to_recommend, n=10,
     exclude=X[user_to_recommend, , drop=TRUE])

## -----------------------------------------------------------------------------
### A handy function for visualizing recommendations
movie_names <- colnames(X)
n_ratings <- colSums(as.csc.matrix(X, binary=TRUE))
avg_ratings <- colSums(as.csc.matrix(X)) / n_ratings
print_recommended <- function(rec, txt) {
    cat(txt, ":\n",
        paste(paste(1:length(rec), ". ", sep=""),
              movie_names[rec],
              " - Avg rating:", round(avg_ratings[rec], 2),
              ", #ratings: ", n_ratings[rec],
              collapse="\n", sep=""),
        "\n", sep="")
}
recommended <- topN(model.w.sideinfo, user=user_to_recommend, n=5,
                    exclude=X[user_to_recommend, , drop=TRUE])
print_recommended(recommended, "Recommended for user_id=10")

## -----------------------------------------------------------------------------
recommended_new <- topN_new(model.w.sideinfo, n=5,
                            exclude=X[user_to_recommend, , drop=TRUE],
                            X=X[user_to_recommend, , drop=TRUE],
                            U=U[user_to_recommend, , drop=TRUE])
print_recommended(recommended_new, "Recommended for user_id=10 as new user")

## -----------------------------------------------------------------------------
recommended_new <- topN_new(model.w.sideinfo, n=5,
                            exclude=X[user_to_recommend, , drop=TRUE],
                            X=X[user_to_recommend, , drop=TRUE])
print_recommended(recommended_new, "Recommended for user_id=10 as new user (NO sideinfo)")

## -----------------------------------------------------------------------------
recommended_cold <- topN_new(model.w.sideinfo, n=5,
                             exclude=X[user_to_recommend, , drop=TRUE],
                             U=U[user_to_recommend, , drop=TRUE])
print_recommended(recommended_cold, "Recommended for user_id=10 as new user (NO ratings)")

