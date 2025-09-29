########################################################################################################################

# Light GBM 

library(lightgbm)
library(caret)
library(dplyr)
set.seed(2025)
X_train = read.csv("X.csv")
y_train = read.csv("y.csv")

full_X = read.csv("full_X.csv")
test_oppur = read.csv("test_clean.csv")
oppur_mat = cbind(full_X$Opportunity, full_X$max_progress)
oppur_mat = oppur_mat[!duplicated(oppur_mat), , drop = FALSE]

test = read.csv("test.csv")

X_train_all_vars = X_train
test_all_vars = test

uni_ranks = read.csv("uni_ranks.csv")

uni_counts = X_train %>% count(Institution_encoded) %>% filter(n>250)

#remove other 
uni_counts = uni_counts[-4,]

#create mapping from rank to encoding
uni_ranks_encoded = uni_ranks

uni_ranks$ranks = scale(uni_ranks$ranks, scale = T, center = F)

found = FALSE

for (i in 1:nrow(X_train)){
  
  for (j in 1:nrow(uni_counts)){
    if (X_train$Institution_encoded[i] == uni_counts$Institution_encoded[j]){
      
      for (k in 1:nrow(uni_ranks)){
        if (uni_counts$n[j] == uni_ranks$n[k]){
          X_train$Institution_encoded[i] = uni_ranks$ranks[k]
          found = T
        }
      }
      
    }
  }
  
  if (found == F){ #replace other encoding with 0 ranking 
    
    X_train$Institution_encoded[i] = 0
  }
  found = FALSE
}


## test set 

found = FALSE

for (i in 1:nrow(test)){
  
  for (j in 1:nrow(uni_counts)){
    if (test$Institution_encoded[i] == uni_counts$Institution_encoded[j]){
      
      for (k in 1:nrow(uni_ranks)){
        if (uni_counts$n[j] == uni_ranks$n[k]){
          test$Institution_encoded[i] = uni_ranks$ranks[k]
          found = T
        }
      }
      
    }
  }
  
  if (found == F){ #replace other encoding with 0 ranking 
    
    test$Institution_encoded[i] = 0
  }
  found = FALSE
}

#remove quali

# X_train = X_train[,-9]
# test = test[,-9]
# 
# # remove company 
# 
# X_train = X_train[,-5]
# test = test[,-5]

# Assume: train_df has Progress_normalized in [0,1]; test_df has no labels
y <- as.matrix(y_train)#train_df$Progress_normalized
X <- as.matrix(X_train)#train_df[setdiff(names(train_df), "Progress_normalized")]
Xtest <- test#test_df

y_fac = as.factor(y)
label <- as.integer(y_fac) - 1L          # 0..K-1 for LightGBM
K <- max(label) + 1#nlevels(y_fac)

#categorical columns 

cat_cols = c("Industry_encoded", "Company_encoded", "Gender_encoded", "Race_encoded", "Qualification_encoded") # "Institution_encoded",
cat_idx  <- which(colnames(X_train) %in% cat_cols)

dtrain = lgb.Dataset(X, label = y, categorical_feature = cat_cols, 
                     params = list(max_bin = 511, feature_pre_filter = FALSE)) #label #511

folds <- createFolds(cut(y, breaks = 10), k = 10, returnTrain = FALSE)

#--------- Random grid ---------
grid <- expand.grid(
  min_data_in_leaf  = c( 10, 25, 40),
  learning_rate     = c(0.04,0.03, 0.02, 0.01),
  num_leaves        = c(180,190, 200, 210),#c(90, 140, 170),
  feature_fraction  = c( 0.7, 0.8, 0.9, 1.0),
  bagging_fraction  = c( 0.8, 0.9, 1.0),
  lambda_l1         = c(0.1, 0.25, 0.5, 1),#c(0.0, 0.5, 1.5, 2),
  lambda_l2         = c(0.0, 0.5, 1.0, 1.5),
  min_gain_to_split = c(0.0, 0.02, 0.05)
)

set.seed(2025)
grid <- grid[sample(nrow(grid), min(70, nrow(grid))), ]

best <- list(score = Inf)

for (i in seq_len(nrow(grid))) {
  g <- grid[i, ]
  
  params <- list(
    objective          = "regression",
    metric             = "rmse",
    learning_rate      = g$learning_rate,
    num_leaves         = g$num_leaves,
    min_data_in_leaf   = g$min_data_in_leaf,
    feature_fraction   = g$feature_fraction,
    bagging_fraction   = g$bagging_fraction,
    bagging_freq       = ifelse(g$bagging_fraction < 1.0, 1L, 0L),
    lambda_l1          = g$lambda_l1,
    lambda_l2          = g$lambda_l2,
    feature_pre_filter = FALSE,
    min_gain_to_split  = g$min_gain_to_split,
    max_bin            = 511,
    verbosity          = -1
  )
  
  cv <- lgb.cv(
    params = params,
    data = dtrain,
    folds = folds,
    nrounds = 50000,
    early_stopping_rounds = 300,
    verbose = 0
  )
  
  rmse <- cv$best_score
  it   <- cv$best_iter
  
  if (rmse < best$score) {
    best <- list(score = rmse, iter = it, params = params)
  }
}

best$params; best$score; best$iter

# --------- Refit on full training with best params/iter ---------


fit <- lgb.train(
  params  = best$params,
  data    = dtrain,
  nrounds = best$iter,
  verbose = 0)#,


###################################

#parallel

#X_train_mat = as.matrix(X_train)
#y_train_mat = as.matrix(y_train) 

library(doParallel); library(foreach)
# --- Parallel backend ---
n_workers <- max(1, parallel::detectCores() - 2)
cl <- makeCluster(n_workers)
registerDoParallel(cl)

# --- Parallel evaluation of grid combos ---
results <- foreach(i = seq_len(nrow(grid)), .packages = c("lightgbm")) %dopar% {
  g <- grid[i, ]
  
  # Build Dataset INSIDE the worker (prevents the "no raw data" error)
  dtrain <- lgb.Dataset(
    data  = X,#X_train_mat,
    label = y,#y_train_mat,
    categorical_feature = cat_idx,        # pass indices or names
    params = list(
      feature_pre_filter = FALSE,         # so we can vary min_data_in_leaf etc.
      max_bin = 511
    )
  )
  
  params <- list(
    objective          = "regression",
    metric             = "rmse",
    learning_rate      = g$learning_rate,
    num_leaves         = g$num_leaves,
    min_data_in_leaf   = g$min_data_in_leaf,
    feature_fraction   = g$feature_fraction,
    bagging_fraction   = g$bagging_fraction,
    bagging_freq       = ifelse(g$bagging_fraction < 1.0, 1L, 0L),
    lambda_l1          = g$lambda_l1,
    lambda_l2          = g$lambda_l2,
    min_gain_to_split  = g$min_gain_to_split,
    feature_pre_filter = FALSE,
    max_bin            = 511,
    verbosity          = -1,
    num_threads        = 1,               # <- 1 thread per worker
    seed               = 2025 + i
  )
  
  cv <- lgb.cv(
    params = params,
    data   = dtrain,
    folds  = folds,
    nrounds = 50000,
    early_stopping_rounds = 300,
    verbose = 0
  )
  
  list(score = cv$best_score, iter = cv$best_iter, params = params)
}

stopCluster(cl)

# --- Pick best ---
scores <- vapply(results, `[[`, numeric(1), "score")
best_i <- which.min(scores)
best   <- results[[best_i]]

best$params
best$score
best$iter

# --- Final refit on full training (single process) ---
dtrain_final <- lgb.Dataset(
  data  = X,#X_train_mat,
  label = y,#y_train_mat,
  categorical_feature = cat_idx,
  params = list(feature_pre_filter = FALSE, max_bin = 511)
)

fit <- lgb.train(
  params  = best$params,
  data    = dtrain_final,
  nrounds = best$iter,
  verbose = 0
)


# Result
summary(predict(fit, X))

training_error = sum(abs(y - predict(fit,X)))/nrow(test)

yhat = predict(fit, X)

summary(predict(fit, as.matrix(test)))

lightgbm_pred = predict(fit, as.matrix(test))

lightgbm_pred = pmax(0,pmin(1,lightgbm_pred))

lightgbm_pred_actual = c()

for (i in 1:34512){
  oppur = test_oppur$Opportunity[i]
  max_p_pos = which(oppur_mat[,1] == oppur) 
  yhat = 1 + lightgbm_pred[i] * (oppur_mat[max_p_pos, 2] - 1)
  lightgbm_pred_actual = c(lightgbm_pred_actual, yhat )
}


submission = data.frame("ID" = c(0:34511), 'Progress' = round(lightgbm_pred_actual)) #round the predictions to get progress

write.csv(submission, "CIP8.csv", row.names = FALSE)
