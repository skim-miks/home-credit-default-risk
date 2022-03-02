library(data.table)
library(caret)
library(tidyverse)
library(pROC)

train <- fread("application_train.csv")
test <- fread("application_test.csv")

str(train)
hist(train$TARGET)
summary(train)
table(train$TARGET)

train$NAME_CONTRACT_TYPE <- as.numeric(factor(train$NAME_CONTRACT_TYPE, levels=unique(train$NAME_CONTRACT_TYPE)))
train$CODE_GENDER <- ifelse(train$CODE_GENDER == "F", 1, 0)
train$FLAG_OWN_CAR <- ifelse(train$FLAG_OWN_CAR == "Y", 1, 0)
train$FLAG_OWN_REALTY <- ifelse(train$FLAG_OWN_REALTY == "Y", 1, 0)
train$NAME_TYPE_SUITE <-  as.numeric(factor(train$NAME_TYPE_SUITE, levels=unique(train$NAME_TYPE_SUITE)))
train$NAME_INCOME_TYPE <-  as.numeric(factor(train$NAME_INCOME_TYPE, levels=unique(train$NAME_INCOME_TYPE))) 
train$NAME_EDUCATION_TYPE <-  as.numeric(factor(train$NAME_EDUCATION_TYPE, levels=unique(train$NAME_EDUCATION_TYPE)))
train$NAME_FAMILY_STATUS <-  as.numeric(factor(train$NAME_FAMILY_STATUS, levels=unique(train$NAME_FAMILY_STATUS)))
train$NAME_HOUSING_TYPE <- as.numeric(factor(train$NAME_HOUSING_TYPE, levels=unique(train$NAME_HOUSING_TYPE)))
train$OCCUPATION_TYPE <- as.numeric(factor(train$OCCUPATION_TYPE, levels=unique(train$OCCUPATION_TYPE)))
train$WEEKDAY_APPR_PROCESS_START <- as.numeric(factor(train$WEEKDAY_APPR_PROCESS_START, levels=unique(train$WEEKDAY_APPR_PROCESS_START)))
train$ORGANIZATION_TYPE <- as.numeric(factor(train$ORGANIZATION_TYPE, levels=unique(train$ORGANIZATION_TYPE)))
train$FONDKAPREMONT_MODE <- as.numeric(factor(train$FONDKAPREMONT_MODE, levels=unique(train$FONDKAPREMONT_MODE)))
train$HOUSETYPE_MODE <- as.numeric(factor(train$HOUSETYPE_MODE, levels=unique(train$HOUSETYPE_MODE)))
train$WALLSMATERIAL_MODE <-  as.numeric(factor(train$WALLSMATERIAL_MODE, levels=unique(train$WALLSMATERIAL_MODE)))
train$EMERGENCYSTATE_MODE <- ifelse(train$EMERGENCYSTATE_MODE == "Yes", 1, 0)

test$NAME_CONTRACT_TYPE <- as.numeric(factor(test$NAME_CONTRACT_TYPE, levels=unique(test$NAME_CONTRACT_TYPE)))
test$CODE_GENDER <- ifelse(test$CODE_GENDER == "F", 1, 0)
test$FLAG_OWN_CAR <- ifelse(test$FLAG_OWN_CAR == "Y", 1, 0)
test$FLAG_OWN_REALTY <- ifelse(test$FLAG_OWN_REALTY == "Y", 1, 0)
test$NAME_TYPE_SUITE <-  as.numeric(factor(test$NAME_TYPE_SUITE, levels=unique(test$NAME_TYPE_SUITE)))
test$NAME_INCOME_TYPE <-  as.numeric(factor(test$NAME_INCOME_TYPE, levels=unique(test$NAME_INCOME_TYPE))) 
test$NAME_EDUCATION_TYPE <-  as.numeric(factor(test$NAME_EDUCATION_TYPE, levels=unique(test$NAME_EDUCATION_TYPE)))
test$NAME_FAMILY_STATUS <-  as.numeric(factor(test$NAME_FAMILY_STATUS, levels=unique(test$NAME_FAMILY_STATUS)))
test$NAME_HOUSING_TYPE <- as.numeric(factor(test$NAME_HOUSING_TYPE, levels=unique(test$NAME_HOUSING_TYPE)))
test$OCCUPATION_TYPE <- as.numeric(factor(test$OCCUPATION_TYPE, levels=unique(test$OCCUPATION_TYPE)))
test$WEEKDAY_APPR_PROCESS_START <- as.numeric(factor(test$WEEKDAY_APPR_PROCESS_START, levels=unique(test$WEEKDAY_APPR_PROCESS_START)))
test$ORGANIZATION_TYPE <- as.numeric(factor(test$ORGANIZATION_TYPE, levels=unique(test$ORGANIZATION_TYPE)))
test$FONDKAPREMONT_MODE <- as.numeric(factor(test$FONDKAPREMONT_MODE, levels=unique(test$FONDKAPREMONT_MODE)))
test$HOUSETYPE_MODE <- as.numeric(factor(test$HOUSETYPE_MODE, levels=unique(test$HOUSETYPE_MODE)))
test$WALLSMATERIAL_MODE <-  as.numeric(factor(test$WALLSMATERIAL_MODE, levels=unique(test$WALLSMATERIAL_MODE)))
test$EMERGENCYSTATE_MODE <- ifelse(test$EMERGENCYSTATE_MODE == "Yes", 1, 0)


target = train$TARGET
train$TARGET = NULL

train$EXT_SOURCE_COMB <- train$EXT_SOURCE_1 + train$EXT_SOURCE_2
train$EXT_SOURCE_COMB1 <- abs(train$EXT_SOURCE_1 - train$EXT_SOURCE_2)
train$EXT_SOURCE_COMB2 <- (train$EXT_SOURCE_1+train$EXT_SOURCE_2)/2
train <- train %>%
  select(-SK_ID_CURR)

test <- test %>%
  mutate(EXT_SOURCE_COMB = EXT_SOURCE_1 + EXT_SOURCE_2,
         EXT_SOURCE_COMB1 = abs(EXT_SOURCE_1 - EXT_SOURCE_2),
         EXT_SOURCE_COMB2 = (EXT_SOURCE_1 + EXT_SOURCE_2)/2)

nrounds = 5
folds <- createFolds(target, k=5, list=FALSE)
library(xgboost)
result <- rep(0, nrow(train))
for(this.round in 1:nrounds) {
  valid <- c(1:length(target)) [folds == this.round]
  dev <- c(1:length(target)) [folds != this.round]
  
  dtrain <- xgb.DMatrix(data=as.matrix(train[dev,]),
                        label=target[dev])

  dvalid <- xgb.DMatrix(data=as.matrix(train[valid,]),
                        label=target[valid])
  param = list(objective="binary:logistic",
               eval_metric="auc",
               max_depth=3,
               eta=.02,
               booster="gbtree"
               )
  
  model <- xgb.train(data=dtrain, params=param, nrounds=20000, list(val1=dtrain, val2=dvalid),
                     early_stopping_rounds = 250, print_every_n = 100)
  pred = predict(model, as.matrix(train[valid,]))
  result[valid] = pred
}
auc(target, result)
importance <- xgb.importance(colnames(train), model=model)
xgb.plot.importance(importance, top_n=30)

te_pred <- predict(model, as.matrix(test[,2:ncol(test)]), type="response")
sub <- data.frame(test[,1],te_pred)
colnames(sub) <- c("SK_ID_CURR", "TARGET")

write.csv(sub, file="\submission.csv")
