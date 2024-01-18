library(jsonlite)
library(dplyr) 
library(data.table)

load('nhanes_data.rda')
load('nhanes_key.rda')
data = nhanes_data


# nhanes to .json 

set.seed(123)

data = nhanes_data %>% filter(htn_jnc7=='Yes')
n = round(0.1*nrow(data))

data = data[sample(1:nrow(data), nrow(data)), ] %>% as.data.frame(check.names=F)

input_cols = colnames(data)[c(2:9,11:13,30:51,60:62,65,67,69,82:101,102:111)]
keys = data.table(variable = input_cols) %>% left_join(nhanes_key %>% select(variable,label))
input <- data[,keys$variable] 

#fwrite(cbind(data.table(bp_uncontrolled_140_90=data$bp_uncontrolled_140_90),input)[1:n,], file = "./with_chol/nhanes_input_test.csv", row.names = FALSE,col.names=TRUE,sep=',',quote=FALSE)
#fwrite(cbind(data.table(bp_uncontrolled_140_90=data$bp_uncontrolled_140_90),input)[(n+1):(2*n),], file = "./with_chol/nhanes_input_validation.csv", row.names = FALSE,col.names=TRUE,sep=',',quote=FALSE)
#fwrite(cbind(data.table(bp_uncontrolled_140_90=data$bp_uncontrolled_140_90),input)[(2*n+1):nrow(input),], file = "./with_chol/nhanes_input_train.csv", row.names = FALSE,col.names=TRUE,sep=',',quote=FALSE)

colnames(input) <- keys$label

input_new = input
for(i in 1:ncol(input_new)){
    input_new[,i] = paste0(colnames(input_new)[i], ': ', input_new[,i])
}
input_new$string_input = apply(input_new, 1, paste, collapse = ', ')
input_new$string_input_reason = paste0('Suppose I give you the following person information:',input_new$string_input, '. Determin whether this person has hypertension and explain the reason for your answer. Reason:')
input_new$string_input = paste0(input_new$string_input, '. Does this patient have uncontrolled blood pressure (SBP ≥ 140 mm Hg or DBP ≥ 90 mm Hg)?')
output <- paste0(data$bp_uncontrolled_140_90,'.')

to_JSON <- data.frame(input=input_new$string_input[1:(2*n)])
#to_JSON$input = input_new[,'string_input']
to_JSON$output = output[1:n]
# conversion & save 
nhanes_LLM_JSON <- toJSON(to_JSON)
write(nhanes_LLM_JSON, file = "./with_chol/nhanes_JSON_test.json")

to_JSON <- data.frame(input=input_new$string_input[(n+1):(2*n)])
#to_JSON$input = input_new[,'string_input']
to_JSON$output = output[(n+1):(2*n)]
# conversion & save 
nhanes_LLM_JSON <- toJSON(to_JSON)
write(nhanes_LLM_JSON, file = "./with_chol/nhanes_JSON_validation.json")


to_JSON <- data.frame(input=input_new$string_input[(2*n+1):nrow(input_new)])
#to_JSON$input = input_new[,'string_input']
to_JSON$output = output[(2*n+1):nrow(input_new)]
# conversion & save 
nhanes_LLM_JSON <- toJSON(to_JSON)
write(nhanes_LLM_JSON, file = "./with_chol/nhanes_JSON_train.json")


############ no chol info ############


set.seed(123)

data = nhanes_data %>% filter(htn_jnc7=='Yes')
n = round(0.1*nrow(data))

data = data[sample(1:nrow(data), nrow(data)), ] %>% as.data.frame(check.names=F)

input_cols = colnames(data)[c(2:9,11:13,30:51,102:111)]
keys = data.table(variable = input_cols) %>% left_join(nhanes_key %>% select(variable,label))
input <- data[,keys$variable] 

#fwrite(cbind(data.table(bp_uncontrolled_140_90=data$bp_uncontrolled_140_90),input)[1:n,], file = "./no_chol/nhanes_input_test.csv", row.names = FALSE,col.names=TRUE,sep=',',quote=FALSE)
#fwrite(cbind(data.table(bp_uncontrolled_140_90=data$bp_uncontrolled_140_90),input)[(n+1):(2*n),], file = "./no_chol/nhanes_input_validation.csv", row.names = FALSE,col.names=TRUE,sep=',',quote=FALSE)
#fwrite(cbind(data.table(bp_uncontrolled_140_90=data$bp_uncontrolled_140_90),input)[(2*n+1):nrow(input),], file = "./no_chol/nhanes_input_train.csv", row.names = FALSE,col.names=TRUE,sep=',',quote=FALSE)

colnames(input) <- keys$label

input_new = input
for(i in 1:ncol(input_new)){
    input_new[,i] = paste0(colnames(input_new)[i], ': ', input_new[,i])
}
input_new$string_input = apply(input_new, 1, paste, collapse = ', ')
input_new$string_input_reason = paste0('Suppose I give you the following person information:',input_new$string_input, '. Determin whether this person has hypertension and explain the reason for your answer. Reason:')
input_new$string_input = paste0(input_new$string_input, '. Does this patient have uncontrolled blood pressure (SBP ≥ 140 mm Hg or DBP ≥ 90 mm Hg)?')
output <- paste0(data$bp_uncontrolled_140_90,'.')

to_JSON <- data.frame(input=input_new$string_input[1:(2*n)])
#to_JSON$input = input_new[,'string_input']
to_JSON$output = output[1:n]
# conversion & save 
nhanes_LLM_JSON <- toJSON(to_JSON)
write(nhanes_LLM_JSON, file = "./no_chol/nhanes_JSON_test.json")

#to_JSON <- data.frame(input=input_new$string_input[(n+1):(2*n)])
##to_JSON$input = input_new[,'string_input']
#to_JSON$output = output[(n+1):(2*n)]
# conversion & save 
#nhanes_LLM_JSON <- toJSON(to_JSON)
#write(nhanes_LLM_JSON, file = "./no_chol/nhanes_JSON_validation.json")


to_JSON <- data.frame(input=input_new$string_input[(2*n+1):nrow(input_new)])
#to_JSON$input = input_new[,'string_input']
to_JSON$output = output[(2*n+1):nrow(input_new)]
# conversion & save 
nhanes_LLM_JSON <- toJSON(to_JSON)
write(nhanes_LLM_JSON, file = "./no_chol/nhanes_JSON_train.json")

