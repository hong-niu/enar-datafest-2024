from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd  
import sklearn


tokenizer = AutoTokenizer.from_pretrained("./1.8B/lora/with_chol", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    './1.8B/lora/with_chol',
    device_map="auto",
    trust_remote_code=True,cache_dir="./cache/"
).eval()


inputs = tokenizer('Mobile examination center weights: 7991.127474, Primary sampling unit: 1, Strata: 157, NHANES cycle: 2017-2020, Subpopulation for hypertension: 1, Subpopulation for cholesterol: 0, Age category, years: 75+, Race/ethnicity: Non-Hispanic White, Age, years: 80, Pregnant: No, Gender: Men, Self-reported antihypertensive medication use: No, Antihypertensive medication use recommended by the JNC7 guideline: Yes, Antihypertensive medication use recommended by the 2017 ACC/AHA BP guideline: Yes, Antihypertensive medication use recommended by the 2018 ESC/ESH arterial hypertension guideline: Yes, Number of antihypertensive medication classes: Three, Number of antihypertensive medication pills: Two, Combination therapy: Yes, Taking two or more antihypertensive medication pills: Yes, ACE inhibitors: No, Aldosterone antagonists: No, Alpha-1 blockers: No, Angiotensin receptor blockers: Yes, Beta blockers: No, Central alpha1 agonist and other centrally acting agents: No, Calcium channel blockers: Yes, Dihydropyridine calcium channel blockers: No, Non-dihydropyridine calcium channel blockers: Yes, Potassium sparing diuretics: No, Loop diuretics: No, Thiazide or thiazide-type diuretics: Yes, Direct renin inhibitors: No, Direct vasodilators: No, Smoking status: Never, Body mass index, kg/m2: NA, Prevalent diabetes: Yes, Prevalent chronic kidney disease: Yes, History of myocardial infarction: No, History of coronary heart disease: No, History of stroke: No, History of ASCVD: No, History of heart failure: No, History of CVD: No. Does this patient have uncontrolled blood pressure (SBP ≥ 140 mm Hg or DBP ≥ 90 mm Hg)?', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

inputs = tokenizer('suppose you are a data analysis expert. what are the causes of hypertension? choose from the following variables: Participant identifier, Primary sampling unit, Strata, Mobile examination center weights, Subpopulation for hypertension, NHANES cycle, Smoking status, Body mass index, Prevalent diabetes, Prevalent chronic kidney disease,  History of myocardial infarction, History of coronary heart disease, History of stroke, History of ASCVD, History of heart failure, History of CVD, Age category, Race/ethnicity, Age, Pregnant, Gender. Your answer: ', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))


inputs = tokenizer('suppose I give you the information including '+
'Mobile examination center weights, Primary sampling unit, Strata, NHANES cycle, Subpopulation for hypertension, Subpopulation for cholesterol,'+
'Age category, years, Race/ethnicity, Age, years, Pregnant, Gender, Self-reported antihypertensive medication use,'+
'Antihypertensive medication use recommended by the JNC7 guideline, Antihypertensive medication use recommended by the 2017 ACC/AHA BP'+
'guideline, Antihypertensive medication use recommended by the 2018 ESC/ESH arterial hypertension guideline, '+
'Number of antihypertensive medication classes, Number of antihypertensive medication pills, '+
'Combination therapy, Taking two or more antihypertensive medication pills, ACE inhibitors, Aldosterone antagonists,'+
'Alpha-1 blockers, Angiotensin receptor blockers, Beta blockers, Central alpha1 agonist and other centrally acting agents,'+
'Calcium channel blockers, Dihydropyridine calcium channel blockers, Non-dihydropyridine calcium channel blockers,'+
'Potassium sparing diuretics, Loop diuretics, Thiazide or thiazide-type diuretics,'+
'Direct renin inhibitors, Direct vasodilators, Smoking status, Body mass index, kg/m2, Prevalent diabetes,'+
'Prevalent chronic kidney disease, History of myocardial infarction, '+
'History of coronary heart disease, History of stroke, History of ASCVD, History of heart failure, History of CVD. '+
'I want you to tell me whether this person has uncontroled blood pressure. What are the variables affect your decision?\n', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

inputs = tokenizer('suppose I give you the information including '+
'Mobile examination center weights, Primary sampling unit, Strata, NHANES cycle, Subpopulation for hypertension, Subpopulation for cholesterol,'+
'Age category, years, Race/ethnicity, Age, years, Pregnant, Gender, Self-reported antihypertensive medication use,'+
'Antihypertensive medication use recommended by the JNC7 guideline, Antihypertensive medication use recommended by the 2017 ACC/AHA BP'+
'guideline, Antihypertensive medication use recommended by the 2018 ESC/ESH arterial hypertension guideline, '+
'Number of antihypertensive medication classes, Number of antihypertensive medication pills, '+
'Combination therapy, Taking two or more antihypertensive medication pills, ACE inhibitors, Aldosterone antagonists,'+
'Alpha-1 blockers, Angiotensin receptor blockers, Beta blockers, Central alpha1 agonist and other centrally acting agents,'+
'Calcium channel blockers, Dihydropyridine calcium channel blockers, Non-dihydropyridine calcium channel blockers,'+
'Potassium sparing diuretics, Loop diuretics, Thiazide or thiazide-type diuretics,'+
'Time since having their cholesterol measured, Total cholesterol, mg/dL, HDL cholesterol, mg/dL, Triglycerides, mg/dL,'+
'LDL cholesterol, mg/dL, Taking a cholesterol-lowering medication, Self-reported cholesterol-lowering medication use,'+
'Taking a statin, Taking ezetimibe, Taking a PCSK9 inhibitor, Taking a bile acid sequestrant, Taking a fibrate, Taking atorvastatin,'+
'Taking simvastatin, Taking rosuvastatin, Taking pravastatin, Taking pitavastatin, Taking fluvastatin, Taking lovastatin, '+
'Taking other cholesterol-lowering medication, Taking add-on lipid-lowering therapy (ezetimibe or PCSK9 inhibitor),'+
'Recommended add-on lipid-lowering therapy by the 2018 AHA/ACC guideline, Recommended a statin by the 2018 AHA/ACC guideline,'+
'Ever been told to take cholesterol-lowering medication, '+
'Direct renin inhibitors, Direct vasodilators, Smoking status, Body mass index, kg/m2, Prevalent diabetes,'+
'Prevalent chronic kidney disease, History of myocardial infarction, '+
'History of coronary heart disease, History of stroke, History of ASCVD, History of heart failure, History of CVD. '+
'What are the causal variables for worsen BP control??\n', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

inputs = tokenizer(
"inputs: A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."+
'Human: What variables cause the prevalence of BP control among US adults with hypertension has decreased since 2013? \n Answer and Explanation:', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

inputs = tokenizer(
"inputs: A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."+
'Human: Which medicine is most important for blood pressure control in hypertension patient? \n Answer and Explanation:', return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

# classification accuracy

prediction_1_8B = pd.read_json(path_or_buf='./Qwen/prediction/with_chol/Qwen1.8B_lora/generated_predictions.jsonl', lines=True)

y_true = prediction_1_8B['label'].values
y_pred = prediction_1_8B['predict'].values
confusion_matrix(y_true, y_pred)
y_pred[y_pred == 'Yes.'] = 1
y_pred[y_pred == 'No.'] = 0
y_pred = y_pred.astype(int)
y_true[y_true == 'Yes.'] = 1
y_true[y_true == 'No.'] = 0
y_true = y_true.astype(int)
sklearn.metrics.f1_score(y_true, y_pred) 
sklearn.metrics.accuracy_score(y_true, y_pred)

prediction_1_8B = pd.read_json(path_or_buf='./Qwen/prediction/no_chol/Qwen1.8B_lora/generated_predictions.jsonl', lines=True)
y_true = prediction_1_8B['label'].values
y_pred = prediction_1_8B['predict'].values
confusion_matrix(y_true, y_pred)
y_pred[y_pred == 'Yes.'] = 1
y_pred[y_pred == 'No.'] = 0
y_pred = y_pred.astype(int)
y_true[y_true == 'Yes.'] = 1
y_true[y_true == 'No.'] = 0
y_true = y_true.astype(int)
sklearn.metrics.f1_score(y_true, y_pred) 
sklearn.metrics.accuracy_score(y_true, y_pred)

prediction_7B = pd.read_json(path_or_buf='./Qwen/prediction/with_chol/Qwen7B_qlora/generated_predictions.jsonl', lines=True)

y_true = prediction_7B['label'].values
y_pred = prediction_7B['predict'].values
confusion_matrix(y_true, y_pred)
y_pred[y_pred == 'Yes.'] = 1
y_pred[y_pred == 'No.'] = 0
y_pred = y_pred.astype(int)
y_true[y_true == 'Yes.'] = 1
y_true[y_true == 'No.'] = 0
y_true = y_true.astype(int)
sklearn.metrics.f1_score(y_true, y_pred) 
sklearn.metrics.accuracy_score(y_true, y_pred)

prediction_7B = pd.read_json(path_or_buf='./Qwen/prediction_old/no_chol/Qwen7B_qlora/generated_predictions.jsonl', lines=True)
y_true = prediction_7B['label'].values
y_pred = prediction_7B['predict'].values
confusion_matrix(y_true, y_pred)
y_pred[y_pred == 'Yes.'] = 1
y_pred[y_pred == 'No.'] = 0
y_pred = y_pred.astype(int)
y_true[y_true == 'Yes.'] = 1
y_true[y_true == 'No.'] = 0
y_true = y_true.astype(int)
sklearn.metrics.f1_score(y_true, y_pred) 
sklearn.metrics.accuracy_score(y_true, y_pred)
