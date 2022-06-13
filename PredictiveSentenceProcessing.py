import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import torch


#Bert functions
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
bert_mlm = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
ids2token = tokenizer.convert_ids_to_tokens
token2ids = tokenizer.convert_tokens_to_ids
token2string = tokenizer.convert_tokens_to_string


#open the files
def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

data1 = read_data('ex1_data.txt')
data2 = read_data('ex2_data.txt')


#probability & surprisal
def calc_surp(seq,val):
  seq = seq.detach().numpy()
  val = val.detach().numpy()
  exp_sum = sum(np.exp(seq))
  portion = np.exp(val) / exp_sum
  surp = - np.log(portion)
  return portion, surp


def pair_prob_all(s1,arr,s2,mask,model):
  inputs  = tokenizer(s1, return_tensors="pt")
  indice  = token2ids(arr)
  inputs["input_ids"] = torch.tensor([indice])
  labels  = tokenizer(s2, return_tensors="pt")["input_ids"]
  label = labels[0][int(mask)]
  if len(inputs["input_ids"][0]) == len(labels[0]):
    outputs = model(**inputs, labels=labels)
    loss    = outputs[0]
    logits  = outputs[1]
    probs   = logits[0][int(mask)]
    prob    = logits[0][int(mask)][int(label)]
    portion = calc_surp(probs, prob)[0]
    surp    = calc_surp(probs,prob)[1]
    return loss, logits, portion, surp
  else:
    return 0, 0, 0, 0


#experiment settings
def return_probability_all(corpus, model):
    corpus_all = []
    for i in range(len(corpus)):
        indexed = tokenizer(corpus[i][1], return_tensors="pt")["input_ids"][0]
        arr1_token = ids2token(indexed)
        arr = ids2token(indexed)
        if "whom" in arr1_token:
            index_c = int(arr1_token.index("whom") + 2)
            index_b = int(arr1_token.index("whom") + 4)
        elif "who" in arr1_token:
            index_c = int(arr1_token.index("who") + 2)
            index_b = int(arr1_token.index("who") + 4)

        fillers_b = corpus[i][3].split(' ')
        fillers_c = corpus[i][4].split(' ')

        temp = []
        temp.append(corpus[i][0])
        temp.append(corpus[i][1])
        temp.append(corpus[i][2])

        ###Predictive Sentence Processing
        for j in range(0, 2):
            for k in range(0, 2):
                arr[index_c] = fillers_c[j]
                arr1_token[index_c] = fillers_c[j]
                arr1_token[index_b] = fillers_b[k]
                text_b = token2string(arr1_token[1:-2]) + "."
                loss, logits, portion, surp = pair_prob_all(corpus[i][1], arr, text_b, index_b, model)
                if loss != 0:
                    scores = '(' + fillers_c[j] + fillers_b[k] + ',' + str(portion) + ')'
                    temp.append(scores)
                else:
                    scores = '(' + fillers_b[j] + fillers_b[k] + ',' + "0" + ')'
                    temp.append(scores)

        ### Integration
        ## B_active & C
        for j in range(0, 2):
            arr[index_b] = fillers_b[0]
            arr1_token[index_b] = fillers_b[0]
            arr1_token[index_c] = fillers_c[j]
            text_c = token2string(arr1_token[1:-2]) + "."
            loss, logits, portion, surp = pair_prob_all(corpus[i][1], arr, text_c, index_c, model)
            if loss != 0:
                scores = '(' + fillers_c[j] + fillers_b[0] + ',' + str(portion) + ')'
                temp.append(scores)
            else:
                scores_c_b = '(' + fillers_c[j] + fillers_b[0] + ',' + "0" + ')'
                temp.append(scores)

        ## B_passive & C
        for j in range(0, 2):
            arr[index_b] = fillers_b[1]
            arr1_token[index_b] = fillers_b[1]
            arr1_token[index_c] = fillers_c[j]
            text_c = token2string(arr1_token[1:-2]) + "."
            loss, logits, portion, surp = pair_prob_all(corpus[i][1], arr, text_c, index_c, model)
            if loss != 0:
                scores = '(' + fillers_c[j] + fillers_b[1] + ',' + str(portion) + ')'
                temp.append(scores)
            else:
                scores = '(' + fillers_c[j] + fillers_b[1] + ',' + "0" + ')'
                temp.append(scores)

        corpus_all.append(temp)

    for a in range(len(corpus_all)):
        p_pre_00 = float(corpus_all[a][3].replace("(", "").replace(")", "").split(",")[1])
        p_pre_01 = float(corpus_all[a][4].replace("(", "").replace(")", "").split(",")[1])
        p_pre_10 = float(corpus_all[a][5].replace("(", "").replace(")", "").split(",")[1])
        p_pre_11 = float(corpus_all[a][6].replace("(", "").replace(")", "").split(",")[1])
        p_itg_00 = float(corpus_all[a][7].replace("(", "").replace(")", "").split(",")[1])
        p_itg_01 = float(corpus_all[a][8].replace("(", "").replace(")", "").split(",")[1])
        p_itg_10 = float(corpus_all[a][9].replace("(", "").replace(")", "").split(",")[1])
        p_itg_11 = float(corpus_all[a][10].replace("(", "").replace(")", "").split(",")[1])

        sur_pre_00 = - np.log(p_pre_00)
        sur_pre_01 = - np.log(p_pre_01)
        sur_pre_10 = - np.log(p_pre_10)
        sur_pre_11 = - np.log(p_pre_11)
        sur_itg_00 = - np.log(p_itg_00)
        sur_itg_01 = - np.log(p_itg_01)
        sur_itg_10 = - np.log(p_itg_10)
        sur_itg_11 = - np.log(p_itg_11)

        corpus_all[a].append(sur_pre_00)
        corpus_all[a].append(sur_pre_01)
        corpus_all[a].append(sur_pre_10)
        corpus_all[a].append(sur_pre_11)
        corpus_all[a].append(sur_itg_00)
        corpus_all[a].append(sur_itg_01)
        corpus_all[a].append(sur_itg_10)
        corpus_all[a].append(sur_itg_11)

    return corpus_all


#practice
p_all_1 = return_probability_all(data1, bert_mlm)
p_all_2 = return_probability_all(data2, bert_mlm)


#save files
file_ex1 = open('ex1_result.txt','w')
file_ex2 = open('ex2_result.txt','w')

def write_cases(corpus,wfiles):
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            wfiles.write(str(corpus[i][j])+'\t')
            if j == len(corpus[i])-1:
                wfiles.write('\n')

write_cases(p_all_1,file_ex1)
write_cases(p_all_2,file_ex2)