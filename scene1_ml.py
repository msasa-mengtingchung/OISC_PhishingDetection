import pandas as pd
import numpy as np
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve



# nltk.download('stopwords')
# nltk.download('punkt_tab')

stop_words = nltk.corpus.stopwords.words('english')


stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop_words:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)

def ConfMatrixDisp(y_test, predictions, label, name, title):
    cm = confusion_matrix(y_test, predictions, labels=label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot()
    plt.title(title + " Confusion Matrix")
    #plt.show()
    plt.savefig('images/'+name + '.png', dpi=100)
    plt.clf()

def Tokenizer(df):
    for index, row in df.iterrows():
        token_word = nltk.tokenize.word_tokenize(row.iloc[0])
        #print(token_word)
        filtered_token = []
        for word in token_word:
            if word not in stop_words:
                filtered_token.append(word)


        filtered_sentence = " ".join(filtered_token)

        df.loc[index, 'body'] = filtered_sentence
    return df

def TestModel(model_name, model, X_test, y_test, proc):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn+fp)
    if proc == 0:
        print(model_name)
        ConfMatrixDisp(y_test,y_pred, model.classes_, model_name, model_name)
    elif proc == 1:
        print(model_name+" AI ")
        ConfMatrixDisp(y_test,y_pred, model.classes_, model_name+" AI", model_name+" AI")
    elif proc == 2:
        print(model_name+" Overall ")
        ConfMatrixDisp(y_test,y_pred, model.classes_, model_name+" AI", model_name+" AI")

    if proc == 0:
        fpr, tpr, _ = roc_curve(y_test,  y_pred)
        auc_lr = roc_auc_score(y_test, y_pred)
        plt.plot(fpr,tpr,label="AUC="+str(auc_lr))
        plt.title('ROC Curve for '+model_name)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.savefig('images/'+model_name+'.png')
        plt.clf()
    elif proc == 1:
        fpr, tpr, _ = roc_curve(y_test,  y_pred)
        auc_lr = roc_auc_score(y_test, y_pred)
        plt.plot(fpr,tpr,label="AUC="+str(auc_lr))
        plt.title('ROC Curve for '+model_name+" AI ")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.savefig('images/'+model_name+' AI.png')
        plt.clf()
    elif proc==2:
        fpr, tpr, _ = roc_curve(y_test,  y_pred)
        auc_lr = roc_auc_score(y_test, y_pred)
        plt.plot(fpr,tpr,label="AUC="+str(auc_lr))
        plt.title('ROC Curve for '+model_name+" Overall ")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.savefig('images/'+model_name+' Overall.png')
        plt.clf()

    return accuracy_score(y_test,y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), specificity

def PopulateDF(df, model_name, accuracy, precision, recall, specificity):
    print("Populating ", model_name)
    df.loc[model_name, 'Accuracy'] = accuracy
    df.loc[model_name, 'Precision'] = precision
    df.loc[model_name, 'Sensitivity'] = recall
    df.loc[model_name, 'Specificity'] = specificity
    print("Done Populating: "+model_name)
    return df
    


df_human = pd.read_csv('clean_dataset.csv')
df_ai = pd.read_csv('generated_email.csv')

#make sure that they are in order
df_human = df_human[['body', 'label']]
df_ai = df_ai[['body', 'label']]
# print(len(df_human[df_human['label'] == 0].index))


#AI 9100 generated, Human = 15686
#shuffle gen AI
df_human = df_human.sample(frac=1, random_state=1).reset_index(drop=True)
df_ai = df_ai.sample(frac=1, random_state=1).reset_index(drop=True)
df_human = df_human.sample(frac=1, random_state=1).reset_index(drop=True)
df_ai = df_ai.sample(frac=1, random_state=1).reset_index(drop=True)

df_human = Tokenizer(df_human)
df_ai = Tokenizer(df_ai)

df_human['body'] = df_human['body'].apply(stem_text)
df_ai['body'] = df_ai['body'].apply(stem_text)

df_combined_with_ai = pd.concat([df_human, df_ai], ignore_index=True)

texts = []

for index, row in df_combined_with_ai.iterrows():
    texts.append(row.iloc[0])

cv = TfidfVectorizer(max_features=10000)

X_combined = cv.fit_transform(texts).toarray()
y_combined = df_combined_with_ai.iloc[:,1].values

human_data_portion = len(df_human.index)

X_human = X_combined[0:human_data_portion]
y_human = y_combined[0:human_data_portion]

X_ai = X_combined[human_data_portion:]
y_ai = y_combined[human_data_portion:]

# print(type(X_human))
# print(type(X_ai))
# print(len(X_human))
# print(len(X_ai))
# print(len(X_combined))

#HUMAN DATA =  15686    TRAIN = 12500 (80%)      TEST = 3137  (20%)
#AI DATA    =  8209     TRAIN = 6567             TEST = 1642
X_human_train = X_human[0:12500]
X_human_test = X_human[12500:]
y_human_train = y_human[0:12500]
y_human_test = y_human[12500:]

X_ai_train = X_ai[0:6567]
X_ai_test = X_ai[6567:]
y_ai_train = y_ai[0:6567]
y_ai_test = y_ai[6567:]

#########CREATE DATAFRAME################################
models = ['Logistic Regression', 'Naive Bayes', 'SVM', 'Decision Trees', 'Ensemble', 'Neural Network']
df_results = pd.DataFrame({'Model': models, 'Accuracy': models, 'Precision': models, 
                            'Sensitivity': models, 'Specificity':models})

df_results_ai = pd.DataFrame({'Model': models, 'Accuracy': models, 'Precision': models, 
                            'Sensitivity': models, 'Specificity':models})
df_results_overall = pd.DataFrame({'Model': models, 'Accuracy': models, 'Precision': models, 
                            'Sensitivity': models, 'Specificity':models})
df_results_ai = df_results_ai.set_index(['Model'])
df_results = df_results.set_index(['Model'])
df_results_overall = df_results_overall.set_index(['Model'])
############################################################
# print("Creating Models")
# print("Creating Log Reg")
# LR = LogisticRegression()
# model_lr = LR.fit(X_human_train, y_human_train)
# print("Done Creating Log Reg")
# print("Creating Naive Bayes")
# NB = GaussianNB()
# model_nb = NB.fit(X_human_train, y_human_train)
# print("Done Creating Naive Bayes")
# print("Creating Decision Trees")
# DT = DecisionTreeClassifier(random_state=0, ccp_alpha=0.0002)

# model_dt = DT.fit(X_human_train, y_human_train)
# path = DT.cost_complexity_pruning_path(X_human_train, y_human_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities

# clfs = []
# for ccp_alpha in ccp_alphas:
#     clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
#     clf.fit(X_human_train, y_human_train)
#     clfs.append(clf)
# print(
#     "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
#         clfs[-1].tree_.node_count, ccp_alphas[-1]
#     )
# )

# clfs = clfs[:-1]
# ccp_alphas = ccp_alphas[:-1]

# node_counts = [clf.tree_.node_count for clf in clfs]
# depth = [clf.tree_.max_depth for clf in clfs]

# train_scores = [clf.score(X_human_train, y_human_train) for clf in clfs]
# test_scores = [clf.score(X_human_test, y_human_test) for clf in clfs]

# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
# ax.legend()
# plt.show()

# print("Done Creating Decision Trees")

# x='x'+1

print("Creating SVM")
SVM = SVC(kernel = 'rbf', random_state = 0, C=0.001)
model_svm = SVM.fit(X_human_train, y_human_train)
print("Done SVM")
# print("Creating Ensemble")
# Ensamble = VotingClassifier([('LR', LR), ('NB', NB), ('DT', DT)], voting = 'hard')
# model_ens = Ensamble.fit(X_human_train, y_human_train)
# print("Done Ensemble")
# print("Creating Neural Network")
# NN = MLPClassifier()
# model_nn = NN.fit(X_human_train, y_human_train)
# print("Done Creating Neural Network")

# model_name = ["Logistic Regression", 'Naive Bayes', "SVM", "Decision Trees", "Ensemble", "Neural Network"]
model_name=["SVM"]
# models = [model_lr, model_nb, model_dt, model_svm, model_ens, model_nn]
models = [model_svm]

for idx, model in enumerate(models):
    print("Starting test for human dataset: ", model_name[idx])
    accuracy, precision, recall, specificity = TestModel(model_name[idx],
                                                        model,
                                                        X_human_test,
                                                        y_human_test,
                                                        0)
    df_results = PopulateDF(df_results, model_name[idx], accuracy, precision, recall, specificity)
    print(accuracy, precision, recall, specificity) 
    print("Finished Testing for", model_name[idx])
   
for idx, model in enumerate(models):
    print("Starting test for AI dataset: ", model_name[idx])
    accuracy, precision, recall, specificity = TestModel(model_name[idx],
                                                        model,
                                                        X_ai_test,
                                                        y_ai_test,
                                                        1)
    df_results_ai = PopulateDF(df_results_ai, model_name[idx], accuracy, precision, recall, specificity)
    print(accuracy, precision, recall, specificity) 
    print("Finished Testing for AI:", model_name[idx])



for idx, model in enumerate(models):
    print("Starting test for overall dataset: ", model_name[idx])
    X_overall_test = np.concatenate((X_human_test, X_ai_test))
    y_overall_test = np.concatenate((y_human_test, y_ai_test))
    accuracy, precision, recall, specificity = TestModel(model_name[idx],
                                                        model,
                                                        X_overall_test,
                                                        y_overall_test,
                                                        2)
    df_results_overall = PopulateDF(df_results_overall, model_name[idx], accuracy, precision, recall, specificity)
    print(accuracy, precision, recall, specificity) 
    print("Finished Testing for Overall", model_name[idx])


df_results.to_excel("Scene1.xlsx")
df_results_ai.to_excel("Scene1AI.xlsx")
df_results_overall.to_excel("Scene1Overall.xlsx")