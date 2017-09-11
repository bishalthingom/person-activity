from __future__ import division
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

def ml_models(train, test, lab, labt):
    #Random Forest
    forest = RandomForestClassifier(n_estimators=200, max_leaf_nodes=50, criterion="entropy")
    forest = forest.fit(train, lab)
    output_rf = forest.predict(test).astype(int)
    suc_rf = 0
    totals_rf = [0 for m in range(num)]
    preds_rf = [0 for m in range(num)]
    for i in range(0, len(labt)):
        totals_rf[labt[i]] += 1
        if output_rf[i] == labt[i]:
            suc_rf = suc_rf + 1
            preds_rf[labt[i]] += 1


    accuracy_rf = suc_rf / len(labt)

    #KNearest Neighbour

    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(train, lab)
    output_kn = neigh.predict(test)
    suc_kn = 0
    totals_kn = [0 for m in range(num)]
    preds_kn = [0 for m in range(num)]
    for i in range(0, len(labt)):
        totals_kn[labt[i]] += 1
        if output_kn[i] == labt[i]:
            suc_kn = suc_kn + 1
            preds_kn[labt[i]] += 1

    accuracy_kn = suc_kn / len(labt)

    # Logistic Regression

    model = LogisticRegression()
    model.fit(train, lab)
    output_lr = model.predict(test)
    suc_lr = 0
    totals_lr = [0 for m in range(num)]
    preds_lr = [0 for m in range(num)]
    for i in range(0, len(labt)):
        totals_lr[labt[i]] += 1
        if output_lr[i] == labt[i]:
            suc_lr = suc_lr + 1
            preds_lr[labt[i]] += 1

    accuracy_lr = suc_lr / len(labt)

    # Naive Bayes

    model = GaussianNB()
    model.fit(train, lab)
    # print(model)
    # make predictions
    # expected = y
    output_nb = model.predict(test)

    suc_nb = 0
    totals_nb = [0 for m in range(num)]
    preds_nb = [0 for m in range(num)]
    for i in range(0, len(labt)):
        totals_nb[labt[i]] += 1
        if output_nb[i] == labt[i]:
            suc_nb = suc_nb + 1
            preds_nb[labt[i]] += 1

    accuracy_nb = suc_nb / len(labt)

    # Decision Tree Classifier

    model = DecisionTreeClassifier()
    model.fit(train, lab)
    output_dt = model.predict(test)

    suc_dt = 0
    totals_dt = [0 for m in range(num)]
    preds_dt = [0 for m in range(num)]
    for i in range(0, len(labt)):
        totals_dt[labt[i]] += 1
        if output_dt[i] == labt[i]:
            suc_dt = suc_dt + 1
            preds_dt[labt[i]] += 1

    accuracy_dt = suc_dt / len(labt)

    # Support Vector Machine

    clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
    clf.fit(train, lab)
    output_sv = clf.predict(test)

    suc_sv = 0
    totals_sv = [0 for m in range(num)]
    preds_sv = [0 for m in range(num)]
    for i in range(0, len(labt)):
        totals_sv[labt[i]] += 1
        if output_sv[i] == labt[i]:
            suc_sv = suc_sv + 1
            preds_sv[labt[i]] += 1

    accuracy_sv = suc_sv / len(labt)

   # Majority voting

    def Most_Common(lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

    output_mv = []
    for i in range(0, len(labt)):
        c = [output_dt[i], output_rf[i], output_lr[i]]
        output_mv.append(Most_Common(c))

    suc_mv = 0
    totals_mv = [0 for m in range(num)]
    preds_mv = [0 for m in range(num)]
    for i in range(0, len(labt)):
        totals_mv[labt[i]] += 1
        if output_mv[i] == labt[i]:
            suc_mv = suc_mv + 1
            preds_mv[labt[i]] += 1


    accuracy_mv = suc_mv / len(labt)

    return accuracy_rf, accuracy_kn, accuracy_lr, accuracy_nb, accuracy_dt, accuracy_sv, accuracy_mv, \
           preds_rf, preds_kn, preds_lr, preds_nb, preds_dt, preds_sv, preds_mv, \
           totals_rf, totals_kn, totals_lr, totals_nb, totals_dt, totals_sv, totals_mv

acts = ['walk', 'run', 'cycl']
pers = ['adi', 'asi', 'bis', 'dinesh', 'rj', 'vin', 'yog', 'ayush']
labi = []
lab = []
temp = []
results = []
x = [1]
df = pd.DataFrame(
    columns=['macx', 'macy', 'macz', 'mfacx', 'mfacy', 'mfacz', 'mfaxz', 'mfayz', 'cou', 'fcou', 'peakx', 'peaky',
             'peakz', 'stdx', 'stdy', 'stdz', 'stdav', 'dix', 'diy', 'diz', 'maxi', 'may', 'maz', 'mix', 'miy', 'miz',
             'specx', 'specy', 'specz', 'spec'])

num = len(acts)

for i in range(0, len(acts)):
    for j in range(0, len(pers)):
        try:
            a = pd.read_csv('./Feat/' + acts[i]  + '_' + pers[j] + '_ft_01.csv')
            a1 = pd.read_csv('./Feat/' + acts[i]  + '_' + pers[j] + '_ft_02.csv')
            a2 = pd.read_csv('./Feat/' + acts[i] + '_' + pers[j] + '_ft_03.csv')
            df = pd.concat([df, a, a1, a2], ignore_index=True)
            labi = np.repeat(x, [len(a) + len(a1) + len(a2)]).tolist()
            for j in range(0, len(a) + len(a1) + len(a2)):
                results.append(i)
        except:
            pass
    x[0] = x[0] + 1

df = df.assign(act = results)

df = shuffle(df)

train, test = train_test_split(df, test_size=0.1)

train_output = train[['act']].values
train = train.drop(train.columns[[0, 31]], axis=1)
test_output = test[['act']].values
test = test.drop(test.columns[[0, 31]], axis=1)
train = train.values
test = test.values
lab = train_output.flatten()
labt = test_output.flatten()


accuracies_rf = []
accuracies_kn = []
accuracies_lr = []
accuracies_nb = []
accuracies_dt = []
accuracies_sv = []
accuracies_mv = []
total_preds_rf = [0 for i in range(num)]
total_preds_kn = [0 for i in range(num)]
total_preds_lr = [0 for i in range(num)]
total_preds_nb = [0 for i in range(num)]
total_preds_dt = [0 for i in range(num)]
total_preds_sv = [0 for i in range(num)]
total_preds_mv = [0 for i in range(num)]
total_totals_rf = [0 for i in range(num)]
total_totals_kn = [0 for i in range(num)]
total_totals_lr = [0 for i in range(num)]
total_totals_nb = [0 for i in range(num)]
total_totals_dt = [0 for i in range(num)]
total_totals_sv = [0 for i in range(num)]
total_totals_mv = [0 for i in range(num)]

for i in range(0,10,1):

    df = shuffle(df)

    train, test = train_test_split(df, test_size=0.1)

    train_output = train[['act']].values
    train = train.drop(train.columns[[0, 31]], axis=1)
    test_output = test[['act']].values
    test = test.drop(test.columns[[0, 31]], axis=1)
    train = train.values
    test = test.values
    lab = train_output.flatten()
    labt = test_output.flatten()

    accuracy_rf, accuracy_kn, accuracy_lr, accuracy_nb, accuracy_dt,accuracy_sv, accuracy_mv, \
    preds_rf, preds_kn, preds_lr, preds_nb, preds_dt, preds_sv, preds_mv, \
    totals_rf, totals_kn, totals_lr, totals_nb, totals_dt, totals_sv, totals_mv\
        = ml_models(train, test, lab, labt)

    for j in range(num):
        total_preds_rf[j] += preds_rf[j]
        total_preds_kn[j] += preds_kn[j]
        total_preds_lr[j] += preds_lr[j]
        total_preds_nb[j] += preds_nb[j]
        total_preds_dt[j] += preds_dt[j]
        total_preds_sv[j] += preds_sv[j]
        total_preds_mv[j] += preds_mv[j]
        total_totals_rf[j] += totals_rf[j]
        total_totals_kn[j] += totals_kn[j]
        total_totals_lr[j] += totals_lr[j]
        total_totals_nb[j] += totals_nb[j]
        total_totals_dt[j] += totals_dt[j]
        total_totals_sv[j] += totals_sv[j]
        total_totals_mv[j] += totals_mv[j]

    accuracies_rf.append(accuracy_rf)
    accuracies_kn.append(accuracy_kn)
    accuracies_lr.append(accuracy_lr)
    accuracies_nb.append(accuracy_nb)
    accuracies_dt.append(accuracy_dt)
    accuracies_sv.append(accuracy_sv)
    accuracies_mv.append(accuracy_mv)

print "Random Forest: " + str(np.average(accuracies_rf))
print "K Neighbours: " + str(np.average(accuracies_kn))
print "Logistic Regression: " + str(np.average(accuracies_lr))
print "Naive Bayes: " + str(np.average(accuracies_nb))
print "Decision Tree: " + str(np.average(accuracies_dt))
print "Support Vector Machine: " + str(np.average(accuracies_sv))
print "Majority Voting" + str(np.average(accuracies_mv))

for i in range(num):
    print str(i) + " RF: " + str((total_preds_rf[i]*100.0/num)/(total_totals_rf[i]*1.0/num)) + " KN: " + str((total_preds_kn[i]*100.0/num)/(total_totals_kn[i]*1.0/num)) \
    + " LR: " + str((total_preds_lr[i]*100.0/num)/(total_totals_lr[i]*1.0/num)) + " NB: " + str((total_preds_nb[i]*100.0/num)/(total_totals_nb[i]*1.0/num)) \
    + " DT: " + str((total_preds_dt[i]*100.0/num)/(total_totals_dt[i]*1.0/num)) + " SV: " + str((total_preds_sv[i]*100.0/num)/(total_totals_sv[i]*1.0/num)) \
    + " MV: " + str((total_preds_mv[i]*100.0/num)/(total_totals_mv[i]*1.0/num))