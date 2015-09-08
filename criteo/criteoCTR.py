# -*- coding: utf-8 -*-
from pymining import itemmining
from collections import defaultdict
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import linear_model
from sklearn import datasets
from binascii import crc32
from math import exp
from sklearn.externals import joblib
import operator
import json
from sklearn.svm import SVC, LinearSVC

path='/home/alex/data/'

Config = {
          "datas" : path + "day_0_small_data.txt",
          "datab" : path + "day_0_big_data.txt",
          "workdir" : path,
          "freqc" : path + "day_0_small_data/freq30_categorial.txt",
          "features" : path + "features.txt",
          "features_c1" : path + "features_c1.txt",
          "features_cb": path + "features_c_bool.txt",
          "features_cb_c1": path + "features_c_bool_c1.txt",
          "freqCrSelectionDump": path + "fcdump.txt"
          }
NHead = 1000000
def loadFreqCategorial():
    res = defaultdict(dict)
    with open (Config["freqc"]) as inf:
        for line in inf:
            fields = line[:-1].split("\t")
            res[int (fields[0])][fields[2]] = int(fields[1])
    return res

def testDS():
    X, y = datasets.make_hastie_10_2(n_samples=120, random_state=1)
    print X[:4]
    print y[:4]

def categorialFeatures():
    freqCategorial = loadFreqCategorial()
    headN = NHead
    nrows = 0
    with open(Config["datas"]) as inf:
        for line in inf:
            if nrows <= headN:
                rec = line.split("\t")
                click = int(rec[0])
                categorialFields = rec[14:]
                i = 15;
                frqItemsTransaction = [] 
                for category in categorialFields:
                    freqInDataSet = freqCategorial[i].get(category, 0)
                    #print freqInDataSet
                    if freqInDataSet > 0:
                        frqItemsTransaction.append("%i:%s" % (i, category))
                    i += 1
                if len(frqItemsTransaction) > 0 and click > 0:
                    frqItemsTransaction.append("0:%i" % click)
                    #print frqItemsTransaction
                    yield frqItemsTransaction
                nrows += 1
    #        yield categorialFields

class FeaturesBuilder:
    def __init__(self):
        self.nFeatures = 14
        self.categoryMinFreq = 3000
        self.categoryIdex = defaultdict(dict)
        self.loadCategoryIndex()
        
    def loadCategoryIndex(self):
        freqCategorial = loadFreqCategorial()
        self.nFeatures = 14
        for i, categorials in freqCategorial.items():
            for categ, cfreq in categorials.items():
                if cfreq >= self.categoryMinFreq:
                    self.categoryIdex[i - 1][categ] = self.nFeatures
                    self.nFeatures += 1
                    
    def buildCategoryToFlatBoolVal(self, rawFields):
        qfeatures = ['0'] * self.nFeatures
        for i, val in enumerate(rawFields):
            if  i < 14:
                if val != '':
                    qfeatures[i] = val
                else:
                    qfeatures[i] = '-1'
            else:
                cfIndex = self.categoryIdex[i].get(val, -1)
                if cfIndex != -1:
                    qfeatures[cfIndex] = '1'
        return qfeatures

    def buildFFromStr(self, rawRow):
        return self.buildf(rawRow[:-1].split("\t"))

def buildFeaturesTxt():
    fBuilder = FeaturesBuilder()
    with open(Config["datas"]) as fin, open(Config["features"], "w") as fout:
        for row in fin:
            fout.write("\t".join(fBuilder.buildFFromStr(row)) + "\n")

def criteriaSelectionTest():
    cs = criteriaSelection()
    #cs.ststDump()
    cs.statLoad()
    sbc = cs.simpleBayes()
    rait = 0.83
    fscb = filter(lambda item: item[1][0] > rait and item[1][1] > 5000, sbc.items())
    print rait, len(fscb), fscb


class criteriaSelection:
    def __init__(self):
        self.statFName = Config["freqCrSelectionDump"]
        self.stat = {}
        self.CTR = 0.02904
        self.Clicks = self.CTR * 1000000.

    def ststDump(self):
        criterionStat = defaultdict(lambda: {"freq":0., "clicks": 0., })
        with open(Config["featores_cb"]) as fin, open(self.statFName, "w") as fout:
            c = 0.
            s = 0.
            for row in fin:
                fields = row[:-1].split("\t")
                click = float(fields[0])
                for i, val in enumerate(fields):
                    if i >= 14:
                        if float(val) > 0.:
                            criterionStat[i]["freq"] += 1
                            criterionStat[i]["clicks"] += click
                    else:
                        c += click
                    s += 1
            json.dump(dict(criterionStat), fout)
    def statLoad(self):
        with open(self.statFName) as fin:
            self.stat = json.load(fin)
    def simpleBayes(self, rate = 0.30, freqb = 4000):
        self.statLoad()
        res = {}
        for icriteria, stat in self.stat.items():
           ctri =  stat["clicks"] / stat["freq"]
           r = (ctri - self.CTR) / self.CTR
           if abs(r) > rate and stat["freq"] > freqb:
               res[icriteria] = (r, stat["freq"], stat["freq"]/ self.Clicks, ctri)
        return res
    def learnLinearRegressionModels(self):
        informBoolFeatures = self.simpleBayes(0.83, 4000)
        #maxfreq = max(informBoolFeatures, key = lambda x: x[1][2])
        for ift in informBoolFeatures.items():
            itarget = int(ift[0])
            print ift
            size = int(ift[1][1])
            learnY = []
            learnX = []
            size0 = 0
            size1 = 0
            with open(Config["featores_cb"]) as fin:
                for row in fin:
                    fields = row[:-1].split("\t")
                    if int (fields[itarget]) == 1 and size0 <= size:
                        learnY.append(1)
                        size0 += 1
                    elif size1 <= size:
                        learnY.append(0)
                        size1 += 1
                    if len (learnX) < size * 2:
                        learnX.append([ -1 if val == '' else float(val) for val in fields[1:14]])
                    if len (learnY) == size * 2:
                        break
            #regr  = linear_model.LogisticRegression()
            regr = SVC()
            #regr = GradientBoostingClassifier()
            print len (learnX), len (learnY)
            print regr.fit(learnX, learnY)
            print regr.score(learnX, learnY)
            errors = [0, 0]
            for c in  [ (x[0], (x[0] + x[1]) % 2) for x in zip (learnY, regr.predict(learnX))]:
                errors[c[0]] += c[1]
            print errors[0] / float(size0), errors[1] / float(size1)
            model = Config['workdir'] + 'regrSVM_%i.pkl' % itarget
            joblib.dump(regr,  model)
            return

def calcFeatures(finName, foutName):
    cs = criteriaSelection()
    cs.statLoad()
    indexFilter = [int(x) for x in cs.simpleBayes()]
    with open(finName) as fin, open(foutName, 'w') as fout:
        for row in fin:
            fields = row[:-1].split('\t')
            result = fields[:13]
            for i in indexFilter:
                result.append(fields[i])
            fout.write('\t'.join(result) + '\n')


class DataSets:
    def __init__(self):
        self.FeaturesTxt = Config["features"]
        self.FeaturesC1Txt = Config["features_c1"]
        self.xtest = []
        self.ytest = []
        self.xtrain = []
        self.ytrain = []
        self.nC0 = 0
        self.testLearnProportion = 5
        self.c0Proportion = 0.
    def getTrainTest(self):
        y = []
        x = []
        c0 = 0
        with open(self.FeaturesTxt) as fin:
            for row in fin:
                if c0 == self.nC0:
                    break
                rcrc = crc32(row)
                fields = row[:-1].split("\t")
                click = int(fields[0])
                c0 += 1 if click == 0 else 0
                if rcrc % self.testLearnProportion == 0:
                    x, y = self.xtest, self.ytest
                else:
                    x, y = self.xtrain, self.ytrain
                y.append(click)
                x.append( [ -1 if val == '' else float(val) for val in fields[1:]])
        c1 = sum(self.ytrain)
        nc1 = len(self.ytrain) - c1
        self.c0Proportion =  float(len(self.ytrain)) / float(c1)
        self.ytrain += [1] * nc1
        with open(self.FeaturesC1Txt) as fin:
            for row in fin:
                fields = row[:-1].split("\t")
                self.xtrain.append( [ -1 if val == '' else float(val) for val in fields[1:]])
                if len(self.xtrain) == len(self.ytrain):
                    break
        return self.xtrain, self.ytrain, self.xtest, self.ytest

def freqPaatterns(transactions, len):
    print "fp mining"
    relim_input = itemmining.get_relim_input(transactions)
    report = itemmining.relim(relim_input, min_support = len * 0.05)
    with open("d:\data\day_0_small_data\\0f.txt", "w") as fout:
        for fp in report.items():
            #print "%s\t%f\n" % (";".join(fp[0]), fp[1] / float(len))
            fout.write("%s\t%f\n" % (";".join(fp[0]), fp[1] / float(len)))

def testTransactions():
    raw = '25,52,274;71;71,274;52;25,52;274,71'
    transactions = [line.split(',') for line in raw.split(';')]
    return transactions

def featureImportances():
    model = Config['workdir'] + 'gbclogit_cqf.pkl'
    gbclogit = joblib.load(model)
    d = { i:v for i, v in  enumerate( gbclogit.feature_importances_)}
    print sorted(d.items(), key=operator.itemgetter(1))


def classifacation():
    gbclogit = GradientBoostingClassifier(max_features="auto", min_samples_leaf=10, subsample=0.3)
    ds = DataSets()
    ds.nC0 = 10000
    xTrain, yTrain, xTest, yTest = ds.getTrainTest()
    print "size train, test = %i\t%i" % (len(yTrain), len(yTest))
    gbclogit.fit(xTrain, yTrain,  sample_weight=[4 if y == 1 else ds.c0Proportion for y in yTrain])
    print gbclogit.score(xTest, yTest)
    model = Config['workdir'] + 'gbclogit_cqf.pkl'
    joblib.dump(gbclogit,  model)
    gbclogit = joblib.load(model)
    #print gbclogit.feature_importances_
    errors = 0
    tprob = [(exp(lpob[0]), exp(lpob[1])) for lpob in gbclogit.predict_log_proba(xTest)]
    nErrNoclickTruePredict = 0
    for px, y, prob in zip (gbclogit.predict(xTest), yTest, tprob):
        if px - y != 0:
            if y - px < 0:
                print px - y, y, prob
                nErrNoclickTruePredict += 1
            errors += 1
    print "size train, test = %i\t%i" % (len(yTrain), len(yTest))
    print "1test=%i" % sum(yTest)
    print "1train=%i" % sum(yTrain)
    print "nErr1= %f" % ((errors - nErrNoclickTruePredict) / float(sum(yTest)))
    print "nErr0= %f" % ( float(nErrNoclickTruePredict) / float(len(yTest)) )
    print nErrNoclickTruePredict

if __name__ == "__main__":
    #calcFeatures(Config["features_cb"], Config["features"])
    #calcFeatures(Config["features_cb_c1"], Config["features_c1"])
    classifacation()
    #testCategorialToQuantity()
    #featureImportances()
    #classifacation()
    #print "len=%i" % len(transactions)
    #print transactions
    #freqPaatterns(categorialFeatures(), len(transactions))
    