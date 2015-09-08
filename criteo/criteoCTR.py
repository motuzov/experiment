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
from optparse import OptionParser

#path='/home/alex/data/'
path = "d:\\data\\criteo\\"
Config = {
          "datas" : path + "day_0_small_data.txt",
          "datab" : path + "day_0_big_data.txt",
          "workdir" : path,
          "freqc" : path + "freq30_categorial.txt",
          "features" : path + "features.txt",
          "features_c1" : path + "features_c1.txt",
          "f_flat_category": path + "features_flat_category.txt",
          "f_flat_category_c1": path + "features_flat_category_c1.txt",
          "freqCrSelectionDump": path + "fcdump.txt",
          "freqPatterns0": path + "0fp.txt",
          "freqPatterns1": path + "1fp.txt" 
          }
NHead = 1000000
def loadFreqCategorial():
    res = defaultdict(dict)
    with open (Config["freqc"]) as inf:
        for line in inf:
            fields = line[:-1].split("\t")
            res[int (fields[0])][fields[2]] = int(fields[1])
    return res

def testTransactions():
    raw = '25,52,274;71;71,274;52;25,52;274,71'
    transactions = [line.split(',') for line in raw.split(';')]
    return transactions

def getTransaction(classCond = 1, headN = 10000):
    freqCategorial = loadFreqCategorial()
    nrows = 0
    with open(Config["datas"]) as inf:
        for line in inf:
            if nrows > headN:
                break
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
            if len(frqItemsTransaction) > 0 and click == classCond:
                frqItemsTransaction.append("0:%i" % click)
                yield frqItemsTransaction
            nrows += 1
            

def freqPaatterns(transactions, len, outfName):
    print "fp mining"
    relim_input = itemmining.get_relim_input(transactions)
    report = itemmining.relim(relim_input, min_support = len * 0.01)
    with open(outfName, "w") as fout:
        for fp in report.items():
            #print "%s\t%f\n" % (";".join(fp[0]), fp[1] / float(len))
            fout.write("%s\t%f\n" % (";".join(fp[0]), fp[1] / float(len)))

def calcFP():
    n = 10000
    freqPaatterns(getTransaction(0, n), n, Config["freqPatterns0"])
    #freqPaatterns(getTransaction(1, n), n, Config["freqPatterns1"])

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
        return self.buildCategoryToFlatBoolVal(rawRow[:-1].split("\t"))

def toFlatBoolVal(data, flatFeatues, big = False):
    fBuilder = FeaturesBuilder()
    i = 0
    with open(data) as fin, open(flatFeatues, "w") as fout:
        for row in fin:
            i+=1
            if big and row[0] == '0':
                continue
            fout.write("\t".join(fBuilder.buildFFromStr(row)) + "\n")
            if i % 500000 == 0:
                print i

def criteriaSelectionTest(load = True):
    cs = criteriaSelection()
    if load:
        cs.statLoad()
    else:
        cs.statDump()
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

    def statDump(self):
        criterionStat = defaultdict(lambda: {"freq":0., "clicks": 0., })
        with open(Config["f_flat_category"]) as fin, open(self.statFName, "w") as fout:
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

def featureImportances():
    model = Config['workdir'] + 'gbclogit_cqf.pkl'
    gbclogit = joblib.load(model)
    d = { i:v for i, v in  enumerate( gbclogit.feature_importances_)}
    print sorted(d.items(), key=operator.itemgetter(1))

def clacErr(xTrain, yTrain, xTest, yTest, model):
    errors = 0
    #tprob = [(exp(lpob[0]), exp(lpob[1])) for lpob in model.predict_log_proba(xTest)]
    err1 = 0
    print xTest
    for predict, y in zip (model.predict(xTest), yTest):
        err = predict - y
        if err != 0:
            #y = 1, predict = 0, err1
            if err < 0:
                #print err, y, prob
                err1 += 1
            errors += 1
    err0 = float(errors - err1)
    err1 = float(err1)
    nTest = float(sum(yTest))
    nc1 = float(sum(yTest))
    nc0 = float(len(yTrain) - nc1)
    print "size train, test = %i\t%i" % (len(yTrain), len(yTest))
    print "1test=%i" % sum(yTest)
    print "1train=%i" % sum(yTrain)
    print "nErr1= %f" % ( err1 / nc1)
    print "nErr0= %f" % ( err0 / nc0 )
    print err0, err1

def classifacation(loadModel = False):
    #gbclogit = GradientBoostingClassifier(max_features="auto", min_samples_leaf=10, subsample=0.3)
    gbclogit = SVC( probability=True)
    ds = DataSets()
    ds.nC0 = 10000
    xTrain, yTrain, xTest, yTest = ds.getTrainTest()
    c1train = sum(yTrain)
    c0train= len(yTrain) - c1train
    print "c1Train=%i" % c1train
    print "c0Train=%i" % c0train
    print "c1/c0 Train=%f" % (float(c1train) / float(c0train))
    print "size train, test = %i\t%i" % (len(yTrain), len(yTest))
    print "c0 Proportion=%i" % ds.c0Proportion
    c0w = 2
    print "c0 weight=%i" % c0w
    model = Config['workdir'] + 'scv.pkl'
    #model = Config['workdir'] + 'gbclogit_cqf.pkl'
    if not loadModel:
        gbclogit.fit(xTrain, yTrain,  sample_weight=[4 if y == 1 else c0w for y in yTrain])
        print gbclogit.score(xTest, yTest)
        joblib.dump(gbclogit,  model)
    else:
        gbclogit = joblib.load(model)
    clacErr(xTrain, yTrain, xTest, yTest, gbclogit)

def linearModel():
    regr  = linear_model.LogisticRegression(class_weight={0:1.9, 1:1})
    model = Config['workdir'] + 'scv.pkl'
    svcm = joblib.load(model)
    model = Config['workdir'] + 'gbclogit_cqf.pkl'
    gbclogit = joblib.load(model)
    
    ds = DataSets()
    ds.nC0 = 100000
    xTrain, yTrain, xTest, yTest = ds.getTrainTest()
    newTrain = svcm.predict(xTrain)
    newTrain = zip(newTrain, gbclogit.predict(xTrain))
    newXtest = svcm.predict(xTest)
    newXtest = zip(newXtest, gbclogit.predict(xTrain))
    print len(newTrain)
    print len(yTrain)
    regr.fit(newTrain, yTrain)
    clacErr(newTrain, yTrain, newXtest, yTest, regr)

def main():
    parser = OptionParser()
    parser.add_option("--fc",
                  help="""
                  1. From imput format to flat category format
                  2. Prune infrequent category in fields greater then 14 
                  Input: 
                  freq30_categorial.txt - see criteo_freq.sh
                  day_0_small_data.txt
                  day_0_big_data.txt
                  
                  day_0_small_data.txt -> features_flat_category.txt
                  day_0_big_data.txt -> features_flat_category_c1.txt
                  From day_0_big_data.txt gives only 1 class rows
                  """,
                  action="store_true", dest="flatCategory"
                  )
    parser.add_option("--fstat",
                      help = """
                      Calc and save frequent category statistics
                      """,
                      action="store_true", dest="statDump")
    parser.add_option("--fs",
                      help = """
                      Select  features.
                      Imput: features_flat_category.txt, features_flat_category_c1.txt
                      Output: features.txt, features_c1.txt 
                      """,
                      action="store_true", dest="featuresSelection")
    parser.add_option("-c",
                  help = """
                  Select  features.
                  Imput:features.txt, features_c1.tx
                  Output: model
                  """,
                  action="store_true", dest="classifacation")
    parser.add_option("--fp",help = """
                  frequent pattern maining
                  """,
                  action="store_true", dest="fp")
    parser.add_option("--lm",help = """
              Final linear model.
              Imput: scv.pkl, gbclogit_cqf.pkl
              Output: features.txt, features_c1.txt 
              """,
                  action="store_true", dest="linearModel")
    (options, args) = parser.parse_args()
    if options.flatCategory:
        print "flatCategory"
        print "%s -> %s" % (Config["datas"], Config["f_flat_category"])
        toFlatBoolVal(Config["datas"], Config["f_flat_category"])
        print "%s -> %s" % (Config["datab"], Config["f_flat_category_c1"])
        toFlatBoolVal(Config["datab"], Config["f_flat_category_c1"], True)
    if options.statDump:
        cs = criteriaSelection()
        cs.statDump()
    if options.featuresSelection:
        print "featuresSelection"
        print "%s -> %s" % (Config["f_flat_category"], Config["features"])
        calcFeatures(Config["f_flat_category"], Config["features"])
        print "%s -> %s" % (Config["f_flat_category_c1"], Config["features_c1"])
        calcFeatures(Config["f_flat_category_c1"], Config["features_c1"])
    if options.classifacation:
        print "Classifacation"
        classifacation(True)
    if options.fp:
        calcFP()
    if options.linearModel:
        linearModel()

if __name__ == "__main__":
    main()
