# -*- coding: utf-8 -*-
from selenium import webdriver
import json
import time

def salonemilano():
    driver = webdriver.Firefox()
    driver.get("http://salonemilano.it/en-us/EXHIBITORS/Exhibitor-List-2015")
    rb = driver.find_element_by_id("dnn_ctr4578_View_ChkSaloni_repChkList_divChk_2")
    rb.send_keys()
    rb.click()
    abclist = driver.find_element_by_id("dnn_ctr4578_View_divAbc")
    res = []
    refIds = [ref.get_attribute("id") for ref in abclist.find_elements_by_tag_name("a")]
    for refId in refIds:
        print refId
        elem = driver.find_element_by_id(refId)
        print elem.text
        elem.click()
        for e in driver.find_elements_by_class_name("livinchart"):
            compattr = ["exeption"]
            try:
                compattr = e.text.split('\n')
                print compattr[6]
            except ValueError as e:
                print  str(e)
                compattr.append(str(e))
            res.append(compattr)
    with open("salonemilano.txt", "w") as fout:
        fout.write(json.dumps(res))

def automechanika():
    
    res = []
    for p in range(1, 159): 
        urlsStr = "http://automechanika-istanbul.tr.messefrankfurt.com/content/automechanikaistanbul/istanbul/en/besucher/ausstellersuche/exhibitor-product-brand-search.html?aranan=&secenek=1&page=%i#choose" % p
        print urlsStr
        driver = webdriver.Firefox()
        driver.get(urlsStr)
        pagerows = [urlsStr]
        for e in driver.find_elements_by_class_name("search_row"):
            if hasattr(e, "text"):
                pagerows.append(str(e.text))
        driver.close()
        res.append(pagerows)
    print res
    with open("automechanika.json", "w") as fout:
        fout.write(json.dumps(res))

def automechanikaFormater():
    with open("automechanika.json", "r") as fin:
        jsdata = json.load(fin)
        for prows in jsdata:
            print prows[0]
            comp = []
            i = 0
            for comprow in prows[1:]:
                fieldId = i % 10
                if fieldId == 0:
                    if len(comp) != 0:
                        print '\t'.join(comp)
                        comp = []
                if fieldId == 6:
                    comp.append(comprow.replace('\t', ' '))
                i += 1

if __name__ == '__main__':
    #automechanika()
    automechanikaFormater()



