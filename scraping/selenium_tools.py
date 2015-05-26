from selenium import webdriver
import json
if __name__ == '__main__':
    driver = webdriver.Firefox()
    driver.get("http://salonemilano.it/en-us/EXHIBITORS/Exhibitor-List-2015")
    abclist = driver.find_element_by_id("dnn_ctr4578_View_divAbc")
    res = []
    refIds = [ref.get_attribute("id") for ref in abclist.find_elements_by_tag_name("a")]
    for refId in refIds:
        print refId
        elem = driver.find_element_by_id(refId)
        print elem.text
        elem.click()
        for e in driver.find_elements_by_class_name("livinchart"):
            compattr = e.text.split('\n')
            print compattr[6]
            res.append(compattr)
    with open("salonemilano.txt", "w") as fout:
        fout.write(json.dumps(res))

