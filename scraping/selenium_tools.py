import time
from selenium import webdriver
 
if __name__ == '__main__':
    driver = webdriver.Firefox()
    driver.get("http://salonemilano.it/en-us/EXHIBITORS/Exhibitor-List-2015")
    elem = driver.find_element_by_id("dnn_ctr4578_View_repLettere_lnklettere_26")
    print elem.text
    elem.click()
    for e in driver.find_elements_by_class_name("inserisci"):
        print e.text
