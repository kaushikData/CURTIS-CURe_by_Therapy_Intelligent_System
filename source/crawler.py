import configparser
import requests
import logging
from bs4 import BeautifulSoup
import pandas as pd
import re
import sys
import argparse

#loding configuration
config = configparser.ConfigParser()
config.read('config.ini')
website_config = config['WEBSITE']
logging_config = config['LOGGING']

#metadata of the data to be crawled
data_columns = [
"questionID",
"questionTitle",
"questionText",
"questionLink",
"topic",
"therapistName",
"therapistTitle",
"therapistURL",
"answerText",
"upvotes",
"views"]

def remove_HTML_from_string(rawText):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', rawText)
    return cleantext

def find_question_item_url(questionItem):
    return website_config['WEBSITE_MAIN_URL'] + questionItem.find("a", {"class": "question-title"}).attrs['href']

def get_question_urls_for_a_topic(topic):
    logging.info('Enter get_question_urls_for_a_topic') 
    result = []
    count = 1
    while (count==1 or questionItems != []):
        url = website_config['WEBSITE_MAIN_URL'] + website_config['TOPICS_URL'] + "/" + topic + "?page=" + str(count)
        logging.debug("Topic URL: " + url) 
        page = requests.get(url)    
        soup = BeautifulSoup(page.text, 'html.parser')
        questionItems = soup.find("div", {"class": "list-group list-question-group"}).find_all("div", {"class": "item-question"})                         
        questionUrls = list(map(find_question_item_url,questionItems))    
        logging.debug("Question URLs")
        logging.debug(questionUrls)         
        result+=questionUrls         
        count += 1

    logging.debug("Result of get_question_urls_for_a_topic")
    logging.debug(result)    
    logging.info('Exit get_question_urls_for_a_topic') 
    return result

def write_list_into_file(filename,listItems):
    logging.info('Enter write_list_into_file')     
    with open(filename, "a") as filehandle:
        for listItem in listItems:
            filehandle.write('%s\n' % listItem)
    logging.info('Exit write_list_into_file')

def read_list_from_file(filename):    
    logging.info('Enter read_list_from_file')     
    url_list = []    
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]            
            url_list.append(currentPlace)            
    logging.info('Exit read_list_from_file')
    return url_list    
        
def crawl_questions():    
    logging.info('Enter crawl_questions') 
    topics = website_config["TOPICS"].split(",")
    logging.debug("Topics: "+str(topics))
    for topic in topics:
        url_list = get_question_urls_for_a_topic(topic)
        write_list_into_file("questions/"+topic+".txt",url_list)
    logging.info('Exit crawl_questions') 


def get_string_from_list(listItems):
    result = ""
    for listItem in listItems:
        result += listItem.text
    return result

def get_question_and_answer_details(topic,questionId,url):
    logging.info('Enter get_question_and_answer_details') 
    result = []    
    page = requests.get(url)    
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find("body").find("div",{"id":"content"})
    
    # question
    questionDetailHTML = content.find("div",{"class":"row"})
    questionTitle = remove_HTML_from_string(questionDetailHTML.find("h1",{"class":"page-title"}).text) 
    logging.debug("QuestionTitle: "+ questionTitle)        
    questionTextHTML = questionDetailHTML.find("div",{"class":"page-description"})
    paragraphs = questionTextHTML.findAll("p")    
    questionText = remove_HTML_from_string(get_string_from_list(paragraphs))
    logging.debug("QuestionText: "+ questionText)  
    # answers
    answersDetails = content.find_all("div", {"class": "item-answer"})
    logging.debug("answerDetails: ") 
    logging.debug(answersDetails) 


    for answerDetail in answersDetails:
        # question and answer row    
        dataItem = []        
        # 1. questionID
        dataItem.append(questionId)
        # 2. questionTitle 
        dataItem.append(questionTitle)
        # 3. questionText 
        dataItem.append(questionText)
        # 4. questionLink
        dataItem.append(url)
        # 5. topic
        dataItem.append(topic)                
        therapistDetail = answerDetail.find("div",{"class":"therapist-summary"}).find("div",{"class":"name-title"})
        # 6. therapistName
        dataItem.append(therapistDetail.find("a",{"class":"name"}).text)
        # 7. therapistTitle
        dataItem.append(therapistDetail.find("div",{"class":"title"}).text)
        # 8. therapistURL
        dataItem.append(website_config['WEBSITE_MAIN_URL'] + therapistDetail.find("a",{"class":"name"}).attrs['ng-href'])
        answerParagraphsList = answerDetail.find("div",{"class":"description"}).findAll("p") 
        # # 9. answerText
        answerText = remove_HTML_from_string(get_string_from_list(answerParagraphsList))

        #removing extraspaces, single and double quotes        
        cleanText = ' '.join(answerText.split()).replace("'","").replace("\"","")                        
        dataItem.append(cleanText)

        actions = answerDetail.find("div",{"class":"actions"}).findAll("li")
        # 10. upvotes
        dataItem.append(int(actions[0].find("a").attrs['ng-init'].split("=")[1].strip()))
        # # 11. views
        dataItem.append(int(actions[1].find("span").text.split(' ')[0]))        
        result.append(dataItem)

    logging.debug("Result of get_question_and_answer_details: ")      
    logging.debug(result)      
    logging.info('Exit get_question_and_answer_details') 
    return result
    

def crawl_answers():
    logging.info('Enter crawl_answers')       
    topics = website_config["TOPICS"].split(",")
    logging.debug("Topics: "+str(topics))
    topic_count = 1
    for topic in topics:    
        logging.debug("Crawl for Topic: " + topic)
        print ("Topic:  "+str(topic_count)+"/"+str(len(topics)))        
        logging.info("Topic:  "+str(topic_count)+"/"+str(len(topics))) 
        df = pd.DataFrame(columns=data_columns)        
        url_list = read_list_from_file("questions/"+topic+".txt")
        count = 1
        for url in url_list:                     
            logging.info('Crwaling URL: '+str(count)+"/"+str(len(url_list)))      
            logging.info("Crawl for URL: " + url)                    
            dataItems = get_question_and_answer_details(topic,count,url)
            for dataItem in dataItems:
                df.loc[len(df)] = dataItem                
            count+=1
        logging.debug('Output dataframe of a topic') 
        logging.debug(df) 
        df.to_excel("output/"+topic+".xlsx")  
        topic_count+=1
    logging.info('Exit crawl_answers') 

def crawl_questions_and_answers():
    crawl_questions() 
    crawl_answers()

def main():    
    #logging
    logging.basicConfig(filename=logging_config["LOG_FILE_NAME"],level=logging.INFO,filemode="w")
    logging.info('Application Started')
    # crawl_questions() 
    crawl_answers()
    # crawl_questions_and_answers()
    logging.info('Application Ended')

if __name__ == '__main__':
    main()