#!/usr/bin/python
# -*- coding: utf-8 -*-
import requests
import urllib
import urllib2
from bs4 import BeautifulSoup
import ssl
import json
import time
from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
from PIL import Image

class InstaImageScraper:

    def __init__(self):
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE


    def getlinks(self, hashtag):
        try:
            url = 'https://www.instagram.com/explore/tags/' + hashtag + '/'
            links = []
            html = urllib.urlopen(url, context=self.ctx).read()
            soup = BeautifulSoup(html, 'html.parser')
            script = soup.find('script', text=lambda t: \
                            t.startswith('window._sharedData'))
            page_json = script.text.split(' = ', 1)[1].rstrip(';')
            data = json.loads(page_json)
            print ('Scraping links with #' + hashtag+"...........")
            for post in data['entry_data']['TagPage'][0]['graphql'
                    ]['hashtag']['edge_hashtag_to_media']['edges']:
                image_src = post['node']['thumbnail_resources'][1]['src']
                links.append(image_src)
            return links
        except:
            return []

    def scrap(self, hashtag, number_of_images):
        links = set()
        while len(links) < number_of_images:
            new_links = self.getlinks(hashtag)
            for new_link in new_links:
                links.add(new_link)
            print(len(links))
            time.sleep(1)
        
        return list(links)[:number_of_images]

class GoogleImageScraper:        

    def scrap(self, searchtext, number_of_images):
        url = "https://www.google.co.in/search?q="+searchtext+"&source=lnms&tbm=isch"

        display = Display(visible=0, size=(800, 600))
        display.start()
        driver = webdriver.Firefox()
        driver.get(url)

        number_of_scrolls = number_of_images / 400 + 1 
        for _ in xrange(number_of_scrolls):
            for __ in xrange(10):
                # multiple scrolls needed to show all 400 images
                driver.execute_script("window.scrollBy(0, 1000000)")
                time.sleep(0.2)
                # to load next 400 images
            time.sleep(0.5)
            try:
                driver.find_element_by_xpath("//input[@value='Więcej wyników']").click()
            except Exception as e:
                print "Less images found:", e
                break

        imges = driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]')
      
        links = []
        for img in imges:
            img_url = json.loads(img.get_attribute('innerHTML'))["ou"]
            img_type = json.loads(img.get_attribute('innerHTML'))["ity"]
            try:
                if img_type != 'gif':
                    links.append(img_url)
            except Exception as e:
                print "Download failed:", e
            if len(links) >= number_of_images:
                break
        driver.close()
        driver.quit()
        return links

data_path = 'data/'
links_file = data_path + 'metadata/links'

def ConvertImage(img_name, width, height):
    im = Image.open(img_name)
    im.convert('RGB').resize((width, height)).save(img_name, "JPEG")

def DownloadSingleFile(fileURL, fileName):
    print 'Downloading image...'
    resp = urllib2.urlopen(fileURL)
    with open(fileName, 'wb') as f:
        f.write(resp.read())
    print 'Done. Image saved to disk as ' + fileName

def GetFileNames():
    return [file for file in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, file))]

def DownloadFiles(urls):
    prev_names =  GetFileNames()
    numbers = [int(name) for name in prev_names]
    numbers.sort()
    max_number = 0
    if len(numbers) > 0:
        max_number = numbers[-1]

    for i, url in enumerate(urls):
        try:
            fileName = data_path + str(i + max_number + 1)
            DownloadSingleFile(url, fileName)
            ConvertImage(fileName, 240, 240)
        except:
            print "Failed"

def LoadLinks():
    if not os.path.exists(links_file):
        open(links_file, 'a').close()
    with open(links_file, 'r') as f:
        return f.read().splitlines()

def SaveNewLinks(links):
    with open(links_file, 'a') as f:
        for link in links:
            f.write(link)
            f.write('\n')

def GetDistinctNewLinks(links):
    old_links = set(LoadLinks())
    res = []
    for link in links:
        if link not in old_links:
            old_links.add(link)
            res.append(link)
    return res

def RemoveFailedDownloads(old_files_number, new_files_number):
    names = GetFileNames()
    curr_number = old_files_number + 1
    for i in range(old_files_number +1, old_files_number + new_files_number + 1):
        try:
            img = Image.open(data_path + str(i))
            print "renaming ", str(i)
            os.rename(data_path + str(i), data_path + str(curr_number))
            curr_number += 1
        except:
            if os.path.isfile(data_path + str(i)): 
                print "REMOVING ", str(i)
                os.remove(data_path + str(i))

def ensure_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def PrepareDirs():
    ensure_dir(data_path)
    ensure_dir(data_path + 'metadata/')

def ScrapImages(page_to_scrap, text_to_find, number_of_images):
    PrepareDirs()
    scrapper = GoogleImageScraper()
    if page_to_scrap == "Instagram":
        scrapper = InstaImageScraper()
    print "SCRAPING..."
    links = GetDistinctNewLinks(scrapper.scrap(text_to_find, number_of_images))
    print "Found ", len(links), " new images"
    old_files_number = len(GetFileNames())
    DownloadFiles(links)
    SaveNewLinks(links)
    RemoveFailedDownloads(old_files_number, len(links))

ScrapImages('Google', 'stado psow', 50)
