# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from keras import backend as K
from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render
from django.conf import settings
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import os
import h5py
import numpy as np
import argparse
from google_images_download import google_images_download
from googlesearch import search as ims
import requests
import re
import urllib2
import urllib
import os
import argparse
import sys
import json
from extract_cnn_vgg16_keras import VGGNet
import shutil
import numpy as np
import h5py
import urllib2
import re
import os
from os.path import basename
from urlparse import urlsplit
from urlparse import urlparse as up
from urlparse import urlunparse
from posixpath import basename,dirname
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from bs4 import BeautifulSoup as bs
import os
import sys
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import pusher
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'texim')



pusher_client = pusher.Pusher(
  app_id='650084',
  key='67d4a4100ac7bd39e18f',
  secret='b771da9ec6664de0e850',
  cluster='ap2',
  ssl=True
)


# Create your views here.
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.svg')]


def remove_string_special_characters(s):
    stripped = re.sub('[^\w\s]','',s)
    stripped = re.sub('_','', stripped)

    stripped = re.sub('\s+','',stripped)

    stripped = stripped.strip()
    return stripped

def count_words(sent):
    count = 0
    words = word_tokenize(sent)
    for word in words:
        count +=1
    return count


def get_doc(sent):
    i = 0
    doc_info = []
    for doc in sent:
        i+=1
        temp = {'doc_id':i,'doc_length':count_words(doc)}
        doc_info.append(temp)
    return doc_info


def create_freq_dicts(sents):
    i=0
    fdl = []
    for sent in sents:
        i+=1
        freq_dict = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            if word in freq_dict:
                freq_dict[word]+=1
            else:
                freq_dict[word]=1
            temp = {'doc_id':i, 'freq_dict':freq_dict}
        fdl.append(temp)
    return fdl

def computeTF(doc_info,fdl,query):
    tfs = []
    for td in fdl:
        id = td['doc_id']
        for k in td['freq_dict']:
            if k == query:
                temp = {'doc_id':id,'TF_score':float(td['freq_dict'][k])/float(doc_info[id-1]['doc_length']),'key':k}
                tfs.append(temp)
    return tfs

def computeIDF(doc_info,fdl,query):
    ids = []
    for dic in fdl:
        id = dic['doc_id']
        for k in dic['freq_dict'].keys():
            if k == query:
                c = sum([k in tempDict['freq_dict'] for tempDict in fdl])
                temp = {'doc_id':id,'IDF_score':math.log(len(doc_info)/c),'key':k}
                ids.append(temp)
    return ids

def computeTFIDF(tfs,ids):
    tfids = []
    for j in ids:
        for i in tfs:
            if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                temp = {'doc_id':j['doc_id'],'TFIDF_score':j['IDF_score']*i['TF_score'],'key':i['key']}
                tfids.append(temp)
    return tfids


def search_form(request):
	print(BASE_DIR)
	return render(request, 'texim/index.html')



@csrf_exempt
def search(request):
	query = request.POST['message']
	max_images = 20
	save_directory = os.path.join(BASE_DIR,'database')
	query_directory = os.path.join(BASE_DIR,'query')
	image_type="Action"

	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "          Downloading Training images"
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})

	if query not in os.listdir(save_directory):
		response = google_images_download.googleimagesdownload()   #class instantiation
		arguments = {"keywords":query,"limit":max_images,"print_urls":True,"output_directory":save_directory}   #creating list of arguments
		paths = response.download(arguments)


	db = os.path.join(save_directory,query)
	img_list = get_imlist(db)

	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "          feature extraction starts"
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	
	
	feats = []
	names = []

	model = VGGNet()
	for i, img_path in enumerate(img_list):
		try:
			norm_feat = model.extract_feat(img_path)
			img_name = os.path.split(img_path)[1]
			feats.append(norm_feat)
			names.append(img_name)
			print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))
			msg = "Extracting feature from image No."+str(i+1)+" images in total "+str(len(img_list))
			pusher_client.trigger('texim', 'my-event', {'message': msg})
		except Exception:
			print "Skipping Unexpected Error:", sys.exc_info()[1]
			msg = "Skipping Unexpected Error:" + str(sys.exc_info()[1])
			pusher_client.trigger('texim', 'my-event', {'message': msg})
			pass

	feats = np.array(feats)
	names = np.array(names)
	# print(feats)
	# directory for storing extracted features
	# output = os.path.join(BASE_DIR,'feature.h5')

	print("--------------------------------------------------")
	print("      writing feature extraction results ...")
	print("--------------------------------------------------")

	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "      writing feature extraction results ..."
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})


	# FEATURE.h5
	# h5f = h5py.File(output, 'w')
	# h5f.create_dataset('dataset_1', data = feats)
	# # h5f.create_dataset('dataset_2', data = names)
	# h5f.create_dataset('dataset_2', data = names)
	# h5f.close()

	# # read in indexed images' feature vectors and corresponding image names
	# h5f = h5py.File(output,'r')
	# # feats = h5f['dataset_1'][:]
	# feats = h5f.get('dataset_1')
	# # print(feats)
	# feats = np.array(feats)
	# #imgNames = h5f['dataset_2'][:]
	# imgNames = h5f.get('dataset_2')
	# # print(imgNames)
	# imgNames = np.array(imgNames)
	#h5f.close()

	# print(feats)
	# print(imgNames)
	        
	print("--------------------------------------------------")
	print("               searching starts")
	print("--------------------------------------------------")

	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "             searching starts"
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})

	# read and show query image

	sites = []
	N = 5

	#Google search
	for url in ims(query, stop=13):
		print(url)
		sites.append(url)
	sites = sites[:N]
	print(sites)

	# sites = ['https://www.cars.com/',]
	total_img_scores = []
	doc_dic = []
	for site in sites:
		try:
			soup = bs(urllib2.urlopen(site),"html5lib")
			drc = ""
			for p in soup.find_all('p'):
			    drc+=p.getText()
			doc_dic.append(drc)
		except Exception:
			pass


	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "          Ranking documents on basis of tf-idf scores "
	pusher_client.trigger('texim', 'my-event', {'message': msg})
	msg = "--------------------------------------------------"
	pusher_client.trigger('texim', 'my-event', {'message': msg})


	doc_info = get_doc(doc_dic)
	fdl = create_freq_dicts(doc_dic)
	TF_score = computeTF(doc_info,fdl,query)
	IDF_score = computeIDF(doc_info,fdl,query)
	TFIDF_scores = computeTFIDF(TF_score,IDF_score)

	total_doc_scores = [0 for x in range(len(sites))]

	for el in TFIDF_scores:
		total_doc_scores[el['doc_id']-1] = el['TFIDF_score']

	total_doc_scores = np.array(total_doc_scores)
	total_doc_scores.reshape((1, -1))
	rank_ID2 = np.argsort(total_doc_scores)[::-1]
	rank_score2 = total_doc_scores[rank_ID2]
	maxres = N
	doclist = [sites[index] for i,index in enumerate(rank_ID2[0:maxres])]
	print("doclist")
	print(doclist)
	print(rank_score2)

	pusher_client.trigger('results', 'my-event', {"doclist":doclist})



	for site in sites:
		try:
			soup = bs(urllib2.urlopen(site),"html5lib")
			img_tags = soup.find_all('img')
			print(img_tags)


			queryDir = os.path.join(query_directory,str(sites.index(site)))
			os.mkdir(queryDir)
			print("directory created")

			urls = []
			for img in img_tags:
				try:
					urls.append(img['src'])
				except Exception:
					pass

			msg = "--------------------------------------------------"
			pusher_client.trigger('texim', 'my-event', {'message': msg})
			msg = "          Downloading Query Images for Site-"+str(sites.index(site)+1)
			pusher_client.trigger('texim', 'my-event', {'message': msg})
			msg = "--------------------------------------------------"
			pusher_client.trigger('texim', 'my-event', {'message': msg})

			for url in urls:
				filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
				try:
					if 'http' not in url:
					    url = '{}{}'.format(site, url)
					imgdata=urllib2.urlopen(url).read()
					filname=basename(urlsplit(url)[2])
					output=open(os.path.join(queryDir,filname),'wb')
					output.write(imgdata)
					output.close()
				except Exception:
					print "Skipping Unexpected Error:", sys.exc_info()[1]
					pass


			img_list = get_imlist(queryDir)
			qfeats = []
			qnames = []

			model = VGGNet()
			for i, img_path in enumerate(img_list):
				try:
					norm_feat = model.extract_feat(img_path)
					img_name = os.path.split(img_path)[1]
					qfeats.append(norm_feat)
					qnames.append(img_name)
				except Exception:
					print "Skipping Unexpected Error:", sys.exc_info()[1]
					pass


			qfeats = np.array(qfeats)
			qnames = np.array(qnames)


			msg = "--------------------------------------------------"
			pusher_client.trigger('texim', 'my-event', {'message': msg})
			msg = "          Calculating Image Score for Site-"+str(sites.index(site)+1)
			pusher_client.trigger('texim', 'my-event', {'message': msg})
			msg = "--------------------------------------------------"
			pusher_client.trigger('texim', 'my-event', {'message': msg})


			model = VGGNet()

			# extract query image's feature, compute simlarity score and sort
			if qfeats.any():
				scores = []
				scores = np.array(scores)
				for qD in feats:
				#qV = model.extract_feat(qD)
					if scores.any():
						scores += np.dot(qD, qfeats.T)
					else:
						scores = np.dot(qD,qfeats.T)
			else:
				scores = [0]
				scores = np.array(scores)

			total_img_scores.append(np.sum(scores))
		except Exception:
			scores = [0]
			scores = np.array(scores)
			total_img_scores.append(np.sum(scores))
			pass

    
	total_img_scores = np.array(total_img_scores)
	total_img_scores.reshape((1, -1))
	rank_ID1 = np.argsort(total_img_scores)[::-1]
	rank_score1 = total_img_scores[rank_ID1]
	maxres = N
	imlist = [sites[index] for i,index in enumerate(rank_ID1[0:maxres])]
	print("imlist")
	print(imlist)
	print(rank_score1)
	shutil.rmtree(query_directory)
	os.mkdir(query_directory)
	image_type="Action"

	

	final_scores = [sum(x) for x in zip(total_img_scores, total_doc_scores)]
	final_scores = np.array(final_scores)
	final_scores.reshape((1, -1))
	rank_ID3 = np.argsort(final_scores)[::-1]
	rank_score3 = final_scores[rank_ID3]

	totlist = [sites[index] for i,index in enumerate(rank_ID3[0:maxres])]
	print("totlist")
	print(totlist)
	print(rank_score3)

	pusher_client.trigger('results', 'my-event', {"totlist":totlist})
	K.clear_session()
	return render(request,'texim/search_results.html',{"totlist":totlist,"doclist":doclist})