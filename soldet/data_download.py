#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:10:28 2022

@author: sjguo
"""
import progressbar
import urllib.request


pbar = None
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


url = 'https://data.nist.gov/od/ds/ark:/88434/mds2-2363/data.zip'
url = 'https://data.nist.gov/od/id/mds2-2363'
urllib.request.urlretrieve(url, '/Users/sjguo/SolDet/data.zip')


# OR
# import wget
# wget.download(url)