#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:15:04 2022

@author: Christian Beck

christianbeck91@gmx.de

ORCID: https://orcid.org/0000-0001-7214-3447

"""
import numpy as np
import time
import os as os
from lmfit import Model, Parameters
from lmfit.model import save_model, ModelResult
# import lmfit as lm
import pickle
import warnings
import h5py
import re
import scipy as sp
from lmfit.model import load_modelresult
import tempfile
warnings.filterwarnings("ignore")

#%% different functions
# save to pickle file
def save(FileName,Data):
    with open(FileName, "wb") as file:
        pickle.dump(Data, file, pickle.HIGHEST_PROTOCOL)
# load data
def load(Filename):
    with open(Filename, "rb") as file:
        loaded_dict = pickle.load(file)
    return loaded_dict
#%% save to h5py
def savehdf(name,dataset):
    f=h5py.File(name,'w')
    path='/'
    if isinstance(dataset,dict):
        savehdfdict(dataset,path,f)
    elif isinstance(dataset,list):
        savehdflist(dataset,path,f)
    f.close()
    
def savehdflist(liste,path,f):
    for hi,listentry in enumerate(liste):
        if isinstance(listentry,dict):
            savehdfdict(listentry, path + 'Dataset_' + str(hi) +'/', f)
        elif isinstance(listentry,list):
            savehdflist(listentry, path + 'Dataset_' + str(hi) +'/', f)
        else:
            # print(path + 'entry' + str(hi) +'/')
            # print('liste: ' + path)
            # print(hi)
            # print(listentry)
            # try:
            f.create_dataset(name=path + 'entry' + str(hi),data=listentry)
            # except:
            #     f.create_dataset(name='/Dataset_0/RedData/Dataset_0/angle/',data=np.array(30))
            #     print('hardcoded')
                    

def savehdfdict(dictionary,path,f):
    # print('dict: ' + path)
    for key in dictionary.keys():
        if isinstance(dictionary[key],dict):
            savehdfdict(dictionary[key],path + key + '/',f)
        elif isinstance(dictionary[key],list):
            savehdflist(dictionary[key],path + key + '/',f)
        elif isinstance(dictionary[key],(str,float,np.ndarray)):
            # print(type(dictionary[key]))
            # print(key)
            # print(path + 'entry/'+key)
            f.create_dataset(name=path + 'entry/'+key,data=dictionary[key])
            if dictionary[key] is None:
                dictionary[key]=np.nan
            f.create_dataset(path + key,data=dictionary[key])
        elif isinstance(dictionary[key],ModelResult):
            save_model(dictionary[key], 'temp_result.txt')
            with open("temp_result.txt", "r") as file:
                content = file.read()
            f.create_dataset(name=path + 'entry/'+key +'/LMFit', data=np.bytes_(content))
            os.remove("temp_result.txt")
        else:
            print(f'skipping {type(dictionary[key])}')
            
# def loadhdf(Filename):
#     with h5py.File(Filename, "r") as f:
#         a_group_key = list(f.keys())[0]
#         print('loading of hdf5 file is not yet implemented')
#%% based on chatgpt laoding routine for hdf
def loadhdf(filename):
    with h5py.File(filename, 'r') as f:
        return _loadhdf_group(f, '/')

def _loadhdf_group(f, path):
    keys = list(f[path].keys())

    # If group has Dataset_# keys, treat as list
    dataset_keys = [key for key in keys if re.match(r'Dataset_\d+', key)]
    if dataset_keys:
        result = []
        for i in sorted(dataset_keys, key=lambda x: int(x.split('_')[1])):
            item = _loadhdf_group(f, path + i + '/')

            # 🔄 Flatten {"entry0": value} to value
            if isinstance(item, dict) and list(item.keys()) == ["entry0"]:
                item = item["entry0"]

            result.append(item)
        return result

    # Otherwise treat as dict
    result = {}
    for key in keys:
        subpath = path + key
        if isinstance(f[subpath], h5py.Group):
            if key == 'entry':
                for entry_key in f[subpath].keys():
                    entry_path = subpath + '/' + entry_key
                    if entry_key == 'LMFit':
                        try:
                            result = _load_modelresult_from_string(f[entry_path][()])
                        except:
                            result[entry_key] = _loadhdf_dataset(f, entry_path)
                    else:
                        result[entry_key] = _loadhdf_dataset(f, entry_path)
            else:
                result[key] = _loadhdf_group(f, subpath + '/')
        elif isinstance(f[subpath], h5py.Dataset):
            if key != 'entry':
                result[key] = _loadhdf_dataset(f, subpath)
    return result

def _loadhdf_dataset(f, path):
    data = f[path][()]
    if isinstance(data, bytes):
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return data
    return data

def _load_modelresult_from_string(model_string):
    if isinstance(model_string, bytes):
        model_string = model_string.decode('utf-8')
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        tmp.write(model_string)
        tmp.flush()
        return load_modelresult(tmp.name)

#%% alternative way of implementation
# def savehdf(name,dataset):
#     f=h5py.File(name,'w')
#     if isinstance(dataset,dict):
#         savehdfdict(dataset,f)
#     elif isinstance(dataset,list):
#         savehdflist(dataset,f)
#     f.close()
# def savehdflist(liste,group):
#     for hi,listentry in enumerate(liste):
#         if isinstance(listentry,dict):
#             grp=group.create_group('Dataset_' + str(hi))
#             savehdfdict(listentry,grp)
#         elif isinstance(listentry,list):
#             grp=group.create_group('Dataset_' + str(hi))
#             savehdflist(listentry, grp)
#         else:
#             print('')
#             # print(liste)
#             # try:
#             #     group['value']=liste
#             # except:
#             #     print(group.name +' already exists...')
            
# def savehdfdict(dictionary,group):
#     #print('dict: ' +  group.name)
#     for key in dictionary.keys():
#         if isinstance(dictionary[key],dict):
#             grp=group.create_group(key)
#             savehdfdict(dictionary[key],grp)
#         elif isinstance(dictionary[key],list):
#             grp=group.create_group(key)
#             savehdflist(dictionary[key],grp)
#         else:
#             if dictionary[key] is None:
#                 dictionary[key]=np.nan
#             # print('==========')
#             # print(dictionary[key])
#             # print(key)
#             # print(group.name)
#             # group[key]=dictionary[key]
        
#%%
# readin data
def readin(file):
    f=open(file,'r',encoding="ISO-8859-1")
    FC={}
    dat=0
    for line in f:
        if dat==0 and line.__contains__(":"):
            idx=line.find(":")
            FC.update({line[0:idx-1]:line[idx+1:-1]})
        elif not line.__contains__(":") and line.__contains__('"'):
            dat=1
            dattype=line.replace('"','')
            DAT=np.array([])
        elif dat==1:
            if len(DAT)==0:
                try:
                    DAT=[np.array(line.split(),dtype=float)]
                except:
                    time.sleep(0)
            elif 'Monitor' in line:
                FC.update({'monitor':float(line[15:])})
            else:
                try:
                    DAT=np.append(DAT,[np.array(line.split(),dtype=float)], axis=0)
                except:
#                    print('##############')
#                    if 'Monitor' in line:
#                        FC.update({'monitor':float(line[15:])})
#                        print('Monitor added' + str(FC['monitor']))
#                    else:
                    time.sleep(0)
                FC.update({dattype:DAT})
    f.close()
    data={"angle": float(FC['Angle [°]      ']),
          "intensities":FC['Count Rate\n'],
          "g2m1":FC['Correlation\n'][:,1:3],
          "tau":FC['Correlation\n'][:,0],#ms
          "T":float(FC['Temperature [K]']),
          "filename":file,
          "Date":str(FC['Date']).replace("\t", ''),
          "Time":str(FC['Time']).replace("\t", ''),
          "Monitor":FC['monitor']}
    wl=6328
    data.update({'q':np.array(4*np.pi*1.332/wl*np.sin(data['angle']/360*np.pi))})#A^-1
    return data
def writesummary(filename,line,perm='a'):
    with open(filename,perm) as f:
        f.write(line)
        f.write('\n')
    print('written to file: '+ line)
def selectcurves(g2m1):
    errors=[]
    for hi2 in np.arange(len(g2m1[0,:])):
        errors.append(np.delete(g2m1,np.s_[hi2],axis=1).mean(axis=1)[100:-1])
    errors=np.array(errors)
    #print(errors)
    idx=np.where(errors==errors.max())
    return idx
#%% define fit functions
def gaussdistexpnorm(x,w1,sig):
    # see https://www.wolframalpha.com/input?i=int_0%5Einf%281%2F%282*pi*s%5E2%29%5E0.5*exp%28-%28x-g%29%5E2%2F2%2Fs%5E2%29exp%28-x+t%29%29dx
    
    y= (0.5*np.exp(0.5*x*(sig**2*x-2*w1))*(sp.special.erf((w1-sig**2*x)/(2*sig**2)**0.5)+1))**2
    y[np.isnan(y)]=0
    y=y/y[0]
    return y
def normalandstretchednorm(x,a1,w1,w2,b):
    return a1*np.exp(-2*w1*x)+(1-a1)*np.exp(-(2*w2*x)**b)
def stretchedstretchednorm(x,a1,w1,w2,b1,b2):
    return a1*np.exp(-(2*w1*x)**b1)+(1-a1)*np.exp(-(2*w2*x)**b2)
def doubleexponential(x,a1,a2,w1,w2):
    return a1*np.exp(-2*w1*x)+a2*np.exp(-2*w2*x)
def singleexponentialnorm(x,w1):
    return np.exp(-2*w1*x)
def singleexponential(x,w1,a):
    return a*np.exp(-2*w1*x)
def doubleexponentialnorm(x,a1,w1,w2):
    return a1*np.exp(-2*w1*x)+(1-a1)*np.exp(-2*w2*x)
def Dq2(x,D):
    return x*D
def returnfitmodel(name):
    if name=='doubleexponential':
        model = Model(doubleexponential)
        params = Parameters()
        params.add('a1', value=0.50, min=0, max=1)
        params.add('a2', value=0.50, min=0, max=1)
        params.add('w1', value=0.50, min=0)
        params.add('delta', value=0.50, min=0)
        params.add('w2', expr='w1+delta')
    elif name=='gaussdistexpnorm':
        model = Model(gaussdistexpnorm)
        params = Parameters()
        params.add('w1', value=10, min=0)
        params.add('scaling',value=0.5,min=0,max=1.1)
        params.add('sig', expr='scaling*w1')
    elif name=='singleexponentialnorm':
        model = Model(singleexponentialnorm)
        params = Parameters()
        params.add('w1', value=0.50, min=0)
    elif name=='singleexponential':
        model = Model(singleexponential)
        params = Parameters()
        params.add('w1', value=0.50, min=0)
        params.add('a', value=1, min=0)
    elif name=='doubleexponentialnorm':
        model = Model(doubleexponentialnorm)
        params = Parameters()
        params.add('a1', value=0.50, min=0, max=1)
        params.add('w1', value=0.50, min=0)
        params.add('delta', value=0.50, min=0)
        params.add('w2', expr='w1+delta')
    elif name=='normalandstretchednorm':
        model = Model(normalandstretchednorm)
        params = Parameters()
        params.add('a1', value=0.50, min=0, max=1)
        params.add('w1', value=0.50, min=0)
        params.add('delta', value=0.50, min=0)
        params.add('w2', expr='w1+delta')
        params.add('b',  value=1, min=0)
    elif name=='stretchedstretchednorm':
        model = Model(stretchedstretchednorm)
        params = Parameters()
        params.add('a1', value=0.50, min=0, max=1)
        params.add('w1', value=0.50, min=0)
        params.add('delta', value=0.50, min=0)
        params.add('w2', expr='w1+delta')
        params.add('b1',  value=1, min=0)
        params.add('b2',  value=1, min=0)
    elif name=='Dq2':
        model=Model(Dq2)
        params = Parameters()
        params.add('D', value=0.50, min=0)
    return model,params