#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:25:29 2025

@author: christian
"""

import DLSLib as DL
import os
import subprocess
import matplotlib.pyplot as plt
# <--- This is important for 3d plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from matplotlib.patches import Circle, Polygon
import numpy as np
from tqdm import tqdm
import time
import ilt
import copy
import warnings
import tkinter as tk
import requests
import types
from tkinter import messagebox
import lmfit as lm
from tkinter import ttk
# matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
# %% load data button


def loaddata(rdpath, DLS):
    if rdpath[-1] != '/':
        rdpath += '/'
    dirfiles = os.listdir(rdpath)
    dirs = []
    for folder in dirfiles:
        if os.path.isdir(rdpath + folder):
            dirs.append(folder)
    print('\n######################\n########Files#########\n######################\n')
    print(dirs)
    time.sleep(0.1)
    print('\n######################\n#######Loading########\n######################\n')
    for i in tqdm(range(len(dirs)), desc="Loading…", ascii=False, ncols=75):
        Rawdata = dirs[i]
        dlsdata = []
        files = os.listdir(rdpath+Rawdata)
        files.sort()
        for F in files:
            if '.ASC' in F:
                dlsdata.append(DL.readin(rdpath+Rawdata + '/' + F))
        DLS.append({'Name': Rawdata,
                    'Data': dlsdata})
    # %%% merging different runs
    colors = plt.get_cmap('jet', 13)
    inttresh = 5
    dT = 1.25
    g2m1threshold = 0.05
    g2errorthresholdhight = 0.06
    for DLSmeas in DLS:
        currT = DLSmeas['Data'][0]['T']
        currangle = DLSmeas['Data'][0]['angle']
        currdata = []
        reddata = {}
        DLSmeas.update({'RedData': []})
        for hi, meas in enumerate(DLSmeas['Data']):
            if meas['angle'] in np.arange(30, 160, 10):
                if (((meas['T']-currT)**2 < dT**2 and currangle == meas['angle']) or len(currdata) == 0) \
                        and hi != len(DLSmeas['Data']):
                    currdata.append(meas)
                else:
                    for entr in currdata[0].keys():
                        if entr not in ['g2m1', 'intensities', 'tau']:
                            reddata.update({entr: []})
                            for curd in currdata:
                                reddata[entr].append(curd[entr])
                    reddata.update({'tau': currdata[0]['tau']})
                    discarted = []
                    for hi1, curd in enumerate(currdata):
                        if hi1 == 0:
                            g2m1 = curd['g2m1']
                            intensity = curd['intensities'][:, 1:3]
                        else:
                            g2m1 = np.append(g2m1, curd['g2m1'], axis=1)
                            if len(intensity[:, 0]) == len(curd['intensities'][:, 0]):
                                intensity = np.append(
                                    intensity, curd['intensities'][:, 1:3], axis=1)
                            else:
                                if len(intensity[:, 0]) < len(curd['intensities'][:, 0]):
                                    intensity = np.append(intensity, curd['intensities'][0:len(
                                        curd['intensities'][:, 0])-1, 1:3], axis=1)
                                else:
                                    intensity = np.append(
                                        intensity, intensity[:, 0:2]*10**6, axis=1)
                                    intensity[0:len(
                                        curd['intensities'][:, 0]), -2:] = curd['intensities'][:, 1:3]
                    tempg2m1 = copy.copy(g2m1)
                    # discard low intensity data
                    for hi2 in np.arange(len(g2m1[0, :])-1, -1, -1):
                        if (intensity[:, hi2] < inttresh).any():
                            g2m1 = np.delete(g2m1, np.s_[hi2], axis=1)
                            discarted.append(hi2)
                    # discard data which are > g2m1 treshold for tau>1000
                    for hi2 in np.arange(len(g2m1[0, :])-1, -1, -1):
                        if np.abs(g2m1[reddata['tau'] > 1000, hi2]).mean()/g2m1[reddata['tau'] < 0.001, hi2].mean() > g2m1threshold:
                            g2m1 = np.delete(g2m1, np.s_[hi2], axis=1)
                            discarted.append(hi2)
                    # check if errors are below threshold at high t
                    tempg2 = copy.copy(g2m1)
                    for hi2 in np.arange(len(tempg2[0, :])):
                        tempg2[:, hi2] = tempg2[:, hi2] / \
                            tempg2[reddata['tau'] < 0.001, hi2].mean()
                    while max(tempg2.std(axis=1)[100:-1]) > g2errorthresholdhight:
                        print(max(tempg2.std(axis=1)[100:-1]))
                        idx = DL.selectcurves(tempg2)
                        g2m1 = np.delete(g2m1, np.s_[idx[0]], axis=1)
                        discarted.append(idx[0][0])
                        tempg2 = copy.copy(g2m1)
                        for hi2 in np.arange(len(tempg2[0, :])):
                            tempg2[:, hi2] = tempg2[:, hi2] / \
                                tempg2[reddata['tau'] < 0.001, hi2].mean()
                    if g2m1.size == 0:
                        print('All Data was deleted.... going back to whole data set')
                        g2m1 = tempg2m1
                    # normalize data and write to structure
                    for hi2 in np.arange(len(g2m1[0, :])):
                        g2m1[:, hi2] = g2m1[:, hi2] / \
                            g2m1[reddata['tau'] < 0.001, hi2].mean()
                    reddata.update({'g2m1': g2m1.mean(axis=1),
                                    'dg2m1': g2m1.std(axis=1),
                                    'discarted': np.array(discarted)})
                    DLSmeas['RedData'].append(reddata)
                    reddata = {}
                    currdata = []
                    currdata.append(meas)
                    currT = meas['T']
                    currangle = meas['angle']
    # %%% create files for summaries
    if not os.path.exists('Summaries'):
        os.mkdir('Summaries')
    for DLSmeas in DLS:
        DL.writesummary('Summaries/' + DLSmeas['Name'] + '.tex',
                        r'\documentclass{article}\usepackage{graphicx}\usepackage{hyperref}\begin{document}', perm='w')
        DL.writesummary('Summaries/' + DLSmeas['Name'] + '.tex', r'\author{Automatically Compiled Summary}\title{' + DLSmeas['Name'].replace(
            '_', ' ') + r'}\maketitle\newpage\tableofcontents')
        # %%% determine Temperatures
    Tlimit = []
    for DLSmeas in DLS:
        for meas in DLSmeas['RedData']:
            Tlimit.append((np.array(meas['T'])*2).mean().round()/2)
    Tlimit = sorted(list(set(Tlimit)))
    Tdifference = np.array(Tlimit)[1:] - np.array(Tlimit)[0:-1]
    Tlimits = []
    hi1 = 0
    # merge different temperatures
    for hi, Td in enumerate(Tdifference):
        # print(Td)
        if Td > dT:
            # print(Tlimit[hi1:hi+1])
            Tlimits.append(np.mean(Tlimit[hi1:hi+1]))
            hi1 = hi+1
    Tlimits.append(np.mean(Tlimit[hi1:]))
    return Tlimits
# %% fitting


def fitting(Fittypesstr, DLS, Tlimits,dT):
    colors = plt.get_cmap('jet', 13)
    Fittypes = Fittypesstr.split(',')
    if 'Contin' in Fittypes:
        # %%% Contin algorithm
        # based on https://github.com/caizkun/pyilt
        bound = np.array([0.0001, 100])
        alpha = 1
        print('\n######################\n#######Contin#########\n######################\n')
        for DLSmeas in DLS:
            DL.writesummary(
                'Summaries/' + DLSmeas['Name'] + '.tex', r'\newpage')
            DL.writesummary(
                'Summaries/' + DLSmeas['Name'] + '.tex', r'\section{Contin Analysis}')
            for j in tqdm(range(len(DLSmeas['RedData'])), desc="Contin " + DLSmeas['Name'] + ': ', ascii=False, ncols=100):
                meas = DLSmeas['RedData'][j]
                t = meas['tau']
                F = meas['g2m1']
                z_kww, f_kww, res_lsq, res_reg = ilt.ilt(
                    t, F, bound, len(F), alpha)
                for hid, dec in enumerate(1./z_kww):
                    if hid == 0:
                        decay = z_kww[hid]*f_kww[hid]*np.exp(-t/dec)
                    else:
                        decay += z_kww[hid]*f_kww[hid]*np.exp(-t/dec)
                decay = decay/decay[0]
                Contin = {'t': 1./z_kww,
                          'distrib': z_kww*f_kww,
                          'res_lsq': res_lsq,
                          'res_reg': res_reg,
                          'decay': decay}
                meas.update({'Contin': Contin})
        # %%% plot Contin
        # Tlimits=np.arange(285,319,2.5)
        if not os.path.exists('figures'):
            os.mkdir('figures')
        if not os.path.exists('figures/contin'):
            os.mkdir('figures/contin')
        containscurve = False
        for Tl in Tlimits:
            for DLSmeas in DLS:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for meas in DLSmeas['RedData']:
                    if np.abs(np.mean(meas['T'])-Tl) < dT:
                        ax.scatter(np.ones(np.shape(meas['Contin']['t']))*np.mean(meas['q'])**2,
                                   np.log10(meas['Contin']['t']), meas['Contin']['distrib'], marker='o',
                                   color=colors(int(np.arange(0, 13)[np.arange(30, 160, 10) == np.mean(meas['angle'])])))
                    if not np.shape([]) == np.shape(meas['Contin']['t']):
                        containscurve = True
                ax.set_title(str(Tl))
                ax.set_xlabel(r'$q^2$ [\AA$^{-2}$]')
                ax.set_ylabel(r'log$_{10}\left(t\right)$')
                ax.set_zlim([0, 1])
                if containscurve:
                    fig.savefig(
                        'figures/contin/' + DLSmeas['Name'] + str(Tl) + '.pdf', bbox_inches='tight')
                    DL.writesummary(
                        'Summaries/' + DLSmeas['Name'] + '.tex', r'\includegraphics[width=.4\textwidth]{'+'../figures/contin/' + DLSmeas['Name'] + str(Tl) + '.pdf'+'} ')
                plt.close()
        Fittypes.remove('Contin')
    for fit in Fittypes:
        if not os.path.exists('figures/'+fit):
            os.mkdir('figures/'+fit)
    for DLSmeas in DLS:
        DL.writesummary('Summaries/' + DLSmeas['Name'] + '.tex', r'\newpage')
        DL.writesummary(
            'Summaries/' + DLSmeas['Name'] + '.tex', r'\section{Fits}')
        for j in tqdm(range(len(DLSmeas['RedData'])), desc="Fit " + DLSmeas['Name'] + ': ', ascii=False, ncols=100):
            meas = DLSmeas['RedData'][j]
            meas.update({'Fit': []})
            for fit in Fittypes:
                plt.errorbar(meas['tau'], meas['g2m1'], meas['dg2m1'], fmt='o',
                             color=colors(int(np.arange(0, 13)[np.arange(30, 160, 10) == np.mean(meas['angle'])])), alpha=0.1)
                plt.plot(meas['tau'], meas['g2m1'], 'o',
                         color=colors(int(np.arange(0, 13)[np.arange(30, 160, 10) == np.mean(meas['angle'])])))
                model, param = DL.returnfitmodel(fit)
                param.add('q', value=np.array(meas['q']).mean(), vary=False)
                try:
                    FR = model.fit(
                        meas['g2m1'], param, x=meas['tau'], weights=1/meas['dg2m1'], nan_policy='omit')
                except:
                    FR = model.fit(meas['g2m1'], param,
                                   x=meas['tau'], nan_policy='omit')
                Fit = {'Name': fit}
                for p in FR.best_values.keys():
                    Fit.update({'LMFit': FR})
                    Fit.update({'d'+p: FR.params[p].stderr})
                    Fit.update({p: FR.params[p].value})
                Fit.update({'redchi': FR.redchi})
                meas['Fit'].append(Fit)
                x = np.logspace(np.log10(min(meas['tau'])), np.log10(
                    max(meas['tau'])), 100)
                # color=colors(int(np.arange(0,13)[np.arange(30,160,10)==np.mean(meas['angle'])]))
                plt.plot(x, FR.eval(x=x), color='k', zorder=10)
                name = DLSmeas['Name'] + ' ' + \
                    str(np.mean(meas['q']))+' ' + str(np.mean(meas['T']))
                plt.xscale('log')
                plt.xlabel(r'$\tau$ [ms]')
                plt.ylabel(r'g$_2$-1')
                plt.title(r'$q$='+str(format(np.array(meas['q']).mean(
                ), '.4f')) + r'\AA$^{-1};~redchi=$' + str(format(np.array(FR.redchi), '.4f')))
                plt.savefig('figures/' + fit + '/' + name +
                            '.pdf', bbox_inches='tight')
                plt.close()
    for fit in Fittypes:
        DL.writesummary(
            'Summaries/' + DLSmeas['Name'] + '.tex', r'\subsection{' + fit + '}')
        for Tl in Tlimits:
            DL.writesummary(
                'Summaries/' + DLSmeas['Name'] + '.tex', r'\subsubsection{' + fit + ':' + str(Tl)+'K}')
            for meas in DLSmeas['RedData']:
                if np.abs(np.mean(meas['T'])-Tl) < 1.25:
                    print('condition fulfilled')
                    name = DLSmeas['Name'] + ' ' + \
                        str(np.mean(meas['q']))+' ' + str(np.mean(meas['T']))
                    DL.writesummary(
                        'Summaries/' + DLSmeas['Name'] + '.tex', r'\includegraphics[width=.4\textwidth]{../figures/' + fit + '/' + name + '.pdf}')
                else:
                    print('do nothing')
                    print(Tlimits)
# %% plot logo


def plotlogo(ax):
    # Create S-curve for snake body
    t = np.linspace(0, 2 * np.pi, 1000)
    x = np.sin(t) * 2
    y = np.sign(np.pi - t) * (1 - np.cos(t-np.pi)) * 2

    # Reverse the snake so the head is at the top
    x = -x[::-1]
    y = y[::-1]
    n = 100
    x = np.concatenate((-x[:n], x[:]))
    y = np.concatenate((y[:n], y[:]))
    x = np.concatenate((x[:], -x[-n:]))
    y = np.concatenate((y[:], y[-n:]))
    # Plot setup
    # fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # ax.axis('off')

    # Draw the snake body
    for i in range(len(x)-1):
        color = cm.Greens(0.6 + 0.4 * (i / len(x)))
        circle = Circle((x[i], y[i]), radius=0.3, color=color, zorder=1)
        ax.add_patch(circle)
    # Draw the head
    xh = x[-n]
    yh = y[-n]
    head = Circle((xh, yh), radius=0.5, color='green', zorder=2)
    ax.add_patch(head)

    # Eyes
    eye1 = Circle((xh, yh + 0.07), radius=0.15, color='white', zorder=3)
    # eye2 = Circle((xh + 0.07, yh + 0.07), radius=0.03, color='white', zorder=3)
    pupil1 = Circle((xh, yh + 0.07), radius=0.05, color='black', zorder=4)
    # pupil2 = Circle((xh + 0.07, yh + 0.07), radius=0.01, color='black', zorder=4)

    ax.add_patch(eye1)
    # ax.add_patch(eye2)
    ax.add_patch(pupil1)
    # ax.add_patch(pupil2)

    # Tongue (triangle)
    tongue = Polygon([[xh, yh],
                      [xh + 1, yh - 0.35],
                      [xh, yh - 0.35]],
                     closed=True, color='red', zorder=1)
    ax.add_patch(tongue)

    # Adjust view limits to fit the full snake
    # padding = 0.5
    # ax.set_xlim(np.min(x) - padding, np.max(x) + padding)
    # ax.set_ylim(np.min(y) - padding, np.max(y) + 1.0)  # Extra top space for tongue
    # plt.xlim([-5,5])
    # plt.ylim([-5,5])
    font_size = 180
    ax.text(-12.5, -1.5, 'D', fontsize=font_size,
            va='center', ha='center', color='blue')
    ax.text(-5, -1.5, 'L', fontsize=font_size,
            va='center', ha='center', color='blue')
    ax.text(-15, -0.5, 'py', fontsize=65, color='green')

    # Adjust view limits to fit the full snake and letters
    ax.set_xlim([-20, 5])
    ax.set_ylim([-5, 5])
    ax.axis('off')  # Optional: turn off axes for a cleaner cartoon look

#%% zenodo
# ACCESS_TOKEN='UqeVLZpCFHh2zSUzRkhKOYUDVtZRtDem3kpYa62kRKeVl3NH8pQwRWZXOg1B' # for sandbox
# file='t.hdf'
# Metadata for your dataset
# metadata = {
#     'metadata': {
#         'title': 'My Dataset Title',
#         'upload_type': 'dataset',
#         'description': 'A description of my dataset.',
#         'creators': [{'name': 'Doe, John', 'affiliation': 'University X'}]
#     }
# }
def get_metadata_from_user(master,Status):
    """
    Opens a modal Tkinter Toplevel window to collect:
    - Access Token (pre-filled if Status['AccessToken'] exists)
    - Title
    - Description
    - Creators

    Returns a dict with keys:
    'access_token', 'title', 'description', 'creators'
    or None if window closed/canceled.
    """
    result = {}

    def on_done():
        access_token = access_token_entry.get().strip()
        title = title_entry.get().strip()
        description = description_entry.get("1.0", tk.END).strip()
        creators = creators_entry.get().strip()

        if not access_token or not title or not description or not creators:
            messagebox.showwarning("Missing Fields", "Please fill out all fields before continuing.")
            window1.lift()
            window1.focus_force()
            return

        result['access_token'] = access_token
        result['title'] = title
        result['description'] = description
        result['creators'] = [{'name':creators}]
        window1.destroy()

    window1 = tk.Toplevel(master)
    window1.title("Enter Metadata and Access Token")
    window1.resizable(False, False)
    window1.update_idletasks()
    window1.grab_set()

    frame = tk.Frame(window1, padx=10, pady=10)
    frame.pack()

    # Access Token
    tk.Label(frame, text="Access Token:").grid(row=0, column=0, sticky="w")
    access_token_entry = tk.Entry(frame, width=50)
    access_token_entry.grid(row=1, column=0, pady=(0, 10))
    if Status and 'AccessToken' in Status:
        access_token_entry.insert(0, Status['AccessToken'])

    # Title
    tk.Label(frame, text="Title:").grid(row=2, column=0, sticky="w")
    title_entry = tk.Entry(frame, width=50)
    title_entry.grid(row=3, column=0, pady=(0, 10))

    # Description
    tk.Label(frame, text="Description:").grid(row=4, column=0, sticky="w")
    description_entry = tk.Text(frame, width=50, height=4)
    description_entry.insert(tk.END, "Dataset from DLS analyzed with pyDLS")
    description_entry.grid(row=5, column=0, pady=(0, 10))

    # Creators
    tk.Label(frame, text="Creators (e.g. Doe, John):").grid(row=6, column=0, sticky="w")
    creators_entry = tk.Entry(frame, width=50)
    creators_entry.grid(row=7, column=0, pady=(0, 10))

    # Done button
    tk.Button(frame, text="Done", command=on_done).grid(row=8, column=0, pady=(10, 0))

    window1.update_idletasks()
    # window1.geometry(f"{window1.winfo_reqwidth()}x{window1.winfo_reqheight()}")

    master.wait_window(window1)

    return result if result else None

def uploadZenodo(ACCESS_TOKEN,metadata,file,DLS):

    ZENODO_URL = 'https://zenodo.org/api/deposit/depositions'
    md={'metadata':{'upload_type':'dataset'}}
    for key in ['creators','description','title']:
        md['metadata'].update({key:metadata[key]})
    # Step 1: Create a new deposition
    r = requests.post(ZENODO_URL,
                      params={'access_token': ACCESS_TOKEN},
                      json=md)
    r.raise_for_status()
    deposition = r.json()
    deposition_id = deposition['id']
    doi = deposition['metadata']['prereserve_doi']['doi']
    for temp in DLS:
        temp.update({'doi':doi})
    DL.savehdf(file.replace('.hdf','_doi.hdf'), DLS)
    # Step 2: Upload your file
    with open(file.replace('.hdf','_doi.hdf'), 'rb') as fp:
        files = {'file': (file.replace('.hdf','_doi.hdf'), fp)}
        r = requests.post(f'{ZENODO_URL}/{deposition_id}/files',
                          params={'access_token': ACCESS_TOKEN},
                          files=files)
        r.raise_for_status()
    
    # Step 3: Publish it (optional – this makes it public and assigns a DOI)
    r = requests.post(f'{ZENODO_URL}/{deposition_id}/actions/publish',
                      params={'access_token': ACCESS_TOKEN})
    r.raise_for_status()

    return DLS,doi
# %% custom fit function


def customfitmodel(independent='q', name='Fickian', functstr='lambda q,D: D*q**2'):
    custommodel = tk.Toplevel()
    custommodel.geometry("600x400")  # width x height
    custommodel.title(f"Custom Model: independent variable = {independent}")

    # One line of text
    tk.Label(custommodel, text="Enter the model name:", font=("Arial", 11)).grid(
        row=0, column=0, columnspan=4, pady=5, sticky='w')

    # Entry field
    entry_field1 = tk.Entry(custommodel, width=80)
    entry_field1.grid(row=1, column=0, columnspan=4,
                      padx=5, pady=5, sticky='w')
    entry_field1.insert(0, name)
    tk.Label(custommodel, text="Enter the desired expression as lambda function:", font=(
        "Arial", 11)).grid(row=2, column=0, columnspan=4, pady=5, sticky='w')

    # Entry field
    entry_field = tk.Entry(custommodel, width=80)
    entry_field.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky='w')
    entry_field.insert(0, functstr)
    result = {"model": None, "params": None, "param": None, 'Name': ''}

    def load_parameters():
        modelstring = entry_field.get()
        try:
            func = eval(modelstring)
            if isinstance(func, types.LambdaType) and func.__name__ == "<lambda>":
                print("Valid lambda function")
                model = lm.Model(func, independent_vars=[independent])
                result["model"] = model
                result["params"] = model.param_names
                result['Name'] = entry_field1.get()
                # Update table
                tree.delete(*tree.get_children())
                for param in result["params"]:
                    tree.insert("", "end", values=(param, "", "", ""))
            else:
                print("Not a lambda")
        except Exception as e:
            print("Invalid input:", e)
    tk.Button(custommodel,
              text="Obtain Parameters",
              command=load_parameters,
              font=("Arial", 11)
              ).grid(row=4, column=0, columnspan=4, pady=5, sticky='w')

    # Table (Treeview)
    tree = ttk.Treeview(custommodel, columns=(
        "param", "value", "min", "max"), show="headings", height=6)
    for col in ("param", "value", "min", "max"):
        tree.heading(col, text=col.capitalize())
        tree.column(col, width=80 if col == "param" else 60, anchor='center')
    tree.grid(row=5, column=0, columnspan=4, padx=5, pady=5)

    # Double-click to edit cell
    def on_double_click(event):
        region = tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        row_id = tree.identify_row(event.y)
        col = tree.identify_column(event.x)
        col_index = int(col[1:]) - 1

        if col_index == 0:
            return  # First column not editable

        x, y, width, height = tree.bbox(row_id, col)
        value = tree.set(row_id, tree["columns"][col_index])

        entry = tk.Entry(custommodel)
        entry.place(x=tree.winfo_rootx() + x - custommodel.winfo_rootx(),
                    y=tree.winfo_rooty() + y - custommodel.winfo_rooty(),
                    width=width, height=height)
        entry.insert(0, value)
        entry.focus()

        def save_value(event):
            tree.set(row_id, tree["columns"][col_index], entry.get())
            entry.destroy()

        entry.bind("<Return>", save_value)
        entry.bind("<FocusOut>", lambda e: entry.destroy())

    tree.bind("<Double-1>", on_double_click)

    # Done button in lower-right
    def on_submit():
        table = [tree.item(i)["values"] for i in tree.get_children()]
        params = lm.Parameters()
        for line in table:
            print(line)
            val = line[1]
            if val == '':
                val = 0
            else:
                val = float(val)
            lim1 = line[2]
            if lim1 == '':
                lim1 = -np.inf
            else:
                lim1 = float(lim1)
            lim2 = line[3]
            if lim2 == '':
                lim2 = np.inf
            else:
                lim2 = float(lim1)
            limits = [lim1, lim2]
            params.add(line[0], value=val, min=min(limits), max=max(limits))
        result['param'] = params
        custommodel.quit()
        custommodel.destroy()

    tk.Button(custommodel,
              text="Done",
              command=on_submit,
              font=("Arial", 12),
              width=15,
              padx=10, pady=5
              ).grid(row=10, column=3, sticky='e', pady=10, padx=10)

    custommodel.mainloop()
    return result["model"], result["param"], result["Name"]
