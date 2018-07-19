import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import random

def MCsample(probabilities,N):
    #probabilities is list of classification probabilties as returned by CNN model predict
    #N is number of times to randomly sample from the distribution
    #function samples the probablity distrubtion using psuedo monte carlo method
    totProb = np.sum(probabilities)
    partition = np.append(np.array((0)),np.cumsum(probabilities)/totProb)
    newList = np.zeros((N,))
    for i in range(N):
        tmp = partition
        num = random.uniform(0,1)
        tmp = np.sort(np.append(tmp,num))
        event = np.where(tmp == num)[0][0] - 1
        newList[i] = event
        
    return newList

def main():
    #load dataset
    sns.set(style="whitegrid")
    data = sio.loadmat("C:\\Users\\Luis\\Desktop\\transfers\\exp_results_on_noise10.mat")
    probabilities = data["p_arrays"]
    predictions = data["p_class_list"]
    actual = data["ylist"]

    df = pd.DataFrame()
    for i in range(5):
        sampled_data = MCsample(probabilities[i,:],int(1e5))
        df.loc[0:99999,str(i)] = sampled_data

    sns_plot = sns.violinplot(data=df, inner="box", palette="Set3", cut=2, linewidth=3)

    sns.despine(left=True)
    fig = sns_plot.get_figure()
    fig.savefig('C:/Users/Luis/Desktop/output.png')

    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))

    axes[0, 0].violinplot(sampled_data, [2], points=20, widths=0.3,
                        showmeans=True, showextrema=True, showmedians=True)
    
    fig.savefig('C:/Users/Luis/Desktop/output2.png')




if __name__ == '__main__':
    main()
