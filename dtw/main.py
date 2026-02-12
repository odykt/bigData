import numpy as np
import argparse
import ast
import csv

import time
from tqdm import tqdm


def getInput(filename):

    ifp = open(filename, mode="r")
    csvReader = csv.reader(ifp)

    next(csvReader) #ignore first line

    for row in csvReader:
        if not row:
            continue

        seriesID, seriesA, seriesB = row

        seriesA = ast.literal_eval(seriesA)
        seriesB = ast.literal_eval(seriesB)

        X = np.array(seriesA, dtype=float)
        Y = np.array(seriesB, dtype=float)

        yield int(seriesID), X, Y

    ifp.close()


def getDTW(X, Y):
    n = len(X)
    m = len(Y)

    Darr = np.full((n+1,m+1),np.inf,dtype=float) #when comparing and getting min, inf helps not skew results
    Darr[0,0] = 0.0

    for i in range(1,n+1):
        x_i = X[i-1]
        for j in range(1, m+1):
            #dist = (x_i-Y[j-1])**2 #euclidean
            dist = abs(x_i-Y[j-1])
            Darr[i,j] = dist+min(Darr[i-1,j],Darr[i,j-1],Darr[i-1,j-1])

    return float(Darr[n,m])


def produceOutputCSV(ifpPath, ofpPath, rows=None):
    start = time.perf_counter() #let's track time

    ofp = open(ofpPath,mode="w",newline="")
    try:
        csvWriter = csv.writer(ofp)
        csvWriter.writerow(["id","DTW distance"])

        data = getInput(ifpPath)
        for seriesID, X, Y in tqdm(data, total=rows, desc="Calculating DTW", unit="pair"):
            dist = getDTW(X,Y)
            csvWriter.writerow([seriesID,dist])

    finally:
        ofp.close()

    end = time.perf_counter()
    print("Computation is done! Runtime: "+str(end-start))






if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("ifpPath",type=str)
    parser.add_argument("ofpPath",type=str)
    parser.add_argument("rows",type=int)

    args = parser.parse_args()
    
    produceOutputCSV(args.ifpPath,args.ofpPath,args.rows)
