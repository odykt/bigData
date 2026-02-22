import numpy as np
import argparse
import ast
import csv
import time
from tqdm import tqdm
import math
import numba as nb

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


@nb.njit
def getDTW(X, Y):
    n = len(X)
    m = len(Y)

    Darr = np.full((n+1,m+1),np.inf,dtype=float) #when comparing and getting min, inf helps not skew results
    Darr[0,0] = 0.0

    for i in range(1,n+1):
        x_i = X[i-1]
        for j in range(1, m+1):
            dist = math.fabs(x_i-Y[j-1])
            Darr[i,j] = dist+min(Darr[i-1,j],Darr[i,j-1],Darr[i-1,j-1])
    return float(Darr[n,m])



@nb.njit
def getDTW_Sakoe(X, Y, R=40):
    n = len(X)
    m = len(Y)

    # optional fast exit: if band covers everything, use full DTW logic
    if R >= max(n, m):
        return getDTW(X, Y)

    Darr = np.full((n+1, m+1), np.inf, dtype=np.float64)
    Darr[0, 0] = 0.0

    for i in range(1, n+1):
        x_i = X[i-1]

        # center band on scaled diagonal
        j_center = int(i * m / n)
        j_start = max(1, j_center - R)
        j_end   = min(m, j_center + R)

        for j in range(j_start, j_end+1):
            dist = math.fabs(x_i - Y[j-1])

            a = Darr[i-1, j]
            b = Darr[i, j-1]
            c = Darr[i-1, j-1]

            if b < a:
                a = b
            if c < a:
                a = c

            Darr[i, j] = dist + a

    return float(Darr[n, m])


@nb.njit
def getDTW_Itakura(X, Y, s=2.0):
    n = len(X)
    m = len(Y)


    Darr = np.full((n+1, m+1), np.inf, dtype=np.float64)
    Darr[0, 0] = 0.0

    for i in range(1, n+1):
        x_i = X[i-1]

        # Itakura parallelogram bounds (slope constraint)
        # lower bound grows ~ i/s, but also must reach the end in time
        j_start = int(i / s)
        j2 = int(m - (n - i) * s)
        if j2 > j_start:
            j_start = j2

        # upper bound grows ~ i*s, but also cannot overshoot end constraint
        j_end = int(i * s)
        j2 = int(m - (n - i) / s)
        if j2 < j_end:
            j_end = j2

        # clamp to valid j range [1, m]
        if j_start < 1:
            j_start = 1
        if j_end > m:
            j_end = m

        # if bounds cross, no valid cells on this row (leave as inf)
        if j_start > j_end:
            continue

        for j in range(j_start, j_end+1):
            dist = math.fabs(x_i - Y[j-1])

            a = Darr[i-1, j]
            b = Darr[i, j-1]
            c = Darr[i-1, j-1]

            if b < a:
                a = b
            if c < a:
                a = c

            Darr[i, j] = dist + a

    return float(Darr[n, m])




def produceOutputCSV(ifpPath, ofpPath, rows=None, use_sakoe=False, R=10, use_itakura=False, s=2.0):
    start = time.perf_counter() #let's track time

    ofp = open(ofpPath,mode="w",newline="")
    try:
        csvWriter = csv.writer(ofp)
        csvWriter.writerow(["id","DTW distance"])

        data = getInput(ifpPath)
        for seriesID, X, Y in tqdm(data, total=rows, desc="Calculating DTW", unit="pair"):
            dist=None
            if(use_sakoe):
                dist = getDTW_Sakoe(X,Y,R)
            elif(use_itakura):
                dist = getDTW_Itakura(X,Y,s)
            else:
                dist = getDTW(X,Y)
            csvWriter.writerow([seriesID,dist])

    finally:
        ofp.close()

    end = time.perf_counter()
    print("Computation is done! Runtime: "+str(end-start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ifpPath", type=str)
    parser.add_argument("ofpPath", type=str)
    parser.add_argument("rows", type=int)

    # mutually exclusive constraints
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sakoe", action="store_true",
                       help="Use Sakoe-Chiba constraint")
    group.add_argument("--itakura", action="store_true",
                       help="Use Itakura parallelogram constraint")

    parser.add_argument("--R", type=int, default=40,
                        help="Band width for Sakoe-Chiba (default=40)")
    parser.add_argument("--s", type=float, default=2.0,
                        help="Slope parameter for Itakura (default=2.0)")

    args = parser.parse_args()

    produceOutputCSV(
        args.ifpPath,
        args.ofpPath,
        args.rows,
        use_sakoe=args.sakoe,
        use_itakura=args.itakura,
        R=args.R,
        s=args.s
    )


