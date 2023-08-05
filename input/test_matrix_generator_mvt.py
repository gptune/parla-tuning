import numpy as np
import os

def generate_problem_mvt(n=2**12, d=2**10, df=1, lr=False):

    # copied code snippet from https://github.com/lessketching/newtonsketch
    # with slight modifications for GA matrix (df=np.inf) and output types

    def cov_mat(d):
        Sigma = np.zeros((d,d))
        for ii in range(d):
            for jj in range(d):
                Sigma[ii,jj] = 2 * 0.5**(np.abs(ii-jj))
        return Sigma 

    def mvt(n_samples, Sigma, df):
        d = len(Sigma)
        if df == np.inf:
            Z = np.random.multivariate_normal(np.zeros(d), Sigma, n_samples)
            return Z
        else:
            g = np.tile(np.random.gamma(df/2., 2./df, n_samples), (d,1)).T
            Z = np.random.multivariate_normal(np.zeros(d), Sigma, n_samples)
            return Z / np.sqrt(g)

    A = mvt(n, cov_mat(d), df=df)
    x_pl = np.ones((d,1))
    x_pl[10:-10] = 0.1
    b = A @ x_pl + 0.09 * np.random.randn(n,1)
    _, sigma, _ = np.linalg.svd(A, full_matrices=False)
    condition_number = sigma[0] / sigma[-1]
    A = A / sigma[0]
    b = b / sigma[0]
    if lr:
        b = np.sign(b)
    return A, b.ravel()
    #return torch.tensor(A), torch.tensor(b)

def gen_problem(n=10000, d=200, df=1):

    A, b = generate_problem_mvt(n=n, d=d, df=df)

    n_rows, n_cols = A.shape

    if df == np.inf:
        mattype = "GA"
    else:
        mattype = "T"+str(df)

    with open("synthetic_mvt/data-nrows_"+str(n_rows)+"-ncols_"+str(n_cols)+"-mattype_"+str(mattype)+".csv", "w") as f_out:
        f_out.write("#matrix_nrows:"+str(n_rows)+", ncols:"+str(n_cols)+", mattype:"+str(mattype)+"\n")
        for i in range(n_rows):
            f_out.write(str(i))
            for j in range(n_cols):
                f_out.write("," + str(A[i][j]))
            f_out.write("\n")

    with open("synthetic_mvt/result-nrows_"+str(n_rows)+"-ncols_"+str(n_cols)+"-mattype_"+str(mattype)+".csv", "w") as f_out:
        f_out.write("#matrix_nrows:"+str(n_rows)+", ncols:"+str(n_cols)+", mattype:"+str(mattype)+"\n")
        for i in range(len(b)):
            f_out.write(str(i)+","+str(b[i])+"\n")

    return

if __name__ == "__main__":

    os.system("mkdir -p synthetic_mvt")

    gen_problem(n=10000, d=1000, df=1)
    gen_problem(n=10000, d=1000, df=3)
    gen_problem(n=10000, d=1000, df=5)
    gen_problem(n=10000, d=1000, df=10)
    gen_problem(n=10000, d=1000, df=np.inf)

    gen_problem(n=50000, d=1000, df=1)
    gen_problem(n=50000, d=1000, df=3)
    gen_problem(n=50000, d=1000, df=5)
    gen_problem(n=50000, d=1000, df=10)
    gen_problem(n=50000, d=1000, df=np.inf)

