import numpy as np
import scipy as sc


def numeric_zero(A):
    I,J = A.shape
    B = np.empty(A.shape)
    for i in range(I):
        for j in range(J):
            if(A[i][j] <= 1e-10):
                B[i][j] = 1e-10
            else:
                B[i][j] = A[i][j]
    return B

def real_zero(A):
    I,J = A.shape
    B = np.empty(A.shape)
    for i in range(I):
        for j in range(J):
            if(A[i][j] <= 1e-8):
                B[i][j] = 0
            else:
                B[i][j] = A[i][j]
    return B

def poisson_loss(V, WH):
    log_frac = np.log(np.divide(V, WH))
    return np.sum(np.multiply(V, log_frac)) - np.sum(V) + np.sum(WH)

def Standard_NMF(V, K):
    # Algorithm 1
    N, M = V.shape
    W = np.random.rand(N,K)
    H = np.random.rand(K,M)

    losses = []
    old_loss = 0
    while True:
        #Update H
        num = W.T @ np.divide(V, W@H)
        den = W.T @ np.ones((N,M))
        frac = np.divide(num,den)
        H = np.multiply(H,frac)
        H = np.diag(1 / np.sum(H,axis = 1)) @ H
        


        #Update W
        num = np.divide(V, W@H) @ H.T
        den = (np.ones((N,M)))@H.T
        frac = np.divide(num,den)
        W = np.multiply(W,frac)
        
        H = numeric_zero(H)
        W = numeric_zero(W)
        Vhat = W@H

        new_loss = poisson_loss(V, Vhat)
        losses.append(new_loss)
        print(new_loss)
        if np.abs(old_loss - new_loss) < 1e-5:
            break
        old_loss = new_loss

    return W, H, losses


def Simple_Spatial_NMF(V, K, neighbours, max_iters=10000, eps = 1e-5):
    # Algorithm 2
    N, M = V.shape
    W = np.random.rand(N,K)
    H = np.random.rand(K,M)
    H = np.diag(1 / np.sum(H,axis = 1)) @ H
    neighbours = neighbours + np.eye(N)
    neighbours = numeric_zero(neighbours)

    losses = []
    old_loss = 0
    for i in range(max_iters):
        #Update W
        num = np.divide(V, W@H) @ H.T
        den = (np.ones((N,M)))@H.T
        frac = np.divide(num,den)
        W = np.multiply(W,frac)

        #Update H
        num = W.T @ np.divide(V, W@H)
        den = W.T @ np.ones((N,M))
        frac = np.divide(num,den)
        H = np.multiply(H,frac)
        H = np.diag(1 / np.sum(H,axis = 1)) @ H
        

        # Neighbour average
        W = (neighbours @ W) 
        W =  ((np.diag(1/np.sum(neighbours, axis = 1)))) @ W
        
        H = numeric_zero(H)
        W = numeric_zero(W)

        new_loss = poisson_loss(V, W@H)
        losses.append(new_loss)
        if(np.abs(new_loss - old_loss)<eps):
            break
        old_loss = new_loss

    return W, H, losses
    



def GMRF_loss(X, Q, mu):
    # with constants removed
    loss += - 1/2 * np.log(np.linalg.det(Q)) + 1/2 * (X - mu).T @ Q @ (X - mu)
    return loss

def GMRF_parameter_estimation(spatial, X, n_neighbours = 3, lr_kappa = 0.0001, lr_beta=0.00001, eps = 1e-5):
    # Algorithm 3
    
    N = len(X)
    neigbourhoods = np.empty((N,n_neighbours))
    for i in range(N):
        distances = np.array([np.linalg.norm(spatial[i] - spatial[j]) for j in range(N)])
        neigbourhoods[i] = np.argsort(distances)[1:n_neighbours+1]
    neigbourhoods = neigbourhoods.astype(int)


    neigbours = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i in neigbourhoods[j] and j in neigbourhoods[i]:
                neigbours[i][j] = 1

    # Set row of neighbours to zero if the corrosponding value of X is zero 
    index_zero = np.argwhere(X <= 1e-10)


    log_x = np.log(X)
    log_x[index_zero] = 0
    mu_hat = np.ones(N) * np.mean(log_x) 
    beta_hat = np.random.uniform(-0.1, 0.1)
    kappa_hat = np.random.uniform(0.1, 1)
    Q = np.identity(N) * kappa_hat + (neigbours * kappa_hat * beta_hat)
    losses = []
    old_loss = np.inf
    i = 0
    while True: 
        Q = np.identity(N) * kappa_hat + (neigbours * kappa_hat * beta_hat)
        Q = sc.sparse.csc_matrix(Q)
        Sigma_hat = sc.sparse.linalg.inv(Q).todense()

        # Update kappa
        Q_wrt_kappa = np.identity(N) + neigbours * beta_hat
        kappa_hat_new = kappa_hat +  lr_kappa * 1/2 * (np.trace(Sigma_hat @ Q_wrt_kappa) - (log_x-mu_hat).T @ Q_wrt_kappa @ (log_x-mu_hat))
        if kappa_hat_new <= 0:
            kappa_hat_new = 1e-10    

        # Update beta
        Q_wrt_beta = neigbours * kappa_hat
        beta_hat_new = beta_hat + lr_beta * 1/2 * (np.trace(Sigma_hat @ Q_wrt_beta) - (log_x-mu_hat).T @ Q_wrt_beta @ (log_x-mu_hat))

        if beta_hat_new <= -1 or beta_hat_new >= 1:
            beta_hat_new = 0.999 * np.sign(beta_hat_new)
        
        Q = np.identity(N) * kappa_hat + (neigbours * kappa_hat * beta_hat)

        new_loss = GMRF_loss(X, Q, mu_hat)
        losses.append(new_loss)
        if(np.abs(new_loss - old_loss)<eps):
            break
       
        beta_hat = beta_hat_new
        kappa_hat = kappa_hat_new
    return kappa_hat, beta_hat



def GMRFNMF_loss(V, W, WH, Qs, mus):
    # With constants removed
    N,_ = V.shape
    _,K = W.shape
    log_frac = np.log(np.divide(V, WH))
    loss = np.sum(np.multiply(V, log_frac)) - np.sum(V) + np.sum(WH)
    for i in range(K):
        W_i = W[:,i]
        mu_i = mus[i]
        Q = Qs[i]
        loss += - 1/2 *np.log(np.linalg.det(Q)) + 1/2 * (np.log(W_i) - mu_i).T @ Q @ (np.log(W_i) - mu_i)
    return loss

def Full_GMRFNMF(V, K, neighbours, lr_W=0.00001, lr_H=0.000001, lr_kappa = 0.00001, lr_beta=0.000001, eps = 1e-5):
    # Algorithm 4
    N, M = V.shape
    W = np.random.uniform(0,2,(N,K))
    H = np.random.rand(K,M)
    H = np.diag(1 / np.sum(H,axis = 1)) @ H
    beta_hat = np.random.uniform(-0.5, 0.5,K)
    kappa_hat = np.random.uniform(1, 7,K)
    mu_hat = np.mean(np.log(W), axis=0)
    mus = [np.ones(N) * mu_hat[i] for i in range(K)]
    V = numeric_zero(V)
    W = numeric_zero(W)
    H = numeric_zero(H)
    losses = []
    old_loss = 0

    Qs = []
    Sigmas = []
    Lambda =np.empty((N,K))
    for i in range(K):
        Q = np.identity(N) * kappa_hat[i] + (neighbours * kappa_hat[i] * beta_hat[i])
        Qs.append(Q)
        Q = sc.sparse.csc_matrix(Q)
        Sigma_hat = sc.sparse.linalg.inv(Q).todense()
        Sigmas.append(Sigma_hat)
        Lambda[:,i] = np.divide(Qs[i] @ (np.log(W[:,i]) - mus[i]),W[:,i])

    while True:
        
        #Update H
        H = H + lr_H * (W.T @ (np.divide(V,W@H))- W.T@np.ones((N,M)))
        H = np.diag(1 / np.sum(H,axis = 1)) @ H
        H = numeric_zero(H)
             

        #Update W
        W = W - lr_W * (Lambda -  np.divide(V,W@H)@H.T + np.ones((N,M))@H.T)
            
        W = numeric_zero(W)
        mu_hat = np.mean(np.log(W), axis=0)
        mus = [np.ones(N) * mu_hat[i] for i in range(K)]

        
        
        for i in range(K):
            W = numeric_zero(W)
            log_W = real_zero(np.log(W))
            log_W = log_W[:,i]
            while True: 
                Q = np.identity(N) * kappa_hat[i] + (neighbours * kappa_hat[i] * beta_hat[i])
                Q = sc.sparse.csc_matrix(Q)
                Sigma_hat = sc.sparse.linalg.inv(Q).todense()
                # Update kappa
                Q_wrt_kappa = np.identity(N) + neighbours * beta_hat[i]
                kappa_hat_new = kappa_hat[i] +  lr_kappa * 1/2 * (np.trace(Sigma_hat @ Q_wrt_kappa) - (log_W-mus[i]).T @ Q_wrt_kappa @ (log_W-mus[i]))
                if kappa_hat_new <= 0:
                    kappa_hat_new = 0.01    

                # Update beta
                Q_wrt_beta = neighbours * kappa_hat[i]
                beta_hat_new = beta_hat[i] + lr_beta * 1/2 * (np.trace(Sigma_hat @ Q_wrt_beta) - (log_W-mus[i]).T @ Q_wrt_beta @ (log_W-mus[i]))

                if beta_hat_new <= -1 or beta_hat_new >= 1:
                    beta_hat_new = 0.999 * np.sign(beta_hat_new)
                if np.abs(kappa_hat[i] - kappa_hat_new) < 0.001 and np.abs(beta_hat[i] - beta_hat_new) < 0.001:
                    break
                beta_hat[i] = beta_hat_new
                kappa_hat[i] = kappa_hat_new
                print(f"Index: {i} beta: {beta_hat[i]} kappa: {kappa_hat[i]}")
        
        Qs = []
        Sigmas = []
        Lambda =np.empty((N,K))
        for i in range(K):
            Q = np.identity(N) * kappa_hat[i] + (neighbours * kappa_hat[i] * beta_hat[i])
            Qs.append(Q)
            Q = sc.sparse.csc_matrix(Q)
            Sigma_hat = sc.sparse.linalg.inv(Q).todense()
            Sigmas.append(Sigma_hat)
            Lambda[:,i] = np.divide(Qs[i] @ (np.log(W[:,i]) - mus[i]),W[:,i])
        
        W = numeric_zero(W)
        Vhat = W@H
        Vhat = numeric_zero(Vhat)
        new_loss = GMRFNMF_loss(V, W, Vhat, Qs, mus)
        losses.append(new_loss)
        if np.abs(old_loss - new_loss) < eps:
            break
        old_loss = new_loss
        
    return W, H, kappa_hat, beta_hat, losses



def Simplified_GMRFNMF(V, K, neighbours, lr_W=0.000001, lr_H=0.0000001, kappa_hat = 5, beta_hat=-0.2, eps = 1e-6):
    # Algorithm 5
    N, M = V.shape
    W = np.random.uniform(0,2,(N,K))
    H = np.random.rand(K,M)
    H = np.diag(1 / np.sum(H,axis = 1)) @ H
    mu_hat = np.mean(np.log(W), axis=0)
    mus = [np.ones(N) * mu_hat[i] for i in range(K)]
    V = numeric_zero(V)
    W = numeric_zero(W)
    H = numeric_zero(H)
    losses = []
    old_loss = 0

    Q = np.identity(N) * kappa_hat + (neighbours * kappa_hat * beta_hat)

    Lambda =np.empty((N,K))
    for i in range(K):
        Lambda[:,i] = np.divide(Q @ (np.log(W[:,i]) - mus[i]),W[:,i])

    while True:
        # Update H
        H = H + lr_H * (W.T @ (np.divide(V,W@H))- W.T@np.ones((N,M)))
        #H = np.diag(1 / np.sum(H,axis = 1)) @ H
        H = numeric_zero(H)

        # Update W
        W = W - lr_W * (Lambda -  np.divide(V,W@H)@H.T + np.ones((N,M))@H.T)
        W = numeric_zero(W)
        
        # Update mu
        mu_hat = np.mean(np.log(W), axis=0)
        mus = [np.ones(N) * mu_hat[i] for i in range(K)]

        # Update Lambda
        Lambda =np.empty((N,K))
        for i in range(K):    
            Lambda[:,i] = np.divide(Q @ (np.log(W[:,i]) - mus[i]),W[:,i])
        
        
        Vhat = W@H
        Vhat = numeric_zero(Vhat)
        new_loss = GMRFNMF_loss(V, W, Vhat, Q, mus)
    
        losses.append(new_loss)
        if np.abs(old_loss - new_loss) < eps:
            break
        old_loss = new_loss
        
    return W, H, losses
