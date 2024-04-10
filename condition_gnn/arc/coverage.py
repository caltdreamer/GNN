import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def wsc(X, y, S, delta=0.1, M=1000, random_state=2020, verbose=False):
    rng = np.random.default_rng(random_state)

    def wsc_v(X, y, S, delta, v):
        #print(X.shape)
        #print(y.shape)
        #print(len(S))
        n = len(y)
        cover = np.array([y[i] in S[i] for i in range(n)])
        #print(f'length is {n}')
        #print(n)
        #print(f'coverage is {cover.sum()}')
        z = np.dot(X,v)
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0-delta)*n))
        ai_best = 0
        bi_best = n-1
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai+int(np.round(delta*n)),n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
            coverage[np.arange(0,bi_min-ai)]=1
            bi_star = ai+np.argmin(coverage)
            cover_star = coverage[bi_star-ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        #print(cover_min)
        #print(ai_best)
        #print(bi_best)
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]
    
    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T
    V = sample_sphere(M, p=X.shape[1])
    
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    #print('wqwqqw')
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, S, delta, V[m])
    else:
        #print('iehkwd')
        for m in range(M):
            #print(m)
            #print(X.shape)
            #print(y.shape)
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, S, delta, V[m])
            #print(m)
    #print('ikfrk')
    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star

def wsc_unbiased(X, y, S, delta=0.1, M=1000, test_size=0.75, random_state=2020, verbose=False):
    #print('iqhd')
    def wsc_vab(X, y, S, v, a, b):
        n = len(y)
        cover = np.array([y[i] in S[i] for i in range(n)])
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        coverage = np.mean(cover[idx])
        return coverage
    #print('skjdh')
    
    max_attempts = 5000
    for attempt in range(max_attempts):
        X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=0.75, random_state=attempt)
        cover = np.array([y_train[i] in S_train[i] for i in range(len(y_train))])
        if not all(cover):
            break

    if all(cover):
        print('May cause problem')
    #print(len(y_train))
    #print(cover)
    #print('dasytrdv')
    #print(X_train.shape)
    #print(X_test.shape)
    ##print(y_train.shape)
    #print(y_test.shape)
    #print(len(S_train))
    #print(len(S_test))
    #print(S_train)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(X_train, y_train, S_train, delta=delta, M=M, random_state=random_state, verbose=verbose)
    #print('ewtddvzdas')
    #print(v_star)
    #print(a_star)
    #print(b_star)
    # Estimate coverage
    coverage = wsc_vab(X_test, y_test, S_test, v_star, a_star, b_star)
    return coverage
