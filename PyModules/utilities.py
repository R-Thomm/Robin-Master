import math
import random
import threading
import numpy as np
import hashlib
import time
from tqdm import tqdm_notebook

def wait(t_wait, n_up=100):
    #print('wait for %.1f seconds: '%t_wait, end=' ', flush=True)
    t0 = time.time()
    t1 = t0+t_wait
    with tqdm_notebook(total=t_wait, desc='waiting: ') as tq: 
        while(True):
            tn=time.time()
            tq.update(round(tn-t0,1) - tq.n)
            #print('%2.0f%%'%((tn-t0)/t_wait*100), end=' ', flush=True)
            if tn>t1:
                break
            time.sleep(t_wait/n_up)
    #print('done')

def hash(name):
    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
    hasher = hashlib.sha3_256()
    with open(name, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()

def do_async(func,  *args, **kwargs):
    thr = threading.Thread(target=func, args=args, kwargs=kwargs)
    thr.start()
    return thr

def grad(ps,zs):
	g = [0]*len(ps[0])
	for i in range(1,len(ps)):
		dp = ps[i]-ps[0]
		dz = zs[i]-zs[0]
		nrm = norm(dp)
		df = dz/nrm
		ep = dp/nrm
		g += df*ep
	return g
	
def norm(vec):
	n = 0
	for v in vec:
		n += v**2
	return np.sqrt(n)
	
def normalize(vec):
	return vec/norm(vec)

def optimize_minimize(func, x0, limits=None, tol=1e-6, cons=None, maxfev=200, **kwds):
    '''
    idx=[2,3]
    cons = ({'type': 'ineq', 'fun': lambda pos: max([max_step-abs(self.mirror_pos(i)-p) for i,p in zip(idx,pos)])},
            {'type': 'eq', 'fun': lambda pos: max([p-int(p) for p in pos])})
    func = lambda pos: self.opt_mirror_script(pos=pos,idx=idx,script='BD')/4.
    '''
    from scipy.optimize import minimize
    res = minimize(func, x0, tol=tol, bounds=limits, constraints=cons, method='SLSQP', options={'disp': True, 'maxiter':maxfev})
    return res
    
def optimize_iminuit(func, x0, limits=None, tol=1e-6, cons=None, maxfev=200, **kwds):
    import iminuit
    fixes=[False,False]
    x0_err = [20,20]
    m = iminuit.Minuit.from_array_func(func, tuple(x0), error=tuple(x0_err), fix=tuple(fixes), limit=tuple(limits), errordef=20, print_level=2)
    #print(m.name)
    #print(m.lower_limit)
    #print(m.upper_limit)
    fmin, param = m.migrad(ncall=maxfev);
    return fmin, param


def NEIGHBORS(p,inc,limits):
    points = [p, [p[0]-inc,p[1]], [p[0]-inc,p[1]+inc], [p[0],p[1]+inc], [p[0]+inc,p[1]+inc], [p[0]+inc,p[1]], [p[0]+inc,p[1]-inc], [p[0],p[1]-inc], [p[0]-inc,p[1]-inc]]
    for i in range(len(points)):
        for j,l in enumerate(limits):
            points[i][j] = int(max(min(points[i][j],l[1]),l[0]))
    return points

def integer_hill_climb(func,x0=[0,0],limits=[[-200,200],[-200,200]],inc=20,maxfev=200,to=10,verbose=True):
    i=0
    k=0
    p = x0

    max_z = -float('inf')
    max_p = None
    while(i<maxfev):
        L = NEIGHBORS(p, inc, limits)
        zs = []
        for x in L:
            z = func(x)
            zs.append(z)
            if (z > max_z):
                max_z = z
                max_p = x
                if verbose:
                    print('iter %i (stag %i): cnt=%.2f p=(%i,%i) stp=%i'%(i,k,z,*x,inc))
                k = 0
        p = max_p
        i+=1
        k+=1
        if k>2:
            inc = max(int(inc/2),1)
        if k>to:
            break
    end_iter = i>=maxfev
    end_stag = k>to
    print('final iter: %i, end: max eval %r, max stag %r'%(i,end_iter,end_stag))
    return max_p, max_z, end_iter, end_stag

def guess(p,inc,limits):
	points = [p,[p[0],p[1]+inc],[p[0]+inc,p[1]]]
	for i in range(len(points)):
		for j,l in enumerate(limits):
			points[i][j] = min(max(points[i][j],l[0]),l[1])
	return np.array(points)

def optimize_gradient(func,x0,init_inc,scale,limits,precision,maxiter):
	p=x0
	inc = init_inc
	z_min_tot = float('inf')
	p_min_tot = x0
	i=0
	while(i<maxiter):
		v=[]
		zs = []
		points=guess(p,inc,limits)
		for l in points:
			z=func(l)
			zs.append(z)
		zs = np.array(zs)
		idx = np.argmin(zs)
		z_min_cur = zs[idx]
		p_min_cur = points[idx]
		if z_min_cur<z_min_tot:
			z_min_tot = z_min_cur
			p_min_tot = p_min_cur
			
		g = grad(points,zs)
		n = norm(g)
		if np.isnan(n):
			break

		print(i, p_min_cur, z_min_cur, n)
		next_p = p - scale*g
		inc = n
		
		#if n <= precision:
		if norm(next_p-p)<=precision:
			break
			
		p = next_p
		i+=1
			
	return p_min_tot,z_min_tot

def guess_int(p,inc,limits):
    inc = max(inc,1.)
    points = [p,[p[0],p[1]+inc],[p[0]+inc,p[1]]]
    for i in range(len(points)):
        for j,l in enumerate(limits):
            points[i][j] = min(max(int(points[i][j]),l[0]),l[1])
    return np.array(points)

def optimize_gradient_integer(func,x0,init_inc,scale,limits,precision,maxiter):
    p=x0
    inc = init_inc
    z_min_tot = float('inf')
    p_min_tot = x0
    i=0
    while(i<maxiter):
        v=[]
        zs = []
        points=guess_int(p,inc,limits)
        for l in points:
            z=func(l)
            zs.append(z)
        zs = np.array(zs)
        idx = np.argmin(zs)
        z_min_cur = zs[idx]
        p_min_cur = points[idx]
        if z_min_cur<z_min_tot:
            z_min_tot = z_min_cur
            p_min_tot = p_min_cur
            
        g = grad(points,zs)
        n = norm(g)
        if np.isnan(n):
            break

        print(i, p_min_cur, z_min_cur, n)
        next_p = p - scale*g
        inc = n
            
        if norm(next_p-p)<.5:
            break

        p = next_p
        i+=1
            
    return p_min_tot,z_min_tot

if __name__ == "__main__":

    wait(10)
    
    def opt_func(x,x0,sig,x_noize,y_noize):
        yn = np.random.uniform(-y_noize/2, y_noize/2,size=1)
        xn = np.random.uniform(-x_noize/2, x_noize/2,size=2)
        x0 += xn
        y = ((x[0]-x0[0])/sig[0])**2 + ((x[1]-x0[1])/sig[1])**2
        return yn+y

    func_int = lambda x: opt_func([int(p) for p in x],x0=[-20,15],sig=[20,7],x_noize=0,y_noize=0)
    func_dbl = lambda x: opt_func(x,x0=[-30,15],sig=[20,7],x_noize=0,y_noize=0)

    '''
    print(optimize_gradient(func_dbl,
                            x0=[0,0], init_inc=10, scale=30,
                            limits=[[-200,200],[-200,200]],
                            precision=1e-7, maxiter=1000))
    
    print(optimize_gradient_integer(func_int,
                                    x0=[0,0], init_inc=10, scale=30,
                                    limits=[[-200,200],[-200,200]],
                                    precision=1e-7, maxiter=1000))
    '''
