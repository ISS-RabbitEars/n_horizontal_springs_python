import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(ic, ti, p):
	m, k, xeq, xp = p
	
	x = []
	v = []
	for i in range(m.size):
		x.append(ic[2*i])
		v.append(ic[2*i+1])

	sub = {}
	for i in range(m.size):
		sub[M[i]] = m[i]
		sub[Xeq[i]] = xeq[i]
		sub[X[i]] = x[i]
		sub[K[i]] = k[i]
	sub[K[m.size]] = k[m.size]
	sub[Xeq[m.size]] = xeq[m.size]
	sub['Xp'] = xp

	diff_eq = []
	for i in range(m.size):
		diff_eq.append(v[i])
		diff_eq.append(A[i].subs(sub))

	print(ti)

	return diff_eq


N = 6 

Xp, t = sp.symbols('Xp t')
M = sp.symbols('M0:%i'%N)
K = sp.symbols('K0:%i'%(N+1))
Xeq = sp.symbols('Xeq0:%i'%(N+1))
X = dynamicsymbols('X0:%i'%N)

Xdot = [X[i].diff(t,1) for i in range(N)]

T = 0
for i in range(N):
	T += M[i] * Xdot[i]**2
T *= sp.Rational(1,2)

V = K[0] * (X[0] - Xeq[0])**2 + K[N] * (X[N-1] - Xp + Xeq[N])**2
for i in range(1,N):
	V += K[i] * (X[i] - X[i-1] - Xeq[i])**2
V *= sp.Rational(1,2)

L = T - V

dL = []
Xddot = []
for i in range(N):
	Xddot.append(X[i].diff(t,2))
	dLdX = L.diff(X[i],1)
	dLdXdot = L.diff(Xdot[i],1)
	ddtdLdXdot = dLdXdot.diff(t,1)
	dL.append(sp.simplify(ddtdLdXdot - dLdX))

sol = sp.solve(dL,Xddot)

A = []
for i in range(N):
	A.append(sp.simplify(sol[Xddot[i]]))

#------------------------------------------------

ma,mb = [1, 2]
ka,kb = [10, 50]
xeqa,xeqb = [3, 5]
xoa,xob = [0, 0] #not used if set to "stagger"
voa,vob = [0, 0]
rad = 0.25
tf = 60 
initialize_position = "stagger"
initialize_mass = "random"
initialize_equilibrium = "increment"
initialize_spring_constant = "random"
initialize_velocity = "increment"


if initialize_equilibrium == "increment":
	xeq = np.linspace(xeqa,xeqb,N+1) 
elif initialize_equilibrium == "random":
	xeq = (xeqb - xeqa) * np.random.rand(N+1) + xeqa

post2 = sum(xeq)

if initialize_position == "incrment":
	xo = np.linspace(xoa,xob,N)
elif initialize_position == "stagger":
	rng = np.random.default_rng(92738273)
	xo = np.zeros(N)
	xo[0] = rng.random() * (xeq[0]+xeq[1])
	for i in range(1,N-1):
		sum = 0
		for j in range(i+1):
			sum += xeq[j]
		dx = rng.random() * (sum - xo[i-1] + xeq[i+1])
		while dx > xeq[i]+xeq[i+1]:
			dx *= rng.random()
		xo[i] = xo[i-1] + dx
	dx = rng.random() * (post2 - xo[N-2])
	while dx > xeq[N-1]+xeq[N]:
		dx *= rng.random()
	xo[N-1] = xo[N-2] + dx

if initialize_mass == "increment":
	m = np.linspace(ma,mb,N)
elif initialize_mass == "random":
	m = (mb - ma) * np.random.rand(N) + ma

if initialize_spring_constant == "increment":
	k = np.linspace(ka,kb,N+1)
elif initialize_spring_constant == "random":
	k = (kb - ka) * np.random.rand(N+1) + ka

if initialize_velocity == "increment":
	vo = np.linspace(voa,vob,N) 

ic = []
for i in range(N):
	ic.append(xo[i])
	ic.append(vo[i])

p = m,k,xeq,post2

nfps = 30
nframes = tf * nfps
ta = np.linspace(0,tf,nframes)

xv = odeint(integrate, ic, ta, args=(p,))

ke = np.zeros(nframes)
pe = np.zeros(nframes)
for i in range(nframes):
	ke_sub = {}
	pe_sub = {}
	for j in range(N):
		ke_sub[M[j]] = m[j]
		ke_sub[Xdot[j]] = xv[i,2*j+1]
		pe_sub[X[j]] = xv[i,2*j]
		pe_sub[Xeq[j]] = xeq[j]
		pe_sub[K[j]] = k[j]
	pe_sub[K[N]] = k[N]
	pe_sub[Xeq[N]] = xeq[N]
	pe_sub['Xp'] = post2
	ke[i] = T.subs(ke_sub)
	pe[i] = V.subs(pe_sub)
E = ke + pe

#--------------------------------------------------

post1 = 0
yline = 0
mr = rad*m/max(m)
xmin = post1 - rad
xmax = post2 + rad
ymin = yline - 2 * max(mr)
ymax = yline + 2 * max(mr)
nl = np.zeros(N+1,dtype=int)
nl[0] = int(np.ceil((max(xv[:,0])-mr[0])/(2*mr[0])))
nl[N] = int(np.ceil((post2-min(xv[:,2*(N-1)])-mr[N-1])/(2*mr[N-1])))
for i in range(1,N):
	nl[i] = int(np.ceil((max(xv[:,2*i]-xv[:,2*(i-1)])-(mr[i]+mr[i-1]))/(mr[i]+mr[i-1])))
for i in range(N+1):
	nl[i] *= int(k[i]/min(k))
xl = np.zeros((N+1,max(nl),nframes))
yl = np.zeros((N+1,max(nl),nframes))
for i in range(nframes):
	l = np.zeros(N+1)
	l[0] = (xv[i,0] - post1 - mr[0])/nl[0]
	l[N] = (post2 - xv[i,2*(N-1)] - mr[N-1])/nl[N]
	xl[0][0][i] = xv[i,0] - mr[0] - 0.5 * l[0]
	for j in range(1,nl[0]):
		xl[0][j][i] = xl[0][j-1][i] - l[0]
	for j in range(1,N):
		l[j] = (xv[i,2*j] - xv[i,2*(j-1)] - (mr[j] + mr[j-1]))/nl[j]
	for j in range(1,N+1):
		xl[j][0][i] = xv[i,2*(j-1)] + mr[j-1] + 0.5 * l[j]
		for k in range(1,nl[j]):
			xl[j][k][i] = xl[j][k-1][i] + l[j]	
	for j in range(nl[0]):
		yl[0][j][i] = yline + ((-1)**j)*(np.sqrt(mr[0]**2 - (0.5*l[0])**2))
	for j in range(nl[N]):
		yl[N][j][i] = yline + ((-1)**j)*(np.sqrt(mr[N-1]**2 - (0.5*l[N])**2)) 
	for j in range(1,N):
		for k in range(nl[j]):
			yl[j][k][i] = yline + ((-1)**k)*(np.sqrt(((mr[j]+mr[j-1])/2)**2 - (0.5*l[j])**2))


fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	for i in range(N):
		circle=plt.Circle((xv[frame,2*i],yline),radius=mr[i],fc='xkcd:red')
		plt.gca().add_patch(circle)
	plt.plot([post1,post1],[ymin,ymax],'xkcd:cerulean',lw=4)
	plt.plot([post2,post2],[ymin,ymax],'xkcd:cerulean',lw=4)
	plt.plot([xv[frame,0]-mr[0],xl[0][0][frame]],[yline,yl[0][0][frame]],'xkcd:cerulean')
	plt.plot([xl[0][nl[0]-1][frame],post1],[yl[0][nl[0]-1][frame],yline],'xkcd:cerulean')
	plt.plot([xl[N][nl[N]-1][frame],post2],[yl[N][nl[N]-1][frame],yline],'xkcd:cerulean')
	for i in range(N):
		plt.plot([xv[frame,2*i]+mr[i],xl[i+1][0][frame]],[yline,yl[i+1][0][frame]],'xkcd:cerulean')
	for i in range(N+1):
		for j in range(1,nl[i]):
			plt.plot([xl[i][j-1][frame],xl[i][j][frame]],[yl[i][j-1][frame],yl[i][j][frame]],'xkcd:cerulean')
	for i in range(1,N):
		plt.plot([xl[i][nl[i]-1][frame],xv[frame,2*i]-mr[i]],[yl[i][nl[i]-1][frame],yline],'xkcd:cerulean')
	plt.title("%i Springs"%(N+1) + " and %i Masses" %N)
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['V','T','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('n_spring_chain.mp4', writer=writervideo)
plt.show()

