#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone cross-check solver for CUMCM2026 A Q3/Q4.

READ THE MODEL, not just the numerical tolerances:
  Original attachments are read from external XLSX files (stdlib ZIP/XML).
  Appendix 3 is used for Q3 FROM t=0; Appendix 4 for Q4 FROM t=0.
  Initial T=28 degC, C=2.55 kg/kg; C is DRY-BASIS moisture.
  h=25 W/(m2 K), hm=8e-7 m/s are inherited modelling assumptions.
  No explicit latent heat, radiation, sorption law, or room feedback model.
  All measured environmental values are retained; after their support a
  DECLARED 60-s linear transition reaches 50 degC / .05 kg/kg, then constant.
  Q4 R(t) is piecewise-linear attachment 2 (cm -> m), no smoothing/fitting.
  Radius extrapolation is forbidden unless --radius-tail hold is explicit.
  Q4 assumes homothetic MATERIAL shrinkage, NOT cropping a stationary body.

Both solvers use s=(r/R(t))**2, removing the cylindrical coordinate singularity:
  cap(C) T_t = 4/R**2 * d_s(s*k(C)*T_s) [+ 1/H**2*d_eta(k*T_eta)]
  C_t        = 4/R**2 * d_s(s*D(C,T)*C_s) [+ 1/H**2*d_eta(D*C_eta)]
  In 2D eta=z/H(t), 0<=eta<=1 is the symmetric HALF cylinder, H0=.125 m.
  Natural radial Robin load: 2*h/R*(Tair-Tsurface); moisture analogous.
  End Robin load: h/H*(Tair-Tend), or zero for the insulated control.
  rho*cp multiplies the material T derivative: NOT d(rho*cp*T)/dt.
  Geometrical compression is NOT added to dry-basis C.
  The 2D end condition and axial shrinkage are assumptions, not supplied data.

Implementation: independent weak-form nodal Galerkin discretizations;
positive diagonal quadrature mass; analytic fully coupled sparse Jacobian;
Numba-compiled local kernels, cached connectivity/quadrature/CSR scatter maps;
adaptive implicit time integration with exact stops at forcing slope changes.
Constant derivatives/geometry are precomputed; changing material coefficients
and Newton matrices are NOT wrongly frozen. Trial states are never clipped.

--workers is an aggregate CPU ceiling, not a promise to occupy every CPU.
Independent Q/mesh/tolerance/end-condition cases run in separate spawn processes.
No identical case is scheduled twice. BLAS threads=1; optional Numba threads per
case are bounded so jobs*threads<=workers<=24. Completed and partial checkpoints
are content-addressed and verified; changing core count does not invalidate them.
At restart BDF/Radau is reinitialized at the saved accepted state; restarting a
multistep method is NOT bitwise identical to uninterrupted integration.

Numerical tolerances are local error controls, NOT a proof of global accuracy.
Reports compare spatial refinements and a separate tighter-time case. A missing
comparison is NOT a pass. A model's error is not estimated by mesh convergence.
References: SciPy solve_ivp/Radau/BDF documentation; Trefethen, Spectral Methods
in MATLAB (Chebyshev nodes, differentiation and Clenshaw-Curtis quadrature).
"""
from __future__ import annotations
import argparse, csv, hashlib, heapq, json, math, multiprocessing as mp
import os, platform, posixpath, sys, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from dataclasses import dataclass, asdict

# Set BEFORE numerical libraries are imported, also in spawned workers.
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['NUMBA_NUM_THREADS']='24'
try:
    import numpy as np
    import scipy
    from scipy.integrate import Radau, BDF
    from scipy.sparse import csr_matrix
    from scipy.optimize import brentq
    from scipy.fft import dct
    import numba
    from numba import njit, prange
except ImportError as e:
    raise SystemExit('Dependencies: python3 -m pip install numpy scipy numba\n'+str(e))
VERSION='1.0.0'
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
DEFAULT_INPUT='/home/zyf/CUMCM/problem/attachment/附件1.xlsx'
DEFAULT_RADIUS='/home/zyf/CUMCM/problem/attachment/附件2.xlsx'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(tmp,path)


def save_npz(path, meta, **arrays):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    h=hashlib.sha256()
    for k,a in sorted(arrays.items()):
        a=np.ascontiguousarray(a);h.update(k.encode());h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        if a.nbytes:h.update(memoryview(a).cast('B'))
    meta={**meta,'arrays_sha256':h.hexdigest()}
    with tmp.open('wb') as f:
        np.savez_compressed(f,meta=np.array(json.dumps(meta,ensure_ascii=False,allow_nan=False)),**arrays)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def load_npz(path,key):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['meta'].item()));a={k:z[k] for k in z.files if k!='meta'}
    if meta['key']!=key:raise ValueError('Cache key mismatch')
    h=hashlib.sha256()
    for k,v in sorted(a.items()):
        if v.dtype.kind not in 'fiu' or not np.all(np.isfinite(v)):raise ValueError('Invalid checkpoint data')
        v=np.ascontiguousarray(v);h.update(k.encode());h.update(str(v.shape).encode())
        h.update(str(v.dtype).encode())
        if v.nbytes:h.update(memoryview(v).cast('B'))
    if h.hexdigest()!=meta['arrays_sha256']:raise ValueError('Cache content digest mismatch')
    return meta,a


def read_excel(path, headers):
    """First worksheet, value-only, no dependency on installed spreadsheet suites."""
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'External input does not exist: {path}')
    with ZipFile(path) as z:
        if z.testzip():raise ValueError(f'Corrupt XLSX: {path}')
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            shared=[''.join(n.text or '' for n in si.iter('{'+NS+'}t'))
                    for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        sh=ET.fromstring(z.read('xl/workbook.xml')).find('{'+NS+'}sheets/{'+NS+'}sheet')
        if sh is None:raise ValueError('No worksheet')
        rid=sh.attrib['{'+REL+'}id']
        rr=next(r for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels')) if r.attrib['Id']==rid)
        if rr.get('TargetMode')=='External':raise ValueError('External XLSX relationship')
        target=rr.attrib['Target'];member=posixpath.normpath(target.lstrip('/') if target.startswith('/') else 'xl/'+target)
        if not member.startswith('xl/'):raise ValueError('Unsafe workbook member')
        rows=[]
        for row in ET.fromstring(z.read(member)).findall('{'+NS+'}sheetData/{'+NS+'}row'):
            vals=[None]*len(headers)
            for cell in row:
                k=0
                for ch in cell.attrib['r']:
                    if not ch.isalpha():break
                    k=k*26+ord(ch.upper())-64
                k-=1
                if k>=len(vals):continue
                if cell.find('{'+NS+'}f') is not None:raise ValueError('Input formula cache is not accepted')
                v=cell.find('{'+NS+'}v');typ=cell.get('t','n')
                if typ=='inlineStr':v=''.join(n.text or '' for n in cell.iter('{'+NS+'}t'))
                elif v is None:v=None
                elif typ=='s':v=shared[int(v.text)]
                elif typ in ('b','e'):raise ValueError('Boolean/error input cell')
                else:v=v.text
                vals[k]=v
            if any(v is not None for v in vals):rows.append(vals)
    if not rows or [str(v).strip() for v in rows[0]]!=headers:raise ValueError(f'Expected headers {headers}: {path}')
    a=np.asarray(rows[1:],dtype=float)
    if len(a)<2 or not np.all(np.isfinite(a)) or a[0,0]!=0 or np.any(np.diff(a[:,0])<=0):
        raise ValueError('Input requires finite, complete rows and strictly increasing times from zero')
    return a


def cpu_cap(requested):
    if not 1<=requested<=24:raise ValueError('--workers must be in [1,24]')
    allowed=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))
    cap=min(requested,len(allowed));quota=None
    roots={Path('/sys/fs/cgroup'),Path('/sys/fs/cgroup/cpu'),Path('/sys/fs/cgroup/cpu,cpuacct')}
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _,ctl,sub=line.split(':',2)
            if '..' not in Path(sub).parts and (not ctl or 'cpu' in ctl.split(',')):
                for root in tuple(roots):
                    p=root/sub.lstrip('/')
                    for parent in (p,*p.parents):
                        if parent==root or root in parent.parents:roots.add(parent)
    except OSError:pass
    for root in roots:
        try:
            v=(root/'cpu.max').read_text().split()
            if v[0]!='max':q=float(v[0])/float(v[1]);quota=q if quota is None else min(quota,q)
        except (OSError,ValueError,IndexError):pass
        try:
            q=float((root/'cpu.cfs_quota_us').read_text());p=float((root/'cpu.cfs_period_us').read_text())
            if q>0:quota=q/p if quota is None else min(quota,q/p)
        except (OSError,ValueError):pass
    if quota is not None:cap=min(cap,max(1,math.floor(quota)))
    # Physical cores first, then siblings, without expanding caller affinity.
    first=[];siblings=[];seen=set()
    for c in allowed:
        r=Path(f'/sys/devices/system/cpu/cpu{c}/topology')
        try:ident=((r/'physical_package_id').read_text(),(r/'core_id').read_text())
        except OSError:ident=('cpu',c)
        (siblings if ident in seen else first).append(c);seen.add(ident)
    selected=(first+siblings)[:cap]
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,selected)
    return cap,{'requested_cap':requested,'effective_cap':cap,'quota':quota,'selected_cpus':selected}


def worker_init(threads):numba.set_num_threads(threads)


@njit(cache=True,nogil=True)
def material(T,C,q):
    n=T.size;val=np.empty((7,n))
    if q==3:r0,r1,p0,p1,k0,k1,d0,a=650.,128.,1450.,2736.,.21,.38,.0024,.45
    else:r0,r1,p0,p1,k0,k1,d0,a=760.,90.,1850.,2150.,.12,.20,.00042,.30
    for i in range(n):
        c=C[i];tk=T[i]+273.15
        if c<=0 or tk<=0 or not math.isfinite(c+tk):raise ValueError('Nonphysical nonlinear trial; no clipping')
        z=c/(1+c);dz=1/(1+c)**2
        rho=r0+r1*c;cp=p0+p1*z;d=d0*math.exp(-a/c-3850/tk)
        if d<=0 or not math.isfinite(d):raise ValueError('Invalid diffusivity')
        val[0,i]=rho*cp;val[1,i]=r1*cp+rho*p1*dz
        val[2,i]=k0+k1*z;val[3,i]=k1*dz
        val[4,i]=d;val[5,i]=d*a/(c*c);val[6,i]=d*3850/(tk*tk)
    return val


@njit(cache=True,nogil=True,parallel=True)
def local_rhs(y,conn,G0,G1,W0,W1,coef,a0,a1):
    ne,m=conn.shape;out=np.empty((ne,m,2))
    for e in prange(ne):
        for i in range(m):out[e,i,0]=0.;out[e,i,1]=0.
        for k in range(m):
            gt0=0.;gt1=0.;gc0=0.;gc1=0.
            for j in range(m):
                idx=conn[e,j];t=y[2*idx]-y[2*conn[e,0]];c=y[2*idx+1]-y[2*conn[e,0]+1]
                gt0+=G0[e,k,j]*t;gc0+=G0[e,k,j]*c
                gt1+=G1[e,k,j]*t;gc1+=G1[e,k,j]*c
            idx=conn[e,k];kval=coef[2,idx];d=coef[4,idx]
            wr=a0*W0[e,k];wz=a1*W1[e,k]
            for i in range(m):
                out[e,i,0]-=kval*(wr*G0[e,k,i]*gt0+wz*G1[e,k,i]*gt1)
                out[e,i,1]-=d*(wr*G0[e,k,i]*gc0+wz*G1[e,k,i]*gc1)
    return out


@njit(cache=True,nogil=True)
def scatter_rhs(local,conn,n):
    out=np.zeros((n,2))
    for e in range(conn.shape[0]):
        for i in range(conn.shape[1]):
            k=conn[e,i];out[k,0]+=local[e,i,0];out[k,1]+=local[e,i,1]
    return out


@njit(cache=True,nogil=True,parallel=True)
def local_jac(y,conn,G0,G1,W0,W1,coef,a0,a1):
    ne,m=conn.shape;out=np.zeros((ne,4,m,m))
    for e in prange(ne):
        for k in range(m):
            gt0=0.;gt1=0.;gc0=0.;gc1=0.
            for j in range(m):
                idx=conn[e,j];t=y[2*idx]-y[2*conn[e,0]];c=y[2*idx+1]-y[2*conn[e,0]+1]
                gt0+=G0[e,k,j]*t;gc0+=G0[e,k,j]*c
                gt1+=G1[e,k,j]*t;gc1+=G1[e,k,j]*c
            idx=conn[e,k];kv=coef[2,idx];kc=coef[3,idx];d=coef[4,idx];dc=coef[5,idx];dt=coef[6,idx]
            wr=a0*W0[e,k];wz=a1*W1[e,k]
            for i in range(m):
                cr=wr*G0[e,k,i];cz=wz*G1[e,k,i]
                qt=cr*gt0+cz*gt1;qc=cr*gc0+cz*gc1
                out[e,1,i,k]-=kc*qt
                out[e,2,i,k]-=dt*qc
                out[e,3,i,k]-=dc*qc
                for j in range(m):
                    v=cr*G0[e,k,j]+cz*G1[e,k,j]
                    out[e,0,i,j]-=kv*v;out[e,3,i,j]-=d*v
    return out


class GalerkinSystem:
    def setup(self):
        """Freeze structure once. Q1/Q2/SEM all have shared physical interface DOFs."""
        self.conn=np.ascontiguousarray(self.conn,dtype=np.int64)
        for name in ('G0','G1','W0','W1','mass','side','end'):
            setattr(self,name,np.ascontiguousarray(getattr(self,name),dtype=np.float64))
        self.n=len(self.mass);ne,m=self.conn.shape;n2=2*self.n
        if np.min(self.mass)<=0 or abs(self.mass.sum()-1)>1e-10:raise ValueError('Bad positive quadrature mass')
        keys=[]
        for fi,fj in ((0,0),(0,1),(1,0),(1,1)):
            keys.append((2*self.conn[:,:,None]+fi)*n2+(2*self.conn[:,None,:]+fj))
        kk=np.stack(keys,axis=1)
        uniq,self.slots=np.unique(kk.ravel(),return_inverse=True)
        self.rows=uniq//n2;self.indices=(uniq%n2).astype(np.int32)
        self.indptr=np.r_[0,np.cumsum(np.bincount(self.rows,minlength=n2))].astype(np.int32)
        diagkeys=np.arange(n2,dtype=np.int64)*(n2+1)
        self.diag_slots=np.searchsorted(uniq,diagkeys)
        if np.any(uniq[self.diag_slots]!=diagkeys):raise AssertionError('Missing Jacobian diagonal')
        self.slots=np.ascontiguousarray(self.slots)
        self.last_y=None;self.last_t=None

    def geometry(self,t):
        R,H,airT,airC=self.forcing(t)
        b=2*self.side/R+self.end/H*self.end_factor
        return 4/R**2,1/H**2,b,airT,airC

    def cache_state(self,t,y):
        if self.last_t!=t or self.last_y is None or not np.array_equal(y,self.last_y):
            c=material(y[::2],y[1::2],self.q)
            a0,a1,b,airT,airC=self.geometry(t)
            raw=scatter_rhs(local_rhs(y,self.conn,self.G0,self.G1,self.W0,self.W1,c,a0,a1),self.conn,self.n)
            raw[:,0]+=self.ht*b*(airT-y[::2]);raw[:,1]+=self.hm*b*(airC-y[1::2])
            den=np.column_stack((self.mass*c[0],self.mass))
            self.f=(raw/den).ravel();self.coef=c;self.den=den.ravel();self.b=b;self.a0=a0;self.a1=a1
            self.last_y=y.copy();self.last_t=t
        return self.f

    def rhs(self,t,y):return self.cache_state(t,y).copy()

    def jac(self,t,y):
        self.cache_state(t,y)
        local=local_jac(y,self.conn,self.G0,self.G1,self.W0,self.W1,self.coef,self.a0,self.a1)
        data=np.bincount(self.slots,weights=local.ravel(),minlength=len(self.rows))
        data[self.diag_slots[::2]]-=self.ht*self.b
        data[self.diag_slots[1::2]]-=self.hm*self.b
        data/=self.den[self.rows]
        # derivative of state-dependent heat capacity in the denominator
        # Each T,C pair is already in the frozen sparsity pattern.
        # This index is computed only on the first Jacobian call.
        if not hasattr(self,'tc_slots'):
            tc_keys=(2*np.arange(self.n,dtype=np.int64))*(2*self.n)+2*np.arange(self.n)+1
            uniq=self.rows*(2*self.n)+self.indices
            self.tc_slots=np.searchsorted(uniq,tc_keys)
        data[self.tc_slots]-=self.f[::2]*self.coef[1]/self.coef[0]
        return csr_matrix((data,self.indices,self.indptr),shape=(2*self.n,2*self.n)).tocsc()


def make_forcing(env,radius,q,o):
    # Data are already validated in the parent and serialized once per task.
    tail=np.vstack((env,[env[-1,0]+o['tail_ramp'],o['air_temperature'],o['air_moisture']]))
    def forcing(t):
        if q==4:
            if t>radius[-1,0]+1e-8 and o['radius_tail']=='error':raise ValueError('Radius coverage exceeded')
            R=float(np.interp(t,radius[:,0],radius[:,1]))
        else:R=.02
        H=.125*(R/.02 if o['axial_shrink']=='isotropic' and q==4 else 1.)
        return R,H,float(np.interp(t,tail[:,0],tail[:,1])),float(np.interp(t,tail[:,0],tail[:,2]))
    return forcing,tail


def knot_times(env,radius,q,limit,o):
    # Exact stops prevent a high-order integrator stepping across a forcing kink.
    times=list(env[:,0])+[env[-1,0]+o['tail_ramp'],limit]
    if q==4:
        # Only remove exactly collinear radius segments, NEVER alter a measured value.
        slope=np.diff(radius[:,1])/np.diff(radius[:,0])
        keep=np.r_[True,np.abs(np.diff(slope))>2e-19,True]
        times.extend(radius[keep,0])
    return np.unique([float(x) for x in times if 0<x<=limit])


def pack_records(records,nprobe):
    return {'times':np.asarray(records['times']),
            'probes':np.asarray(records['probes']).reshape((-1,nprobe,2)),
            'stats':np.asarray(records['stats']).reshape((-1,7)),
            'snap_times':np.asarray(records['snap_times']),
            'snapshots':np.asarray(records['snapshots']),
            'physical':np.asarray(records['physical']),
            'physical_valid':np.asarray(records['physical_valid'],dtype=np.int8)}


def run_case(spec,env,radius,o,key,cache_dir):
    label=f'Q{spec["q"]} {spec["tag"]}'
    folder=Path(cache_dir);finished=folder/(key+'.npz');partial=folder/(key+'.partial.npz')
    if finished.exists() and not o['no_cache']:
        meta,_=load_npz(finished,key);print(f'[cache] {label}: {meta["status"]}',flush=True)
        return {'key':key,'path':str(finished),'meta':meta}
    model=make_model(spec,o)
    model.forcing,tail=make_forcing(env,radius,spec['q'],o)
    model.q=spec['q'];model.ht=o['heat_transfer'];model.hm=o['mass_transfer']
    model.end_factor=1. if spec.get('end','insulated')=='exposed' else 0.
    y=np.empty(2*model.n);y[::2]=28.;y[1::2]=2.55
    t=0.;wall=time.perf_counter();saved_wall=0.;last_save=0.;event=None
    records={k:[] for k in ('times','probes','stats','snap_times','snapshots','physical','physical_valid')}
    calls={'accepted_steps':0,'nfev':0,'njev':0,'nlu':0}
    minC=2.55;maxC=2.55;minT=28.;maxT=28.
    nprobe=model.probe_matrix.shape[0]
    if partial.exists() and not o['no_cache']:
        meta,a=load_npz(partial,key);t=meta['t'];y=a['y'];saved_wall=meta['wall_s'];calls=meta['calls']
        for k in records:records[k]=list(a[k])
        minC,maxC,minT,maxT=meta['ranges'];last_save=t
        print(f'[resume] {label}: {t/3600:.6f} h (integrator history restarted)',flush=True)
    limit=o['max_hours']*3600
    if spec['q']==4 and o['radius_tail']=='error':limit=min(limit,float(radius[-1,0]))
    breaks=knot_times(env,radius,spec['q'],limit,o)
    next_out=60*(math.floor(t/60+1e-10)+1)
    next_log=o['log_seconds']*(math.floor(t/o['log_seconds'])+1)
    next_snap=21600*(math.floor(t/21600)+1)
    last_h=None
    threshold=o['threshold']
    def dry_measure(yy):
        # Near a candidate event, inspect element INTERIORS as well as nodes.
        if np.max(yy[1::2])>threshold+.005:return float(np.max(yy[1::2]))
        return model.maximum(yy[1::2],o['max_enclosure_tol'])[1]
    def append_output(tt,yy):
        R,H,airT,airC=model.forcing(tt)
        pr=model.probe_matrix@yy.reshape(-1,2)
        records['times'].append(float(tt));records['probes'].append(pr)
        nrout=21 if spec['q']==3 else 20
        targets=np.arange(nrout)/10/(100*R)
        if spec['q']==4:targets=np.r_[targets,1.]
        valid=targets<=1.+1e-12
        pts=np.zeros((len(targets),model.coordinates.shape[1]));pts[:,0]=np.minimum(targets,1.)
        ph=model.interpolation(pts)@yy.reshape(-1,2)
        ph[~valid]=0.
        records['physical'].append(ph);records['physical_valid'].append(valid.astype(np.int8))
        lower,upper=model.maximum(yy[1::2],o['max_enclosure_tol']) if np.max(yy[1::2])<threshold+.005 else (float(np.max(yy[1::2])),float(np.max(yy[1::2])))
        records['stats'].append([tt,R,H,float(np.min(yy[1::2])),lower,upper,float(model.mass@yy[1::2])])
    for end in breaks:
        if end<=t+1e-9:continue
        first=min(last_h,end-t) if last_h is not None else None
        solver_cls=Radau if spec['method']=='Radau' else BDF
        atol=np.tile([spec['atol_T'],spec['atol_C']],model.n)
        solver=solver_cls(model.rhs,t,y,end,rtol=spec['rtol'],atol=atol,
                          jac=model.jac,max_step=spec['max_step'],first_step=first)
        while solver.status=='running':
            t0=solver.t;y0=solver.y.copy();g0=dry_measure(y0)-threshold
            msg=solver.step()
            if solver.status=='failed':raise RuntimeError(f'{label} failed at {solver.t}: {msg}')
            t1=solver.t;y1=solver.y;calls['accepted_steps']+=1
            if not np.all(np.isfinite(y1)) or np.min(y1[1::2])<=0 or np.min(y1[::2])<=-273.15:
                raise RuntimeError(f'{label}: invalid accepted state')
            minC=min(minC,float(np.min(y1[1::2])));maxC=max(maxC,float(np.max(y1[1::2])))
            minT=min(minT,float(np.min(y1[::2])));maxT=max(maxT,float(np.max(y1[::2])))
            if maxC>max(2.55,np.max(tail[:,2]))+o['range_tolerance'] or minC<min(2.55,np.min(tail[:,2]))-o['range_tolerance'] or minT<min(28.,np.min(tail[:,1]))-o['range_tolerance'] or maxT>max(28.,np.max(tail[:,1]))+o['range_tolerance']:
                raise RuntimeError(f'{label}: significant range violation at t={t1:.8g}: C=[{minC:.12g},{maxC:.12g}], T=[{minT:.12g},{maxT:.12g}]; refine space/time, do not clip')
            dense=solver.dense_output();g1=dry_measure(y1)-threshold
            stop=t1
            if g1<0 and g0>=0:
                lo=t0;hi=t1
                # A strict-below right bracket, including polynomial interior check.
                while hi-lo>o['event_tol']:
                    mid=.5*(lo+hi)
                    if dry_measure(dense(mid))<threshold:hi=mid
                    else:lo=mid
                stop=hi;ev_y=dense(hi);low,up=model.maximum(ev_y[1::2],o['max_enclosure_tol'])
                event={'lower_s':lo,'upper_s':hi,'width_s':hi-lo,'time_h':hi/3600,
                       'C_max_lower':low,'C_max_upper':up,'method':model.maximum_method,
                       'meaning':'Numerical-interpolant event bracket, NOT a PDE/model error bound'}
            while next_out<=stop+1e-8:
                append_output(next_out,dense(min(next_out,t1)));next_out+=60
            while next_snap<=stop+1e-8:
                records['snap_times'].append(float(next_snap));records['snapshots'].append(dense(next_snap).reshape(-1,2));next_snap+=21600
            if event is not None:t=stop;y=ev_y;break
            t=t1;y=y1.copy();last_h=min(solver.step_size or spec['max_step'],spec['max_step'])
            if t>=next_log:
                print(f'[{label}] t={t/3600:.4f} h; maxC={np.max(y[1::2]):.9f}; DOF={2*model.n}; steps={calls["accepted_steps"]}; wall={time.perf_counter()-wall:.1f}s',flush=True)
                next_log=o['log_seconds']*(math.floor(t/o['log_seconds'])+1)
            if not o['no_cache'] and o['checkpoint_seconds']>0 and t-last_save>=o['checkpoint_seconds']:
                a=pack_records(records,nprobe)
                if a['snapshots'].size==0:a['snapshots']=np.empty((0,model.n,2))
                running_calls={**calls}
                for k in ('nfev','njev','nlu'):running_calls[k]+=getattr(solver,k)
                save_npz(partial,{'key':key,'t':t,'wall_s':saved_wall+time.perf_counter()-wall,
                                 'calls':running_calls,'ranges':[minC,maxC,minT,maxT]},y=y,**a)
                last_save=t
        for k in ('nfev','njev','nlu'):calls[k]+=getattr(solver,k)
        if event is not None:break
        t=float(end);y=solver.y.copy()
    append_output(t,y) if not records['times'] or abs(records['times'][-1]-t)>1e-8 else None
    records['snap_times'].append(t);records['snapshots'].append(y.reshape(-1,2).copy())
    a=pack_records(records,nprobe)
    meta={'key':key,'status':'DRIED' if event else 'NOT_DRIED','spec':spec,'t':t,'event':event,
          'wall_s':saved_wall+time.perf_counter()-wall,'calls':calls,'nodes_per_field':model.n,
          'ranges':[minC,maxC,minT,maxT], 'maximum_nodal_range_overshoot':max(0.,maxC-max(2.55,float(np.max(tail[:,2]))),min(2.55,float(np.min(tail[:,2])))-minC,min(28.,float(np.min(tail[:,1])))-minT,maxT-max(28.,float(np.max(tail[:,1])))), 'mesh':model.mesh_info,
          'method_detail':model.description, 'model_validation':False}
    save_npz(finished,meta,y=y,coordinates=model.coordinates,probe_coordinates=model.probe_coordinates,**a)
    if partial.exists():partial.unlink()
    print(f'[done] {label}: {meta["status"]}; t={t/3600:.9f} h; wall={meta["wall_s"]:.1f}s',flush=True)
    return {'key':key,'path':str(finished),'meta':meta}


def col(j):
    s='';j+=1
    while j:j,k=divmod(j-1,26);s=chr(65+k)+s
    return s


def write_xlsx(path, times, data, headers):
    """Numeric result workbook; no input bytes or synthetic measurements embedded."""
    head=['时间/s']+list(headers)
    parts=['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',f'<worksheet xmlns="{NS}">',
           f'<dimension ref="A1:{col(len(head)-1)}{len(times)+1}"/>',
           '<sheetViews><sheetView workbookViewId="0"><pane xSplit="1" ySplit="1" topLeftCell="B2" activePane="bottomRight" state="frozen"/></sheetView></sheetViews>',
           '<cols><col min="1" max="1" width="22" customWidth="1"/><col min="2" max="40" width="13" customWidth="1"/></cols><sheetData><row r="1">']
    for j,v in enumerate(head):
        if isinstance(v,str):parts.append(f'<c r="{col(j)}1" s="1" t="inlineStr"><is><t>{escape(v)}</t></is></c>')
        else:parts.append(f'<c r="{col(j)}1" s="1"><v>{float(v):.12g}</v></c>')
    parts.append('</row>')
    for i,(t,row) in enumerate(zip(times,data),2):
        parts.append(f'<row r="{i}"><c r="A{i}" s="2"><v>{t:.12g}</v></c>')
        for j,v in enumerate(row,1):
            if np.isfinite(v):parts.append(f'<c r="{col(j)}{i}" s="2"><v>{v:.4f}</v></c>')
        parts.append('</row>')
    parts.append('</sheetData></worksheet>')
    styles=f'''<styleSheet xmlns="{NS}"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000"/></numFmts><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF24476A"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs></styleSheet>'''
    with ZipFile(path,'w',compression=ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        z.writestr('xl/styles.xml',styles);z.writestr('xl/worksheets/sheet1.xml',''.join(parts))
    with ZipFile(path) as z:
        if z.testzip():raise RuntimeError('Output XLSX validation failed')
        rows=ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('{'+NS+'}sheetData/{'+NS+'}row')
        if len(rows)!=len(times)+1:raise AssertionError('Output row count mismatch')


def common_args(kind):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--questions',default='3,4');p.add_argument('--input',default=DEFAULT_INPUT)
    p.add_argument('--radius-input',default=DEFAULT_RADIUS);p.add_argument('--outdir',default=f'results_{kind}')
    p.add_argument('--workers',type=int,default=24);p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--threads-per-case',type=int,default=1)
    p.add_argument('--max-hours',type=float,default=72.);p.add_argument('--radius-tail',choices=['error','hold'],default='error')
    p.add_argument('--air-temperature',type=float,default=50.);p.add_argument('--air-moisture',type=float,default=.05)
    p.add_argument('--tail-ramp',type=float,default=60.);p.add_argument('--axial-shrink',choices=['fixed','isotropic'],default='fixed')
    p.add_argument('--heat-transfer',type=float,default=25.);p.add_argument('--mass-transfer',type=float,default=8e-7)
    p.add_argument('--threshold',type=float,default=.15);p.add_argument('--rtol',type=float,default=2e-9)
    p.add_argument('--atol-T',type=float,default=1e-10);p.add_argument('--atol-C',type=float,default=1e-11)
    p.add_argument('--max-step',type=float,default=30.)
    p.add_argument('--event-tol',type=float,default=1e-4);p.add_argument('--max-enclosure-tol',type=float,default=2e-11)
    p.add_argument('--range-tolerance',type=float,default=2e-3,help='Hard abort threshold for high-order transient overshoot; recorded, never clipped')
    p.add_argument('--checkpoint-seconds',type=float,default=1800.)
    p.add_argument('--log-seconds',type=float,default=1800.)
    p.add_argument('--field-target',type=float,default=2e-6);p.add_argument('--time-target',type=float,default=.05)
    p.add_argument('--no-time-check',action='store_true');p.add_argument('--no-cache',action='store_true')
    p.add_argument('--self-test-only',action='store_true')
    return p


def comparison(a,b):
    ma,aa=load_npz(a['path'],a['key']);mb,bb=load_npz(b['path'],b['key'])
    # Compare common regular 60-s outputs only; independent event rows differ.
    ta=aa['times'];tb=bb['times'];ia=np.flatnonzero(abs(ta/60-np.rint(ta/60))<1e-9);ib=np.flatnonzero(abs(tb/60-np.rint(tb/60))<1e-9)
    _,ix,iy=np.intersect1d(ta[ia],tb[ib],return_indices=True)
    if len(ix)==0:return {'status':'NO_COMMON_OUTPUT'}
    diff=np.max(np.abs(aa['probes'][ia[ix]]-bb['probes'][ib[iy]]),axis=(0,1))
    ev=abs(ma['event']['upper_s']-mb['event']['upper_s']) if ma['event'] and mb['event'] else None
    return {'max_T_difference_C':float(diff[0]),'max_C_difference_kg_kg':float(diff[1]),
            'drying_time_difference_s':ev,'common_times':len(ix),'status':'COMPARED',
            'meaning':'Observed differences only, not rigorous residual error estimates'}


def export_question(q, results, specs, o, folder, env, radius):
    relevant=[r for r in results if r['meta']['spec']['q']==q]
    mains=sorted([r for r in relevant if r['meta']['spec']['role']=='space'],key=lambda r:r['meta']['spec']['resolution'])
    high=mains[-1];meta,a=load_npz(high['path'],high['key'])
    report={'q':q,'solver':KIND,'main':meta,'spatial_comparisons':[],
            'temporal_comparison':None,'end_effect_comparison':None,'verification':'NOT_CHECKED','model_validation':False}
    for ra,rb in zip(mains[:-1],mains[1:]):
        report['spatial_comparisons'].append({'coarse':ra['meta']['spec']['tag'],'fine':rb['meta']['spec']['tag'],**comparison(ra,rb)})
    tight=next((r for r in relevant if r['meta']['spec']['role']=='time'),None)
    if tight:report['temporal_comparison']=comparison(high,tight)
    control=next((r for r in relevant if r['meta']['spec']['role']=='end_control'),None)
    if control:report['end_effect_comparison']=comparison(high,control)
    last=report['spatial_comparisons'][-1] if report['spatial_comparisons'] else None
    temporal=report['temporal_comparison']
    if last and temporal and last['status']=='COMPARED' and temporal['status']=='COMPARED':
        # An observed envelope; no unjustified Richardson order for p-refinement.
        fieldT=2*(last['max_T_difference_C']+temporal['max_T_difference_C'])
        fieldC=2*(last['max_C_difference_kg_kg']+temporal['max_C_difference_kg_kg'])
        de=None
        if last['drying_time_difference_s'] is not None and temporal['drying_time_difference_s'] is not None:
            de=2*(last['drying_time_difference_s']+temporal['drying_time_difference_s'])+2*o['event_tol']
        report['observed_envelope']={'T':fieldT,'C':fieldC,'time_s':de,'is_error_bound':False}
        report['verification']='OBSERVED_DIFFERENCE_PASS' if de is not None and max(fieldT,fieldC)<=o['field_target'] and de<=o['time_target'] else 'NEEDS_REFINEMENT_OR_REVIEW'
    if not meta['event']:report['verification']='NOT_DRIED'
    out=Path(folder)/f'Q{q}';out.mkdir(exist_ok=True)
    atomic_json(out/'comparison_report.json',report)
    atomic_json(out/'drying_time.json',{'event':meta['event'],'verification':report['verification'],'model_validation':False})
    # Preserve full probes and snapshots for independent checking; no rerun for exports.
    save_npz(out/'fields.npz',{'key':high['key']},**a)
    coordinates=a['probe_coordinates'];times=a['times'];probes=a['probes']
    # Actual basis-function evaluation at physical radii was saved during solving.
    headers=list(np.arange(21 if q==3 else 20)/10)+(['药材表面'] if q==4 else [])
    physical=a['physical'].copy()
    physical[np.asarray(a['physical_valid'])==0]=np.nan
    for f,name in enumerate(('temperature','moisture')):
        with (out/f'{name}_midplane.csv').open('w',encoding='utf-8-sig',newline='') as z:
            w=csv.writer(z);w.writerow(['time_s',*headers]);w.writerows(np.column_stack((times,physical[:,:,f])))
    stem=f'result{q}_{KIND}'+('_midplane' if KIND!='spectral' else '')+('' if meta['event'] else '_NOT_DRIED')
    write_xlsx(out/(stem+'.xlsx'),times,physical[:,:,1],headers)
    with (out/'radius_and_maximum.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','R_m','half_length_m','C_min_node','Cmax_lower','Cmax_upper','volume_reference_mean']);w.writerows(a['stats'])
    text=[f'# Q{q}: {KIND}',f'验证状态：{report["verification"]}',
          '本文件是指定数学模型的数值解，不是实测结果；误差判据不是严格上界。',
          f'主配置：{json.dumps(meta["spec"],ensure_ascii=False)}',f'网格：{json.dumps(meta["mesh"],ensure_ascii=False)}',
          '物性按题面附录；外界附件完整保留，之后60秒过渡到50°C/0.05 kg/kg。',
          '轴向长度默认不收缩；二维 exposed 表示端面沿用侧面换热/传质系数，insulated 为绝热不透湿控制。',
          '忽略潜热、辐射和端面以外的外界反馈；干基含水率不加体积浓度压缩项。',
          'XLSX/中截面CSV在当时实际厘米距离处直接计算谱多项式/Q2形函数；不是对稀疏观测点做二次线性插值。',
          '高精度对照使用 fields.npz 中共同坐标 probes；snapshots 为原始自由度场。',
          'fields.npz 包含 6h 全场剖面、终态和每60秒的共同观测点；没有将高密度插值冒充细网格求解。',
          '端面边界改变的是物理模型，端面对照差异不能叫做原一维算法的离散误差。',
          f'事件：{json.dumps(meta["event"],ensure_ascii=False)}',
          '初始均匀含水率与非零表面失水通量不相容；高阶离散可能出现短时小幅超调，程序记录并不裁剪，过大则报错。range_tolerance是安全阈值，不是精度目标。',
          '所有对照见 comparison_report.json。通过仅指共同输出时刻及事件时刻的差异判据，不表示每个瞬间的严格误差。缺少对照、未烘干均不标为通过。']
    selected=sorted(set(np.flatnonzero(abs(times/21600-np.rint(times/21600))<1e-9).tolist()+[len(times)-1]))
    columns=[0,5,10,15,20] if q==3 else [0,5,10,15,len(headers)-1]
    table=[f'# Q{q}: 每6小时及结束时刻含水率',
           '数值模型结果，不是实测；二维表为中截面 eta=0，终止判据仍检查整个二维域。',
           '| 时间/h | 半径/cm | '+' | '.join(str(headers[j]) for j in columns)+' |',
           '|---:|---:|'+ '|'.join(['---:']*len(columns))+'|']
    for i in selected:
        vals=[f'{times[i]/3600:.4f}',f'{a["stats"][i,1]*100:.4f}']
        vals += [f'{physical[i,j,1]:.4f}' if np.isfinite(physical[i,j,1]) else '—' for j in columns]
        table.append('| '+' | '.join(vals)+' |')
    (out/'summary_tables.md').write_text('\n\n'.join(table[:2])+'\n\n'+'\n'.join(table[2:])+'\n',encoding='utf8')
    (out/'README.md').write_text('\n\n'.join(text)+'\n',encoding='utf8')
    return report


def self_tests():
    o={'heat_transfer':25.,'mass_transfer':8e-7,'max_enclosure_tol':2e-11}
    spec=test_spec();model=make_model(spec,o);model.q=spec['q'];model.ht=25.;model.hm=8e-7
    model.end_factor=1.;model.forcing=lambda t:(.02,.125,37.,.03)
    y=np.empty(2*model.n);coord=model.coordinates
    y[::2]=32.+2*coord[:,0]**2+(coord[:,1]**2 if coord.shape[1]>1 else 0.)
    y[1::2]=1.8-.3*coord[:,0]**2-(.1*coord[:,1] if coord.shape[1]>1 else 0.)
    rng=np.random.default_rng(17);v=rng.normal(size=y.size);v[::2]*=2
    eps=1e-5
    fd=(model.rhs(3.,y+eps*v)-model.rhs(3.,y-eps*v))/(2*eps)
    jv=model.jac(3.,y)@v;err=np.max(abs(fd-jv))/max(1e-20,np.max(abs(fd)))
    if err>2e-6:raise AssertionError(f'Analytic Jacobian directional test: {err}')
    eq=y.copy();eq[::2]=37.;eq[1::2]=.03
    eqerr=float(np.max(abs(model.rhs(0.,eq))))
    if eqerr>1e-10:raise AssertionError(f'Uniform equilibrium RHS: {eqerr}')
    low,up=model.maximum(np.ones(model.n)*.7,2e-11)
    if abs(low-.7)>1e-9 or abs(up-.7)>1e-9:raise AssertionError('Polynomial maximum test')
    for q in (3,4):
        c=material(np.array([50.]),np.array([.15]),q)
        expect=(.0024 if q==3 else .00042)*np.exp(-(.45 if q==3 else .30)/.15-3850/323.15)
        if abs(c[4,0]/expect-1)>1e-13:raise AssertionError('Appendix/Kelvin')
    return {'analytic_J_directional_relative_error':float(err),'uniform_equilibrium_rhs':eqerr,
            'positive_mass_sum':float(model.mass.sum()),'polynomial_maximum_constant_test':True,'Appendix_Kelvin':True}


def main():
    args=parse_args();o=vars(args)
    cap,cpu=cpu_cap(args.workers)
    if args.threads_per_case<1 or args.jobs<1:raise ValueError('Positive jobs/threads required')
    threads=min(args.threads_per_case,cap);jobs=min(args.jobs,max(1,cap//threads))
    numba.set_num_threads(threads)
    tests=self_tests()
    if args.self_test_only:print(json.dumps(tests,indent=2));return 0
    for k in ('max_hours','tail_ramp','rtol','atol_T','atol_C','max_step','event_tol','max_enclosure_tol','log_seconds','field_target','time_target'):
        if not math.isfinite(o[k]) or o[k]<=0:raise ValueError(f'Invalid {k}')
    if args.checkpoint_seconds<0:raise ValueError('Negative checkpoint period')
    if not 0<args.air_moisture<args.threshold<2.55:raise ValueError('Require 0<air-moisture<threshold<2.55')
    if args.mass_transfer<=0 or args.heat_transfer<=0:raise ValueError('Positive transfer coefficients required')
    qs=sorted(set(map(int,args.questions.split(','))))
    if not qs or any(q not in (3,4) for q in qs):raise ValueError('questions must be 3,4 or 3 or 4')
    env=read_excel(args.input,['时间','温度','水分浓度'])
    if np.min(env[:,2])<=0 or np.min(env[:,1])<=-273.15:raise ValueError('Invalid forcing')
    radius=read_excel(args.radius_input,['时间','半径']) if 4 in qs else np.array([[0.,2.],[args.max_hours*3600,2.]])
    radius=radius.copy();radius[:,1]/=100
    if abs(radius[0,1]-.02)>1e-12 or np.min(radius[:,1])<=0 or np.any(np.diff(radius[:,1])>1e-12):raise ValueError('Invalid radius measurements')
    root=Path(args.outdir);root.mkdir(parents=True,exist_ok=True);cache=root/'cache';cache.mkdir(exist_ok=True)
    # OS file lock is released on crash; no stale lock deletion guesses.
    lock=(root/'.running.lock').open('a+')
    try:
        import fcntl;fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except ImportError:pass
    except BlockingIOError:raise RuntimeError('Another instance is writing this output directory')
    rawspecs=build_specs(qs,o);specs=[];seen=set()
    for s in rawspecs:
        ident=json.dumps({k:v for k,v in s.items() if k not in ('role','tag')},sort_keys=True)
        if ident not in seen:specs.append(s);seen.add(ident)
    jobs=min(jobs,len(specs))
    provenance={'source_sha256':digest(__file__),'input_sha256':digest(args.input),
                'radius_sha256':digest(args.radius_input) if 4 in qs else None,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'numba':numba.__version__}}
    numeric={k:v for k,v in o.items() if k not in ('workers','jobs','threads_per_case','outdir','no_cache','checkpoint_seconds','log_seconds','input','radius_input','self_test_only','fem_meshes','degrees','questions','no_time_check','field_target','time_target')}
    tasks=[]
    for s in specs:
        key=hashlib.sha256(json.dumps({'spec':s,'numeric':numeric,'provenance':provenance},sort_keys=True).encode()).hexdigest()
        tasks.append((s,key))
    atomic_json(root/'run_metadata.json',{'cpu':cpu,'jobs':jobs,'threads_per_case':threads,'provenance':provenance,'self_tests':tests,'specs':specs,'options':o})
    print(f'{KIND}: {len(tasks)} unique coupled cases, {jobs} processes x {threads} Numba threads (cap={cap}); no nested BLAS threads.',flush=True)
    results=[];started=time.perf_counter();pool=ProcessPoolExecutor(max_workers=jobs,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(threads,))
    try:
        pending={pool.submit(run_case,s,env,radius,o,key,str(cache)):s for s,key in sorted(tasks,key=lambda v:v[0]['resolution'],reverse=True)}
        for f in as_completed(pending):results.append(f.result())
    except BaseException:
        for f in pending:f.cancel()
        # Explicitly terminate active processes; accepted checkpoints remain valid.
        for proc in getattr(pool,'_processes',{}).values():proc.terminate()
        pool.shutdown(wait=True,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    if digest(args.input)!=provenance['input_sha256'] or (4 in qs and digest(args.radius_input)!=provenance['radius_sha256']):raise RuntimeError('Input changed while computing')
    reports=[export_question(q,results,specs,o,root,env,radius) for q in qs]
    atomic_json(root/'summary.json',{'elapsed_s':time.perf_counter()-started,'questions':reports})
    for r in reports:print(f'Q{r["q"]}: {r["verification"]}; see {root}/Q{r["q"]}/comparison_report.json',flush=True)
    return 0 if all(r['verification']=='OBSERVED_DIFFERENCE_PASS' for r in reports) else 2

KIND='spectral'


def clenshaw_curtis(p):
    if p<2:raise ValueError('Polynomial degree >=2 required')
    theta=np.pi*np.arange(p+1)/p
    x=-np.cos(theta);w=np.zeros(p+1);v=np.ones(p-1)
    if p%2==0:
        w[0]=w[-1]=1/(p*p-1)
        for k in range(1,p//2):v-=2*np.cos(2*k*theta[1:-1])/(4*k*k-1)
        v-=np.cos(p*theta[1:-1])/(p*p-1)
    else:
        w[0]=w[-1]=1/p**2
        for k in range(1,(p-1)//2+1):v-=2*np.cos(2*k*theta[1:-1])/(4*k*k-1)
    w[1:-1]=2*v/p
    bary=(-1.)**np.arange(p+1);bary[[0,-1]]*=.5
    dx=x[:,None]-x[None,:]
    np.fill_diagonal(dx,1.)
    D=(bary[None,:]/bary[:,None])/dx
    np.fill_diagonal(D,0.);np.fill_diagonal(D,-D.sum(axis=1))
    return x,w,D,bary


class Spectral(GalerkinSystem):
    description='Multi-domain Chebyshev-Lobatto nodal Galerkin spectral elements, Clenshaw-Curtis quadrature; s=(r/R)^2; Radau IIA(5)'
    maximum_method='All-element Chebyshev derivative-root extrema plus nodal checks; float64, not a certified interval bound'
    def __init__(self,p,breaks):
        self.p=p;self.bounds=np.asarray(breaks)**2;ne=len(breaks)-1;m=p+1
        x,w,D,bary=clenshaw_curtis(p);self.xlocal=x;self.bary=bary
        n=ne*p+1;self.conn=np.arange(n-1).reshape(ne,p)
        self.conn=np.column_stack((self.conn,np.arange(1,ne+1)*p))
        s=np.empty(n);self.mass=np.zeros(n)
        self.G0=np.empty((ne,m,m));self.G1=np.zeros((ne,m,m))
        self.W0=np.empty((ne,m));self.W1=np.zeros((ne,m))
        for e in range(ne):
            a,b=self.bounds[e:e+2];width=b-a;s_e=a+.5*(x+1)*width
            s[self.conn[e]]=s_e;we=w*.5*width
            self.mass[self.conn[e]]+=we
            self.G0[e]=D*2/width;self.W0[e]=we*s_e
        self.side=np.zeros(n);self.side[-1]=1.;self.end=np.zeros(n)
        self.coordinates=np.sqrt(np.maximum(s,0.))[:,None]
        self.probe_coordinates=np.linspace(0,1,801)[:,None]
        self.setup();self.probe_matrix=self.interpolation(self.probe_coordinates)
        gaps=np.diff(self.coordinates[:,0])*.02
        self.mesh_info={'degree_per_element':p,'radial_subdomains_xi':list(breaks),'nodes_per_field':n,
                        'initial_min_radial_gap_um':float(np.min(gaps)*1e6),'initial_max_radial_gap_um':float(np.max(gaps)*1e6),
                        'note':'Spectral polynomial order and DOF, not comparable to second-order FVM node count'}

    def interpolation(self,points):
        s=np.asarray(points)[:,0]**2
        e=np.clip(np.searchsorted(self.bounds,s,side='right')-1,0,len(self.bounds)-2)
        local=2*(s-self.bounds[e])/(self.bounds[e+1]-self.bounds[e])-1
        rows=[];cols=[];weights=[]
        for i in range(len(s)):
            d=local[i]-self.xlocal;k=np.argmin(abs(d))
            if abs(d[k])<1e-12:
                rows.append(i);cols.append(self.conn[e[i],k]);weights.append(1.)
            else:
                ww=self.bary/d;ww/=np.sum(ww)
                rows.extend([i]*len(ww));cols.extend(self.conn[e[i]]);weights.extend(ww)
        return csr_matrix((weights,(rows,cols)),shape=(len(s),self.n))

    def maximum(self,c,tol):
        result=float(np.max(c));padding=0.
        for ids in self.conn:
            vals=c[ids];co=dct(vals[::-1],type=1)/self.p;co[[0,-1]]*=.5
            # coefficients below float64 transform noise do not define reliable extrema
            dc=np.polynomial.chebyshev.chebder(co)
            dc=np.polynomial.chebyshev.chebtrim(dc,tol=5e-14*max(1.,float(np.max(abs(vals)))))
            roots=np.polynomial.chebyshev.chebroots(dc) if len(dc)>1 else np.empty(0)
            good=roots[(np.abs(np.imag(roots))<1e-8)&(np.real(roots)>=-1-1e-12)&(np.real(roots)<=1+1e-12)]
            if len(good):result=max(result,float(np.max(np.polynomial.chebyshev.chebval(np.clip(np.real(good),-1,1),co))))
            padding=max(padding,32*np.finfo(float).eps*(1+float(np.sum(abs(co)))))
        return result,result+padding


def make_model(spec,o):return Spectral(spec['degree'],spec['breaks'])


def parse_args():
    p=common_args(KIND)
    p.set_defaults(jobs=8,threads_per_case=1)
    p.add_argument('--degrees',default='24,40,64',help='Distinct polynomial degrees per subdomain, increasing')
    p.add_argument('--breaks',default='0,0.2,0.4,0.6,0.8,0.9,0.96,0.985,0.997,1',help='Subdomain boundaries in xi=r/R')
    return p.parse_args()


def build_specs(qs,o):
    deg=sorted(set(map(int,o['degrees'].split(','))));breaks=list(map(float,o['breaks'].split(',')))
    if min(deg)<4 or max(deg)>128 or breaks[0]!=0 or breaks[-1]!=1 or min(np.diff(breaks))<=0:
        raise ValueError('Use degrees in [4,128], ordered breaks from 0 to 1')
    specs=[]
    for q in qs:
        for p in deg:
            specs.append({'q':q,'tag':f'Cheb-p{p}','degree':p,'breaks':breaks,'resolution':(len(breaks)-1)*p,
                          'role':'space','method':'Radau','rtol':o['rtol'],'atol_T':o['atol_T'],'atol_C':o['atol_C'],'max_step':o['max_step']})
        if not o['no_time_check']:
            s={**specs[-1]};s.update(tag=f'Cheb-p{deg[-1]}-time-tight',role='time',rtol=o['rtol']/8,
                atol_T=o['atol_T']/8,atol_C=o['atol_C']/8,max_step=o['max_step']/2);specs.append(s)
    return specs


def test_spec():return {'q':4,'degree':8,'breaks':[0.,.4,.9,1.]}


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:print('Interrupted; accepted checkpoints retained.',file=sys.stderr);sys.exit(130)
    except Exception:traceback.print_exc();sys.exit(1)

