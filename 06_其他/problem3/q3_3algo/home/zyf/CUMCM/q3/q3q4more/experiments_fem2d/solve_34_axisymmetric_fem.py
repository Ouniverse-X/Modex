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
adaptive implicit time integration. Production mode uses one persistent BDF solver and
forces accepted steps to LAND exactly on every measured piecewise-linear forcing kink
without destroying multistep history/LU reuse. --knot-policy all retains the stricter
restart-at-every-kink audit baseline; critical is an additional experimental audit mode.
Constant derivatives/geometry are precomputed; changing material coefficients and
Newton matrices are NOT wrongly frozen. Trial states are never clipped.

--workers is an aggregate CPU ceiling, not a promise to occupy every CPU.
Independent Q/mesh/tolerance/end-condition cases run in separate spawn processes.
No identical case is scheduled twice. On high-memory nodes the throughput scheduler
runs all unique validation cases concurrently, because SuperLU is mostly single-threaded;
fine and light cases receive disjoint pinned CPU groups. The older balanced scheduler
remains available for low-memory machines. BLAS threads=1.
Periodic checkpointing is OFF by default for maximum throughput. A tiny live CSV status
is appended at progress cadence. Existing compatible partial state from the interrupted
legacy run can be adopted once; no SIGINT/emergency state serialization is performed.

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
# Spawned FEM workers inherit these before their Python runtime starts.  Keeping the
# number of glibc malloc arenas small prevents allocator fragmentation from inflating RSS.
os.environ.setdefault('MALLOC_ARENA_MAX','2')
os.environ.setdefault('MALLOC_TRIM_THRESHOLD_','262144')
os.environ['NUMBA_NUM_THREADS']='24'
try:
    import numpy as np
    import scipy
    from scipy.integrate import Radau, BDF
    from scipy.sparse import csr_matrix, csc_matrix, issparse
    from scipy.sparse.linalg import splu
    from scipy.optimize import brentq
    from scipy.fft import dct
    import numba
    from numba import njit, prange
except ImportError as e:
    raise SystemExit('Dependencies: python3 -m pip install numpy scipy numba\n'+str(e))


_SciPyBDF = BDF

class FastBDF(_SciPyBDF):
    """Memory-stable BDF with fill-reduced SuperLU and exact field-block splitting.

    IMPORTANT: scipy SuperLU's ``splu`` is not safe to run concurrently from two Python
    threads in this workload: repeated concurrent factorizations retain large amounts of
    native memory.  The previous parallel2 implementation therefore showed monotonic RSS
    growth even though Python references were released.

    For ``--jacobian-mode block`` the Newton matrix is *exactly block diagonal* between
    T and C.  We still exploit that structure, but factor the two independent blocks
    sequentially inside the case process.  This is mathematically the same linear system
    as the previous split-field implementation; only the execution order changes.

    The CSC extraction pattern is cached once and two numeric data buffers / CSC wrappers
    are reused on every factorization.  This avoids repeated sparse slicing and removes a
    second source of allocator churn.  MMD_AT_PLUS_A and the pivot threshold are unchanged.
    """
    pivot_threshold = 0.01
    ordering = 'MMD_AT_PLUS_A'
    split_block_lu = True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._split_fields=False
        self._split_maps=None
        self._AT_data=self._AC_data=None
        self._AT_matrix=self._AC_matrix=None
        if issparse(self.J):
            if self.split_block_lu:
                coo=self.J.tocoo(copy=False)
                self._split_fields=bool(np.all((coo.row & 1)==(coo.col & 1)))

            def _build_split_maps(A):
                if A.shape[0] != A.shape[1] or (A.shape[0] & 1):
                    raise RuntimeError('split-field LU requires an even square matrix')
                n2=A.shape[0]; n=n2//2
                tpos=[];cpos=[];tidx=[];cidx=[];tip=[0];cip=[0]
                for col2 in range(0,n2,2):
                    lo,hi=int(A.indptr[col2]),int(A.indptr[col2+1])
                    rows=A.indices[lo:hi]
                    if np.any(rows & 1):
                        raise RuntimeError('block Jacobian unexpectedly contains T-C entries')
                    tpos.append(np.arange(lo,hi,dtype=np.int64))
                    tidx.append((rows//2).astype(A.indices.dtype,copy=True))
                    tip.append(tip[-1]+hi-lo)
                for col2 in range(1,n2,2):
                    lo,hi=int(A.indptr[col2]),int(A.indptr[col2+1])
                    rows=A.indices[lo:hi]
                    if np.any((rows & 1)==0):
                        raise RuntimeError('block Jacobian unexpectedly contains C-T entries')
                    cpos.append(np.arange(lo,hi,dtype=np.int64))
                    cidx.append((rows//2).astype(A.indices.dtype,copy=True))
                    cip.append(cip[-1]+hi-lo)
                tp=np.concatenate(tpos); cp=np.concatenate(cpos)
                ti=np.concatenate(tidx); ci=np.concatenate(cidx)
                tip=np.asarray(tip,dtype=A.indptr.dtype); cip=np.asarray(cip,dtype=A.indptr.dtype)
                self._AT_data=np.empty(tp.size,dtype=A.data.dtype)
                self._AC_data=np.empty(cp.size,dtype=A.data.dtype)
                self._AT_matrix=csc_matrix((self._AT_data,ti,tip),shape=(n,n),copy=False)
                self._AC_matrix=csc_matrix((self._AC_data,ci,cip),shape=(n,n),copy=False)
                self._split_maps=(tp,cp)

            def lu(A):
                self.nlu += 1
                if not self._split_fields:
                    return splu(A,permc_spec=str(self.ordering),diag_pivot_thresh=float(self.pivot_threshold))
                if self._split_maps is None:
                    _build_split_maps(A)
                tp,cp=self._split_maps
                # np.take(..., out=...) performs no per-factorization data-buffer allocation.
                np.take(A.data,tp,out=self._AT_data)
                np.take(A.data,cp,out=self._AC_data)
                kw=dict(permc_spec=str(self.ordering),diag_pivot_thresh=float(self.pivot_threshold))
                # Sequential SuperLU is deliberate: concurrent splu calls from Python threads
                # exhibit native-memory retention.  Each block remains independently factored.
                LT=splu(self._AT_matrix,**kw)
                LC=splu(self._AC_matrix,**kw)
                return (LT,LC)

            def solve_lu(LU,b):
                if not self._split_fields:
                    return LU.solve(b)
                x=np.empty_like(b)
                x[0::2]=LU[0].solve(np.ascontiguousarray(b[0::2]))
                x[1::2]=LU[1].solve(np.ascontiguousarray(b[1::2]))
                return x

            self.lu=lu
            self.solve_lu=solve_lu

VERSION='2.3.0-memstable-sequential-splitlu'
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
DEFAULT_INPUT='/home/zyf/CUMCM/problem/attachment/附件1.xlsx'
DEFAULT_RADIUS='/home/zyf/CUMCM/problem/attachment/附件2.xlsx'
LEGACY_SOURCE_SHA256='ffa9ad91894af0803ba7de646b8958ba5f5c8a775a26857f09e274b0b4ab4248'


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


def _rss_mib():
    try:
        import resource
        x=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return x/(1024. if sys.platform!='darwin' else 1024.*1024.)
    except Exception:
        return float('nan')



def _mem_available_gb():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return float(line.split()[1])/1024./1024.
    except Exception:
        pass
    return float('nan')

_WORKER_DONATION_EVENT=None
_WORKER_FULL_GROUP=None
_WORKER_DONATED=True

def worker_init(threads,cpu_groups=None,counter=None,donation_event=None,full_groups=None):
    """Pin each worker.  Extreme mode can donate idle light-worker cores later."""
    global _WORKER_DONATION_EVENT,_WORKER_FULL_GROUP,_WORKER_DONATED
    group=None;idx=0
    if cpu_groups and counter is not None:
        with counter.get_lock():
            idx=counter.value;counter.value+=1
        group=cpu_groups[idx % len(cpu_groups)]
        if group and hasattr(os,'sched_setaffinity'):
            try:os.sched_setaffinity(0,group)
            except OSError:pass
    actual=max(1,min(int(threads),len(group) if group else int(threads)))
    numba.set_num_threads(actual)
    _WORKER_DONATION_EVENT=donation_event
    _WORKER_FULL_GROUP=(full_groups[idx % len(full_groups)] if full_groups else group)
    _WORKER_DONATED=(donation_event is None or _WORKER_FULL_GROUP==group)

def maybe_expand_worker():
    global _WORKER_DONATED
    if _WORKER_DONATED or _WORKER_DONATION_EVENT is None or not _WORKER_DONATION_EVENT.is_set():return False
    group=_WORKER_FULL_GROUP
    if group and hasattr(os,'sched_setaffinity'):
        try:os.sched_setaffinity(0,group)
        except OSError:pass
    if group:numba.set_num_threads(len(group))
    _WORKER_DONATED=True
    return True


@njit(cache=True,nogil=True)
def material_into(T,C,q,val):
    n=T.size
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


@njit(cache=True,nogil=True)
def material(T,C,q):
    val=np.empty((7,T.size));material_into(T,C,q,val);return val


@njit(cache=True,nogil=True,parallel=True)
def local_rhs_into(y,conn,G0,G1,W0,W1,coef,a0,a1,out):
    ne,m=conn.shape
    for e in prange(ne):
        for i in range(m):out[e,i,0]=0.;out[e,i,1]=0.
        base=conn[e,0]
        for k in range(m):
            gt0=0.;gt1=0.;gc0=0.;gc1=0.
            for j in range(m):
                idx=conn[e,j];t=y[2*idx]-y[2*base];c=y[2*idx+1]-y[2*base+1]
                gt0+=G0[e,k,j]*t;gc0+=G0[e,k,j]*c
                gt1+=G1[e,k,j]*t;gc1+=G1[e,k,j]*c
            idx=conn[e,k];kval=coef[2,idx];d=coef[4,idx]
            wr=a0*W0[e,k];wz=a1*W1[e,k]
            for i in range(m):
                out[e,i,0]-=kval*(wr*G0[e,k,i]*gt0+wz*G1[e,k,i]*gt1)
                out[e,i,1]-=d*(wr*G0[e,k,i]*gc0+wz*G1[e,k,i]*gc1)


@njit(cache=True,nogil=True)
def scatter_rhs_into(local,conn,out):
    for k in range(out.shape[0]):out[k,0]=0.;out[k,1]=0.
    for e in range(conn.shape[0]):
        for i in range(conn.shape[1]):
            k=conn[e,i];out[k,0]+=local[e,i,0];out[k,1]+=local[e,i,1]


@njit(cache=True,nogil=True,parallel=True)
def local_jac_into(y,conn,G0,G1,W0,W1,coef,a0,a1,out):
    ne,m=conn.shape
    for e in prange(ne):
        for b in range(4):
            for i in range(m):
                for j in range(m):out[e,b,i,j]=0.
        base=conn[e,0]
        for k in range(m):
            gt0=0.;gt1=0.;gc0=0.;gc1=0.
            for j in range(m):
                idx=conn[e,j];t=y[2*idx]-y[2*base];c=y[2*idx+1]-y[2*base+1]
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


@njit(cache=True,nogil=True,parallel=True)
def local_jac_block_into(y,conn,G0,G1,W0,W1,coef,a0,a1,out):
    """Same-field Jacobian blocks only (TT, CC) for a quasi-Newton iteration.

    The RHS remains fully coupled.  Omitting cross derivatives changes only the Newton
    matrix used to converge each implicit BDF stage; it does not split or alter the ODE.
    """
    ne,m=conn.shape
    for e in prange(ne):
        for b in range(2):
            for i in range(m):
                for j in range(m):out[e,b,i,j]=0.
        base=conn[e,0]
        for k in range(m):
            gt0=0.;gt1=0.;gc0=0.;gc1=0.
            for j in range(m):
                idx=conn[e,j];tv=y[2*idx]-y[2*base];cv=y[2*idx+1]-y[2*base+1]
                gt0+=G0[e,k,j]*tv;gc0+=G0[e,k,j]*cv
                gt1+=G1[e,k,j]*tv;gc1+=G1[e,k,j]*cv
            idx=conn[e,k];kv=coef[2,idx];d=coef[4,idx];dc=coef[5,idx]
            wr=a0*W0[e,k];wz=a1*W1[e,k]
            for i in range(m):
                cr=wr*G0[e,k,i];cz=wz*G1[e,k,i]
                qc=cr*gc0+cz*gc1
                out[e,1,i,k]-=dc*qc
                for j in range(m):
                    v=cr*G0[e,k,j]+cz*G1[e,k,j]
                    out[e,0,i,j]-=kv*v
                    out[e,1,i,j]-=d*v


@njit(cache=True,nogil=True)
def state_extrema(y):
    tmin=1e300;tmax=-1e300;cmin=1e300;cmax=-1e300;ok=True
    for i in range(0,y.size,2):
        t=y[i];c=y[i+1]
        if not math.isfinite(t+c):ok=False
        if t<tmin:tmin=t
        if t>tmax:tmax=t
        if c<cmin:cmin=c
        if c>cmax:cmax=c
    return tmin,tmax,cmin,cmax,ok


class GalerkinSystem:
    def setup(self):
        """Freeze sparse topology and allocate reusable work buffers once per case."""
        self.conn=np.ascontiguousarray(self.conn,dtype=np.int64)
        for name in ('G0','G1','W0','W1','mass','side','end'):
            setattr(self,name,np.ascontiguousarray(getattr(self,name),dtype=np.float64))
        self.n=len(self.mass);ne,m=self.conn.shape;n2=2*self.n
        if np.min(self.mass)<=0 or abs(self.mass.sum()-1)>1e-10:raise ValueError('Bad positive quadrature mass')
        keys=[]
        for fi,fj in ((0,0),(0,1),(1,0),(1,1)):
            keys.append((2*self.conn[:,:,None]+fi)*n2+(2*self.conn[:,None,:]+fj))
        kk=np.stack(keys,axis=1)
        uniq,slots=np.unique(kk.ravel(),return_inverse=True)
        self.slots=np.ascontiguousarray(slots,dtype=np.int32)
        self.rows=uniq//n2;self.indices=(uniq%n2).astype(np.int32)
        self.indptr=np.r_[0,np.cumsum(np.bincount(self.rows,minlength=n2))].astype(np.int32)
        diagkeys=np.arange(n2,dtype=np.int64)*(n2+1)
        self.diag_slots=np.searchsorted(uniq,diagkeys)
        if np.any(uniq[self.diag_slots]!=diagkeys):raise AssertionError('Missing Jacobian diagonal')
        tc_keys=(2*np.arange(self.n,dtype=np.int64))*n2+2*np.arange(self.n)+1
        self.tc_slots=np.searchsorted(uniq,tc_keys)
        if np.any(uniq[self.tc_slots]!=tc_keys):raise AssertionError('Missing T-C Jacobian slots')
        # CSR->CSC data permutation is static. Avoid an O(nnz log nnz) conversion every Jacobian refresh.
        marker=csr_matrix((np.arange(len(self.rows),dtype=np.float64)+1.,self.indices,self.indptr),shape=(n2,n2)).tocsc()
        self.csc_indices=np.ascontiguousarray(marker.indices,dtype=np.int32)
        self.csc_indptr=np.ascontiguousarray(marker.indptr,dtype=np.int32)
        self.csc_from_csr=np.ascontiguousarray(marker.data.astype(np.int64)-1,dtype=np.int32)
        # Separate same-field structure for the production block quasi-Newton Jacobian.
        # Explicit T-C/C-T structural zeros are omitted from SuperLU entirely.
        s4=self.slots.reshape(ne,4,m,m)
        bd_src=np.stack((s4[:,0],s4[:,3]),axis=1)
        # Map full-CSR slot ids to a compact same-field CSR id.
        bd_full_ids=np.unique(bd_src.ravel())
        self.bd_full_ids=np.ascontiguousarray(bd_full_ids,dtype=np.int32)
        remap=np.full(len(self.rows),-1,dtype=np.int32);remap[bd_full_ids]=np.arange(len(bd_full_ids),dtype=np.int32)
        self.bd_slots=np.ascontiguousarray(remap[bd_src],dtype=np.int32)
        br=self.rows[bd_full_ids];bi=self.indices[bd_full_ids]
        bindptr=np.r_[0,np.cumsum(np.bincount(br,minlength=n2))].astype(np.int32)
        bdiagkeys=np.arange(n2,dtype=np.int64)*(n2+1)
        bkeys=br.astype(np.int64)*n2+bi.astype(np.int64)
        self.bd_diag_slots=np.searchsorted(bkeys,bdiagkeys)
        if np.any(bkeys[self.bd_diag_slots]!=bdiagkeys):raise AssertionError('Missing block Jacobian diagonal')
        bmark=csr_matrix((np.arange(len(br),dtype=np.float64)+1.,bi,bindptr),shape=(n2,n2)).tocsc()
        self.bd_csc_indices=np.ascontiguousarray(bmark.indices,dtype=np.int32)
        self.bd_csc_indptr=np.ascontiguousarray(bmark.indptr,dtype=np.int32)
        self.bd_csc_from_csr=np.ascontiguousarray(bmark.data.astype(np.int64)-1,dtype=np.int32)
        del bmark,remap,bd_src,bd_full_ids,br,bi,bindptr,bkeys,s4,marker,kk,uniq,keys,slots
        # Persistent workspaces: eliminate repeated multi-MiB allocations inside RHS/Jacobian calls.
        self.coef_buf=np.empty((7,self.n),dtype=np.float64)
        self.local_rhs_buf=np.empty((ne,m,2),dtype=np.float64)
        self.raw_buf=np.empty((self.n,2),dtype=np.float64)
        self.den_buf=np.empty((self.n,2),dtype=np.float64)
        self.b_buf=np.empty(self.n,dtype=np.float64)
        self.f=np.empty(n2,dtype=np.float64)
        self.last_y=np.empty(n2,dtype=np.float64);self.last_valid=False;self.last_t=None
        self.local_jac_buf=None
        self.local_jac_block_buf=None
        self.bd_data_buf=np.empty(len(self.bd_full_ids),dtype=np.float64)
        self.static_nnz=len(self.rows)
        self.block_nnz=len(self.bd_full_ids)
        self.jacobian_mode='exact'

    def geometry(self,t):
        R,H,airT,airC=self.forcing(t)
        self.b_buf[:]=(2./R)*self.side
        if self.end_factor:self.b_buf[:]+=(self.end_factor/H)*self.end
        return 4/R**2,1/H**2,self.b_buf,airT,airC

    def cache_state(self,t,y):
        if self.last_t!=t or not self.last_valid or not np.array_equal(y,self.last_y):
            material_into(y[::2],y[1::2],self.q,self.coef_buf)
            a0,a1,b,airT,airC=self.geometry(t)
            local_rhs_into(y,self.conn,self.G0,self.G1,self.W0,self.W1,self.coef_buf,a0,a1,self.local_rhs_buf)
            scatter_rhs_into(self.local_rhs_buf,self.conn,self.raw_buf)
            self.raw_buf[:,0]+=self.ht*b*(airT-y[::2]);self.raw_buf[:,1]+=self.hm*b*(airC-y[1::2])
            self.den_buf[:,0]=self.mass*self.coef_buf[0];self.den_buf[:,1]=self.mass
            self.f[::2]=self.raw_buf[:,0]/self.den_buf[:,0];self.f[1::2]=self.raw_buf[:,1]/self.den_buf[:,1]
            self.coef=self.coef_buf;self.den=self.den_buf.ravel();self.b=b;self.a0=a0;self.a1=a1
            np.copyto(self.last_y,y);self.last_t=t;self.last_valid=True
        return self.f

    def rhs(self,t,y):return self.cache_state(t,y).copy()

    def jac(self,t,y):
        self.cache_state(t,y)
        ne,m=self.conn.shape
        if self.jacobian_mode=='block':
            if self.local_jac_block_buf is None:self.local_jac_block_buf=np.empty((ne,2,m,m),dtype=np.float64)
            local_jac_block_into(y,self.conn,self.G0,self.G1,self.W0,self.W1,self.coef,self.a0,self.a1,self.local_jac_block_buf)
            data=np.bincount(self.bd_slots.ravel(),weights=self.local_jac_block_buf.ravel(),minlength=len(self.bd_full_ids))
            data[self.bd_diag_slots[::2]]-=self.ht*self.b
            data[self.bd_diag_slots[1::2]]-=self.hm*self.b
            data/=self.den[self.rows[self.bd_full_ids]]
            cdata=np.ascontiguousarray(data[self.bd_csc_from_csr])
            return csc_matrix((cdata,self.bd_csc_indices,self.bd_csc_indptr),shape=(2*self.n,2*self.n),copy=False)
        if self.local_jac_buf is None:self.local_jac_buf=np.empty((ne,4,m,m),dtype=np.float64)
        local_jac_into(y,self.conn,self.G0,self.G1,self.W0,self.W1,self.coef,self.a0,self.a1,self.local_jac_buf)
        data=np.bincount(self.slots,weights=self.local_jac_buf.ravel(),minlength=len(self.rows))
        data[self.diag_slots[::2]]-=self.ht*self.b
        data[self.diag_slots[1::2]]-=self.hm*self.b
        data/=self.den[self.rows]
        data[self.tc_slots]-=self.f[::2]*self.coef[1]/self.coef[0]
        cdata=np.ascontiguousarray(data[self.csc_from_csr])
        return csc_matrix((cdata,self.csc_indices,self.csc_indptr),shape=(2*self.n,2*self.n),copy=False)


def _uniform_grid(t):
    d=np.diff(t)
    if len(d)==0:return None
    h=float(d[0])
    if h<=0 or np.max(np.abs(d-h))>1e-10*max(1.,abs(h)):return None
    return float(t[0]),1./h,float(t[-1])

def _interp_fast(t,values,grid,times):
    if grid is None:return float(np.interp(t,times,values))
    t0,inv,last=grid
    if t<=t0:return float(values[0])
    if t>=last:return float(values[-1])
    x=(t-t0)*inv;i=int(x);f=x-i
    if i>=len(values)-1:return float(values[-1])
    return float(values[i]+f*(values[i+1]-values[i]))

def make_forcing(env,radius,q,o):
    # The supplied environment/radius tables are uniform.  Detect that once so the hot RHS
    # path uses exact O(1) piecewise-linear interpolation instead of repeated binary searches.
    tail=np.vstack((env,[env[-1,0]+o['tail_ramp'],o['air_temperature'],o['air_moisture']]))
    eg=_uniform_grid(tail[:,0]);rg=_uniform_grid(radius[:,0]) if q==4 else None
    def forcing(t):
        if q==4:
            if t>radius[-1,0]+1e-8 and o['radius_tail']=='error':raise ValueError('Radius coverage exceeded')
            R=_interp_fast(t,radius[:,1],rg,radius[:,0])
        else:R=.02
        H=.125*(R/.02 if o['axial_shrink']=='isotropic' and q==4 else 1.)
        return R,H,_interp_fast(t,tail[:,1],eg,tail[:,0]),_interp_fast(t,tail[:,2],eg,tail[:,0])
    return forcing,tail


def knot_times(env,radius,q,limit,o):
    """Hard integration segment boundaries.

    Interior XLSX interpolation knots are continuous (only the derivative jumps). Recreating
    BDF at every 60-s row destroys multistep history and LU reuse, which is very expensive in
    2-D. 'critical' keeps the exact same piecewise-linear forcing but hard-stops only where the
    forcing law itself changes class (measured -> declared tail -> constant) and at the final
    radius support. 'all' reproduces the legacy exact-stop-at-every-kink policy for audits.
    """
    policy=o.get('knot_policy','critical')
    if policy in ('all','aligned'):
        times=list(env[:,0])+[env[-1,0]+o['tail_ramp'],limit]
        if q==4:
            slope=np.diff(radius[:,1])/np.diff(radius[:,0])
            keep=np.r_[True,np.abs(np.diff(slope))>2e-19,True]
            times.extend(radius[keep,0])
    else:
        times=[env[-1,0],env[-1,0]+o['tail_ramp'],limit]
        if q==4:times.append(radius[-1,0])
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
    live_dir=folder.parent/'live';live_dir.mkdir(exist_ok=True)
    live_path=live_dir/(f'Q{spec["q"]}_{spec["tag"]}.csv'.replace('/','_'))
    if finished.exists() and not o['no_cache']:
        meta,_=load_npz(finished,key);print(f'[cache] {label}: {meta["status"]}',flush=True)
        return {'key':key,'path':str(finished),'meta':meta}
    build0=time.perf_counter();print(f'[start] {label}: building Q2 mesh/Jacobian topology; pid={os.getpid()}; threads={numba.get_num_threads()}; RSS={_rss_mib():.1f} MiB',flush=True)
    model=make_model(spec,o)
    print(f'[ready] {label}: nodes/field={model.n:,}; coupled DOF={2*model.n:,}; Jacobian nnz={model.static_nnz:,}; build={time.perf_counter()-build0:.2f}s; RSS={_rss_mib():.1f} MiB',flush=True)
    model.forcing,tail=make_forcing(env,radius,spec['q'],o)
    model.q=spec['q'];model.ht=o['heat_transfer'];model.hm=o['mass_transfer'];model.jacobian_mode=spec.get('jacobian_mode',o.get('jacobian_mode','block'))
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
        print(f'[resume] {label}: {t/3600:.6f} h (integrator history restarted once at checkpoint)',flush=True)
    limit=o['max_hours']*3600
    if spec['q']==4 and o['radius_tail']=='error':limit=min(limit,float(radius[-1,0]))
    breaks=knot_times(env,radius,spec['q'],limit,o)
    next_out=60*(math.floor(t/60+1e-10)+1)
    def next_log_after(tt):
        d=o['startup_log_seconds'] if tt<o['startup_log_until'] else o['log_seconds']
        return d*(math.floor(tt/d+1e-12)+1)
    next_log=next_log_after(t)
    next_snap=21600*(math.floor(t/21600)+1)
    threshold=o['threshold']
    def live_write(tt,cmax,solver=None,status='RUNNING'):
        # Tiny status-only write at progress cadence; never serializes the state/Jacobian.
        new=not live_path.exists()
        with live_path.open('a',encoding='utf8',buffering=1) as f:
            if new:f.write('status,t_h,maxC,accepted_steps,nlu,wall_s,rss_MiB\n')
            nlu=calls['nlu']+(getattr(solver,'nlu',0) if solver is not None else 0)
            f.write(f'{status},{tt/3600:.9f},{cmax:.12g},{calls["accepted_steps"]},{nlu},{time.perf_counter()-wall:.3f},{_rss_mib():.1f}\n')
    atol=np.tile([spec['atol_T'],spec['atol_C']],model.n)
    forcing_T_min=min(28.,float(np.min(tail[:,1])));forcing_T_max=max(28.,float(np.max(tail[:,1])))
    forcing_C_min=min(2.55,float(np.min(tail[:,2])));forcing_C_max=max(2.55,float(np.max(tail[:,2])))

    def dry_measure(yy,nodal=None):
        if nodal is None:nodal=float(np.max(yy[1::2]))
        if nodal>threshold+.005:return nodal
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
        ph=model.sample(pts,yy.reshape(-1,2));ph[~valid]=0.
        records['physical'].append(ph);records['physical_valid'].append(valid.astype(np.int8))
        nodal=float(np.max(yy[1::2]))
        lower,upper=model.maximum(yy[1::2],o['max_enclosure_tol']) if nodal<threshold+.005 else (nodal,nodal)
        records['stats'].append([tt,R,H,float(np.min(yy[1::2])),lower,upper,float(model.mass@yy[1::2])])

    def accept_step(solver,t0,g0):
        nonlocal t,y,event,next_out,next_snap,next_log,last_save,minC,maxC,minT,maxT
        msg=solver.step()
        if solver.status=='failed':raise RuntimeError(f'{label} failed at {solver.t}: {msg}')
        t1=solver.t;y1=solver.y;calls['accepted_steps']+=1
        if (calls['accepted_steps'] & 31)==0 and maybe_expand_worker():
            print(f'[cpu-donation] {label}: expanded to {numba.get_num_threads()} Numba threads; affinity={sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else "n/a"}',flush=True)
        # Four reductions are cheap compared with sparse LU and are kept as hard physical guards.
        tmin,tmax,cmin,cmax,finite=state_extrema(y1)
        if not finite or cmin<=0 or tmin<=-273.15:raise RuntimeError(f'{label}: invalid accepted state')
        minC=min(minC,cmin);maxC=max(maxC,cmax);minT=min(minT,tmin);maxT=max(maxT,tmax)
        if maxC>forcing_C_max+o['range_tolerance'] or minC<forcing_C_min-o['range_tolerance'] or minT<forcing_T_min-o['range_tolerance'] or maxT>forcing_T_max+o['range_tolerance']:
            raise RuntimeError(f'{label}: significant range violation at t={t1:.8g}: C=[{minC:.12g},{maxC:.12g}], T=[{minT:.12g},{maxT:.12g}]; refine space/time, do not clip')
        g1=dry_measure(y1,cmax)-threshold;stop=t1
        need_event=(g1<0 and g0>=0);need_out=(next_out<=stop+1e-8);need_snap=(next_snap<=stop+1e-8)
        dense=solver.dense_output() if (need_event or need_out or need_snap) else None
        if need_event:
            lo=t0;hi=t1
            while hi-lo>o['event_tol']:
                mid=.5*(lo+hi)
                if dry_measure(dense(mid))<threshold:hi=mid
                else:lo=mid
            stop=hi;ev_y=dense(hi);low,up=model.maximum(ev_y[1::2],o['max_enclosure_tol'])
            event={'lower_s':lo,'upper_s':hi,'width_s':hi-lo,'time_h':hi/3600,
                   'C_max_lower':low,'C_max_upper':up,'method':model.maximum_method,
                   'meaning':'Numerical-interpolant event bracket, NOT a PDE/model error bound'}
        while next_out<=stop+1e-8:
            yy=y1 if abs(next_out-t1)<=1e-10*max(1.,abs(t1)) else dense(next_out)
            append_output(next_out,yy);next_out+=60
        while next_snap<=stop+1e-8:
            yy=y1 if abs(next_snap-t1)<=1e-10*max(1.,abs(t1)) else dense(next_snap)
            records['snap_times'].append(float(next_snap));records['snapshots'].append(yy.reshape(-1,2).copy());next_snap+=21600
        if event is not None:t=stop;y=ev_y
        else:t=t1;y=y1
        if t>=next_log:
            print(f'[{label}] t={t/3600:.4f} h; maxC={cmax:.9f}; DOF={2*model.n}; steps={calls["accepted_steps"]}; nlu={calls["nlu"]+getattr(solver,"nlu",0)}; wall={time.perf_counter()-wall:.1f}s; RSS={_rss_mib():.1f} MiB',flush=True)
            live_write(t,cmax,solver)
            next_log=next_log_after(t)
        if not o['no_cache'] and o['checkpoint_seconds']>0 and t-last_save>=o['checkpoint_seconds']:
            a=pack_records(records,nprobe)
            if a['snapshots'].size==0:a['snapshots']=np.empty((0,model.n,2))
            running_calls={**calls}
            for k in ('nfev','njev','nlu'):running_calls[k]+=getattr(solver,k)
            save_npz(partial,{'key':key,'t':t,'wall_s':saved_wall+time.perf_counter()-wall,'calls':running_calls,'ranges':[minC,maxC,minT,maxT]},y=np.asarray(y).copy(),**a)
            last_save=t
        return dense,g1

    FastBDF.pivot_threshold=float(o.get('superlu_pivot',0.01))
    FastBDF.ordering=str(o.get('superlu_ordering','MMD_AT_PLUS_A'))
    FastBDF.split_block_lu=not bool(o.get('no_parallel_block_lu',False))
    solver_cls=Radau if spec['method']=='Radau' else FastBDF
    policy=o.get('knot_policy','aligned')
    if policy=='aligned':
        # One persistent multistep solver. It lands exactly on every forcing/radius kink by
        # temporarily limiting max_step, but DOES NOT destroy BDF history/LU reuse there.
        init0=time.perf_counter()
        solver=solver_cls(model.rhs,t,y,limit,rtol=spec['rtol'],atol=atol,jac=model.jac,max_step=spec['max_step'])
        split_lu_mode = 'sequential-memory-stable' if getattr(solver, '_split_fields', False) else 'off'
        print(f'[integrator] {label}: persistent {solver_cls.__name__} t={t:.3f}->{limit:.3f}s initialized in {time.perf_counter()-init0:.2f}s; exact-knot alignment without restart; knots={len(breaks)}; SuperLU={FastBDF.ordering}; split-field-LU={split_lu_mode}; RSS={_rss_mib():.1f} MiB', flush=True)
        ik=int(np.searchsorted(breaks,t+1e-9,side='right'))
        early_max=float(spec['max_step']);late_max=float(spec.get('late_max_step',early_max));tail_end=float(tail[-1,0]);g_prev=dry_measure(solver.y)-threshold
        while solver.status=='running' and event is None:
            while ik<len(breaks) and breaks[ik]<=solver.t+max(1e-9,2e-13*max(1.,abs(solver.t))):ik+=1
            if ik<len(breaks):
                remain=breaks[ik]-solver.t
                if remain<0:raise RuntimeError(f'{label}: crossed a forcing knot unexpectedly')
                current_cap=late_max if solver.t>=tail_end-1e-9 else early_max
                solver.max_step=min(current_cap,max(remain,1e-12))
            else:solver.max_step=late_max if solver.t>=tail_end-1e-9 else early_max
            t0=solver.t;g0=g_prev
            _,g_prev=accept_step(solver,t0,g0)
            if ik<len(breaks):
                eps=max(2e-8,5e-13*max(1.,abs(breaks[ik])))
                if solver.t>breaks[ik]+eps:raise RuntimeError(f'{label}: BDF stepped across aligned knot {breaks[ik]} -> {solver.t}')
                if abs(solver.t-breaks[ik])<=eps:ik+=1
        for k in ('nfev','njev','nlu'):calls[k]+=getattr(solver,k)
    else:
        # Audit/legacy paths: 'all' restarts at every kink; 'critical' only at class changes.
        last_h=None;g_prev=dry_measure(y)-threshold
        for end in breaks:
            if end<=t+1e-9:continue
            first=min(last_h,end-t) if last_h is not None else None
            init0=time.perf_counter()
            solver=solver_cls(model.rhs,t,y,end,rtol=spec['rtol'],atol=atol,jac=model.jac,max_step=(spec.get('late_max_step',spec['max_step']) if t>=float(tail[-1,0])-1e-9 else spec['max_step']),first_step=first)
            print(f'[integrator] {label}: {solver_cls.__name__} t={t:.3f}->{end:.3f}s initialized in {time.perf_counter()-init0:.2f}s; knot-policy={policy}; RSS={_rss_mib():.1f} MiB',flush=True)
            while solver.status=='running' and event is None:
                t0=solver.t;g0=g_prev
                _,g_prev=accept_step(solver,t0,g0)
                if event is None:last_h=min(solver.step_size or spec['max_step'],spec['max_step'])
            for k in ('nfev','njev','nlu'):calls[k]+=getattr(solver,k)
            if event is not None:break
            t=float(end);y=solver.y

    append_output(t,y) if not records['times'] or abs(records['times'][-1]-t)>1e-8 else None
    records['snap_times'].append(t);records['snapshots'].append(np.asarray(y).reshape(-1,2).copy())
    a=pack_records(records,nprobe)
    meta={'key':key,'status':'DRIED' if event else 'NOT_DRIED','spec':spec,'t':t,'event':event,
          'wall_s':saved_wall+time.perf_counter()-wall,'calls':calls,'nodes_per_field':model.n,
          'ranges':[minC,maxC,minT,maxT], 'maximum_nodal_range_overshoot':max(0.,maxC-forcing_C_max,forcing_C_min-minC,forcing_T_min-minT,maxT-forcing_T_max), 'mesh':model.mesh_info,
          'method_detail':model.description, 'knot_policy':policy,'model_validation':False}
    save_npz(finished,meta,y=np.asarray(y).copy(),coordinates=model.coordinates,probe_coordinates=model.probe_coordinates,**a)
    if partial.exists():partial.unlink()
    live_write(t,float(np.max(np.asarray(y)[1::2])),None,meta['status'])
    print(f'[done] {label}: {meta["status"]}; t={t/3600:.9f} h; wall={meta["wall_s"]:.1f}s; accepted_steps={calls["accepted_steps"]}; nlu={calls["nlu"]}',flush=True)
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
    p.add_argument('--workers',type=int,default=24);p.add_argument('--jobs',type=int,default=10)
    p.add_argument('--threads-per-case',type=int,default=1)
    p.add_argument('--max-hours',type=float,default=72.);p.add_argument('--radius-tail',choices=['error','hold'],default='error')
    p.add_argument('--air-temperature',type=float,default=50.);p.add_argument('--air-moisture',type=float,default=.05)
    p.add_argument('--tail-ramp',type=float,default=60.);p.add_argument('--axial-shrink',choices=['fixed','isotropic'],default='fixed')
    p.add_argument('--heat-transfer',type=float,default=25.);p.add_argument('--mass-transfer',type=float,default=8e-7)
    p.add_argument('--threshold',type=float,default=.15);p.add_argument('--rtol',type=float,default=2e-9)
    p.add_argument('--atol-T',type=float,default=1e-10);p.add_argument('--atol-C',type=float,default=1e-11)
    p.add_argument('--max-step',type=float,default=30.,help='Maximum BDF step during measured/transitional forcing')
    p.add_argument('--late-max-step',type=float,default=300.,help='Maximum BDF step after environment reaches the declared constant tail; tolerances are unchanged')
    p.add_argument('--jacobian-mode',choices=['block','exact'],default='block',help='block = faster quasi-Newton same-field Jacobian with fully coupled RHS; exact = full analytic Jacobian')
    p.add_argument('--superlu-pivot',type=float,default=.01,help='SuperLU diagonal pivot threshold; 1 restores SciPy default')
    p.add_argument('--superlu-ordering',choices=['MMD_AT_PLUS_A','COLAMD','MMD_ATA','NATURAL'],default='MMD_AT_PLUS_A',help='SuperLU fill-reducing ordering; MMD_AT_PLUS_A is much faster/lower-fill on this Q2 Jacobian and is the production default')
    p.add_argument('--no-parallel-block-lu',action='store_true',help='Audit/debug: disable memory-stable T/C block splitting and use one coupled SuperLU factorization')
    p.add_argument('--event-tol',type=float,default=1e-4);p.add_argument('--max-enclosure-tol',type=float,default=2e-11)
    p.add_argument('--range-tolerance',type=float,default=2e-3,help='Hard abort threshold for high-order transient overshoot; recorded, never clipped')
    p.add_argument('--checkpoint-seconds',type=float,default=0.,help='Disabled by default for maximum speed; set >0 only if you explicitly want restart checkpoints')
    p.add_argument('--log-seconds',type=float,default=1800.)
    p.add_argument('--startup-log-seconds',type=float,default=60.,help='Progress cadence before --startup-log-until simulated seconds')
    p.add_argument('--startup-log-until',type=float,default=600.,help='Use fast startup progress logging until this simulated time')
    p.add_argument('--knot-policy',choices=['aligned','all','critical'],default='aligned',help='aligned lands exactly on every input kink WITHOUT restarting BDF history (recommended); all restarts at every kink for audit; critical only hard-stops at forcing-class changes')
    p.add_argument('--scheduler',choices=['extreme','throughput','balanced','flat'],default='extreme')
    p.add_argument('--max-fine-concurrent',type=int,default=2,help='Balanced mode: max simultaneous finest sparse-LU cases')
    p.add_argument('--fine-threads',type=int,default=0,help='Balanced mode: Numba threads per finest case; 0 uses workers/max-fine-concurrent')
    p.add_argument('--throughput-fine-threads',type=int,default=3,help='Throughput mode: Numba threads for finest cases')
    p.add_argument('--throughput-light-threads',type=int,default=2,help='Throughput mode: Numba threads for coarse/medium cases')
    p.add_argument('--throughput-min-memory-gb',type=float,default=96.,help='Below this MemAvailable, throughput mode falls back to balanced')
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
          f'时间积分 forcing-knot 策略：{o.get("knot_policy","critical")}；critical 不改变分段线性输入值，只避免在每个连续导数折点重启 BDF。',
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
    if args.max_fine_concurrent<1 or args.fine_threads<0:raise ValueError('Invalid balanced scheduler limits')
    if args.throughput_fine_threads<1 or args.throughput_light_threads<1 or args.throughput_min_memory_gb<=0:raise ValueError('Invalid throughput scheduler limits')
    threads=min(args.threads_per_case,cap);jobs=min(args.jobs,max(1,cap//threads))
    numba.set_num_threads(max(1,threads))
    tests=self_tests()
    if args.self_test_only:print(json.dumps(tests,indent=2));return 0
    for k in ('max_hours','tail_ramp','rtol','atol_T','atol_C','max_step','late_max_step','superlu_pivot','event_tol','max_enclosure_tol','log_seconds','startup_log_seconds','startup_log_until','field_target','time_target'):
        if not math.isfinite(o[k]) or o[k]<=0:raise ValueError(f'Invalid {k}')
    if args.checkpoint_seconds<0:raise ValueError('Negative checkpoint period')
    if args.late_max_step<args.max_step:raise ValueError('--late-max-step must be >= --max-step')
    if not 0<args.superlu_pivot<=1:raise ValueError('--superlu-pivot must be in (0,1]')
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
    # Performance knobs do not alter the mathematical case/cache identity. knot_policy DOES,
    # because it changes where BDF is restarted and therefore is intentionally retained.
    numeric={k:v for k,v in o.items() if k not in ('workers','jobs','threads_per_case','fine_threads','max_fine_concurrent','throughput_fine_threads','throughput_light_threads','throughput_min_memory_gb','scheduler','outdir','no_cache','checkpoint_seconds','log_seconds','startup_log_seconds','startup_log_until','input','radius_input','self_test_only','fem_meshes','degrees','questions','no_time_check','field_target','time_target')}
    tasks=[];legacy_keys={}
    # Previous v1.2 run compatibility is used only to salvage an already-written partial
    # checkpoint BEFORE the constant-tail stage.  That interval has exactly the same 30-s
    # maximum step and equations, so continuing from it avoids recomputing expensive early
    # transients.  Completed old cases are deliberately not mixed into new spatial/time
    # comparisons because v2 uses a larger late-time step cap.
    legacy_exclude={'late_max_step','jacobian_mode','superlu_pivot'}
    legacy_numeric={k:v for k,v in numeric.items() if k not in legacy_exclude}
    legacy_provenance={**provenance,'source_sha256':LEGACY_SOURCE_SHA256}
    for sp in specs:
        key=hashlib.sha256(json.dumps({'spec':sp,'numeric':numeric,'provenance':provenance},sort_keys=True).encode()).hexdigest()
        tasks.append((sp,key))
        oldsp={k:v for k,v in sp.items() if k not in ('late_max_step','jacobian_mode')}
        oldkey=hashlib.sha256(json.dumps({'spec':oldsp,'numeric':legacy_numeric,'provenance':legacy_provenance},sort_keys=True).encode()).hexdigest()
        legacy_keys[key]=oldkey
    tail_end=float(env[-1,0]+args.tail_ramp)
    if not args.no_cache:
        adopted=0
        for sp,key in tasks:
            newp=cache/(key+'.partial.npz');newf=cache/(key+'.npz');oldkey=legacy_keys[key];oldp=cache/(oldkey+'.partial.npz')
            if newp.exists() or newf.exists() or not oldp.exists():continue
            try:
                meta,a=load_npz(oldp,oldkey)
                if 0<float(meta.get('t',0))<=tail_end+1e-8:
                    meta={**meta,'key':key,'legacy_source_key':oldkey,'adopted_by_version':VERSION}
                    save_npz(newp,meta,**a);adopted+=1
                    print(f'[legacy-resume] adopted {sp["tag"]} Q{sp["q"]} at t={float(meta["t"])/3600:.3f} h from the interrupted v1.2 run.',flush=True)
            except Exception as exc:
                print(f'[legacy-resume] ignored incompatible old partial for Q{sp["q"]} {sp["tag"]}: {exc}',flush=True)
        if adopted:print(f'[legacy-resume] adopted {adopted} compatible pre-tail partial states; no completed old result was reused.',flush=True)

    def cpu_groups(njobs,nthreads):
        cpus=list(cpu['selected_cpus']);need=njobs*nthreads
        if need>len(cpus):raise RuntimeError('Internal CPU allocation exceeds cap')
        return [cpus[i*nthreads:(i+1)*nthreads] for i in range(njobs)]

    def run_wave(wave,wave_name,njobs,nthreads):
        if not wave:return []
        njobs=max(1,min(int(njobs),len(wave),cap))
        nthreads=max(1,min(int(nthreads),cap//njobs))
        groups=cpu_groups(njobs,nthreads)
        ctx=mp.get_context('spawn');counter=ctx.Value('i',0)
        print(f'[schedule] {wave_name}: {len(wave)} cases; concurrency={njobs}; Numba threads/case={nthreads}; CPU groups={groups}',flush=True)
        pool=ProcessPoolExecutor(max_workers=njobs,mp_context=ctx,initializer=worker_init,initargs=(nthreads,groups,counter))
        out=[];pending={}
        try:
            role_order={'space':0,'time':1,'end_control':2}
            ordered=sorted(wave,key=lambda v:(role_order.get(v[0].get('role'),9),v[0].get('q',0),-v[0]['resolution']))
            pending={pool.submit(run_case,s,env,radius,o,key,str(cache)):s for s,key in ordered}
            for f in as_completed(pending):out.append(f.result())
        except BaseException:
            for f in pending:f.cancel()
            for proc in getattr(pool,'_processes',{}).values():
                try:proc.terminate()
                except Exception:pass
            pool.shutdown(wait=True,cancel_futures=True);raise
        else:pool.shutdown(wait=True)
        return out

    selected_plan={}
    started=time.perf_counter();results=[]
    if args.scheduler=='flat':
        flat_jobs=max(1,min(jobs,len(tasks),cap//max(1,threads)))
        flat_threads=max(1,min(threads,cap//flat_jobs))
        selected_plan={'mode':'flat','jobs':flat_jobs,'threads':flat_threads}
        results.extend(run_wave(tasks,'flat legacy-style pool',flat_jobs,flat_threads))
    elif args.scheduler=='balanced':
        maxres=max(s['resolution'] for s,_ in tasks)
        light=[x for x in tasks if x[0]['resolution']<maxres]
        fine=[x for x in tasks if x[0]['resolution']==maxres]
        light_jobs=max(1,min(jobs,len(light) if light else 1,cap//max(1,threads)))
        light_threads=max(1,min(threads,cap//light_jobs))
        fine_jobs=max(1,min(args.max_fine_concurrent,len(fine),cap))
        fine_threads=(args.fine_threads if args.fine_threads>0 else max(threads,cap//fine_jobs))
        fine_threads=max(1,min(fine_threads,cap//fine_jobs))
        selected_plan={'mode':'balanced','light':{'cases':len(light),'jobs':light_jobs,'threads':light_threads},
                       'fine':{'cases':len(fine),'jobs':fine_jobs,'threads':fine_threads,'max_resolution':maxres}}
        results.extend(run_wave(light,'coarse/medium convergence wave',light_jobs,light_threads))
        results.extend(run_wave(fine,'finest sparse-LU wave',fine_jobs,fine_threads))
    elif args.scheduler=='throughput':
        # Throughput mode is intended for machines like the user's 24-core / >200-GiB node.
        # SuperLU is mostly single-threaded, so TOTAL throughput is higher when independent
        # sparse-LU cases overlap.  Fine cases get a small Numba team for assembly; light
        # cases get one core.  No mathematical case is duplicated.
        maxres=max(s['resolution'] for s,_ in tasks)
        light=[x for x in tasks if x[0]['resolution']<maxres]
        fine=[x for x in tasks if x[0]['resolution']==maxres]
        memgb=_mem_available_gb()
        if math.isfinite(memgb) and memgb<args.throughput_min_memory_gb:
            print(f'[schedule] throughput requested but MemAvailable={memgb:.1f} GiB < {args.throughput_min_memory_gb:g} GiB; falling back to balanced',flush=True)
            light_jobs=max(1,min(jobs,len(light) if light else 1,cap//max(1,threads)))
            light_threads=max(1,min(threads,cap//light_jobs))
            fine_jobs=max(1,min(args.max_fine_concurrent,len(fine),cap))
            fine_threads=max(1,min(args.fine_threads if args.fine_threads>0 else max(threads,cap//fine_jobs),cap//fine_jobs))
            selected_plan={'mode':'throughput-fallback-balanced','MemAvailable_GiB':memgb,
                           'light':{'cases':len(light),'jobs':light_jobs,'threads':light_threads},
                           'fine':{'cases':len(fine),'jobs':fine_jobs,'threads':fine_threads}}
            results.extend(run_wave(light,'fallback coarse/medium wave',light_jobs,light_threads))
            results.extend(run_wave(fine,'fallback finest wave',fine_jobs,fine_threads))
        else:
            # Respect --jobs as a maximum ACTIVE worker-process count. For the full
            # Q3+Q4 validation set on 24 cores, --jobs 10 permits the ideal 6 fine workers
            # plus 3 light workers; the 4th light task queues behind those 3.
            process_budget=min(max(1,args.jobs),len(tasks),cap)
            ft=max(1,args.throughput_fine_threads);lt=max(1,args.throughput_light_threads)
            fine_workers=min(len(fine),process_budget,cap//ft)
            # Keep every fine case resident: LU dominates and is mostly one-core. Give each a
            # small Numba team, then use the remaining CPU slots for a queued light pool.
            used_fine=fine_workers*ft;remaining_cpu=max(0,cap-used_fine);remaining_proc=max(0,process_budget-fine_workers)
            light_workers=min(len(light),remaining_proc,remaining_cpu//lt if lt else 0)
            if light and light_workers==0 and fine_workers>1:
                # Make room for at least one light worker without exceeding the aggregate cap.
                fine_workers-=1;used_fine=fine_workers*ft;remaining_cpu=cap-used_fine;remaining_proc=process_budget-fine_workers
                light_workers=min(len(light),remaining_proc,max(1,remaining_cpu//lt))
            used=used_fine+light_workers*lt
            cpus=list(cpu['selected_cpus'])
            fine_groups=[cpus[i*ft:(i+1)*ft] for i in range(fine_workers)]
            offset=fine_workers*ft
            light_groups=[cpus[offset+i*lt:offset+(i+1)*lt] for i in range(light_workers)]
            fine_now=fine[:] if fine_workers else []
            light_now=light[:] if light_workers else []
            deferred=(fine[:] if not fine_workers else [])+(light[:] if not light_workers else [])
            selected_plan={'mode':'throughput','MemAvailable_GiB':memgb,'used_cpu_slots':used,'process_budget':process_budget,
                           'fine':{'cases':len(fine_now),'workers':fine_workers,'threads':ft,'CPU_groups':fine_groups},
                           'light':{'cases':len(light_now),'workers':light_workers,'threads':lt,'CPU_groups':light_groups},
                           'deferred_cases':len(deferred)}
            print(f'[schedule] throughput: fine={len(fine_now)} cases/{fine_workers} workers x{ft} threads + light={len(light_now)} cases/{light_workers} workers x{lt} threads = {used}/{cap} CPU slots; MemAvailable={memgb:.1f} GiB; deferred={len(deferred)}',flush=True)
            ctx=mp.get_context('spawn');pools=[];pending={};out=[]
            try:
                def launch(wave,name,nworkers,nthreads,groups):
                    if not wave or nworkers<=0:return None
                    counter=ctx.Value('i',0)
                    pool=ProcessPoolExecutor(max_workers=nworkers,mp_context=ctx,initializer=worker_init,initargs=(nthreads,groups,counter))
                    pools.append(pool)
                    role_order={'space':0,'time':1,'end_control':2}
                    for spec,key in sorted(wave,key=lambda v:(role_order.get(v[0].get('role'),9),v[0].get('q',0))):
                        pending[pool.submit(run_case,spec,env,radius,o,key,str(cache))]=spec
                    print(f'[schedule] launched {name}: {len(wave)} tasks on {nworkers} workers; CPU groups={groups}',flush=True)
                    return pool
                launch(fine_now,'fine concurrent pool',fine_workers,ft,fine_groups)
                launch(light_now,'light queued pool',light_workers,lt,light_groups)
                for f in as_completed(pending):out.append(f.result())
            except BaseException:
                for f in pending:f.cancel()
                for pool in pools:
                    for proc in getattr(pool,'_processes',{}).values():
                        try:proc.terminate()
                        except Exception:pass
                    pool.shutdown(wait=True,cancel_futures=True)
                raise
            else:
                for pool in pools:pool.shutdown(wait=True)
            results.extend(out)
            if deferred:
                rem_jobs=min(len(deferred),cap//max(1,threads))
                results.extend(run_wave(deferred,'deferred throughput cases',rem_jobs,max(1,min(threads,cap//max(1,rem_jobs)))))
    else:  # extreme
        maxres=max(s['resolution'] for s,_ in tasks)
        light=[x for x in tasks if x[0]['resolution']<maxres]
        fine_time=[x for x in tasks if x[0]['resolution']==maxres and x[0].get('role')=='time']
        fine_regular=[x for x in tasks if x[0]['resolution']==maxres and x[0].get('role')!='time']
        # The optimized 24-core/Q3+Q4 plan runs all 10 unique cases at once:
        # 2 time-tight x4 cores + 4 other finest x3 + 4 light x1 = 24.
        # When all light cases finish, their 4 cores are donated 2+2 to the two
        # time-tight cases, the long-pole jobs, raising them from 4 to 6 threads.
        if cap==24 and len(fine_time)==2 and len(fine_regular)==4 and len(light)==4 and args.jobs>=10:
            ctx=mp.get_context('spawn');cpus=list(cpu['selected_cpus'])
            time_base=[cpus[0:4],cpus[4:8]]
            regular_groups=[cpus[8:11],cpus[11:14],cpus[14:17],cpus[17:20]]
            light_groups=[[cpus[20]],[cpus[21]],[cpus[22]],[cpus[23]]]
            time_full=[time_base[0]+[cpus[20],cpus[21]],time_base[1]+[cpus[22],cpus[23]]]
            donation=ctx.Event();pools=[];pending={};out=[]
            selected_plan={'mode':'extreme','initial_cpu_slots':24,'all_cases_concurrent':True,
                           'time_tight':{'cases':2,'threads_initial':4,'threads_after_light':6,'groups_initial':time_base,'groups_after_light':time_full},
                           'fine_regular':{'cases':4,'threads':3,'groups':regular_groups},
                           'light':{'cases':4,'threads':1,'groups':light_groups}}
            print('[schedule] EXTREME 24-core plan: 2 time-tight x4 + 4 fine x3 + 4 light x1 = 24/24 cores; all 10 unique cases concurrent.',flush=True)
            print('[schedule] after all 4 light cases finish, their cores are donated to time-tight: 2 cases x6 threads.',flush=True)
            def launch_ext(wave,name,nthreads,groups,event=None,full=None):
                counter=ctx.Value('i',0)
                pool=ProcessPoolExecutor(max_workers=len(groups),mp_context=ctx,initializer=worker_init,initargs=(nthreads,groups,counter,event,full))
                pools.append(pool)
                for sp,key in wave:pending[pool.submit(run_case,sp,env,radius,o,key,str(cache))]=(name,sp)
                print(f'[schedule] launched {name}: {len(wave)} cases; groups={groups}',flush=True)
            launch_ext(fine_time,'time-tight',4,time_base,donation,time_full)
            launch_ext(fine_regular,'fine-regular',3,regular_groups)
            launch_ext(light,'light',1,light_groups)
            light_left=len(light);donated=False
            try:
                for f in as_completed(pending):
                    kind,sp=pending[f];out.append(f.result())
                    if kind=='light':
                        light_left-=1
                        if light_left==0 and not donated:
                            donation.set();donated=True
                            print('[schedule] light wave complete: donating 4 freed cores to the 2 time-tight workers.',flush=True)
            except BaseException:
                for f in pending:f.cancel()
                for pool in pools:
                    for proc in getattr(pool,'_processes',{}).values():
                        try:proc.terminate()
                        except Exception:pass
                    pool.shutdown(wait=True,cancel_futures=True)
                raise
            else:
                for pool in pools:pool.shutdown(wait=True)
            results.extend(out)
        else:
            print('[schedule] extreme topology does not match the 24-core/full-Q3+Q4 layout; using high-memory throughput scheduler.',flush=True)
            # Re-enter the generic throughput policy without recursive main().
            maxres=max(s['resolution'] for s,_ in tasks);light2=[x for x in tasks if x[0]['resolution']<maxres];fine2=[x for x in tasks if x[0]['resolution']==maxres]
            ft=max(1,args.throughput_fine_threads);lt=max(1,args.throughput_light_threads);process_budget=min(max(1,args.jobs),len(tasks),cap)
            fw=min(len(fine2),process_budget,cap//ft);remain=cap-fw*ft;lp=max(0,process_budget-fw);lw=min(len(light2),lp,remain//lt if lt else 0)
            results.extend(run_wave(fine2,'extreme-fallback fine wave',max(1,fw),ft))
            results.extend(run_wave(light2,'extreme-fallback light wave',max(1,lw or 1),max(1,lt)))
    atomic_json(root/'run_metadata.json',{'cpu':cpu,'schedule':selected_plan,'provenance':provenance,'self_tests':tests,'specs':specs,'options':o})
    if digest(args.input)!=provenance['input_sha256'] or (4 in qs and digest(args.radius_input)!=provenance['radius_sha256']):raise RuntimeError('Input changed while computing')
    reports=[export_question(q,results,specs,o,root,env,radius) for q in qs]
    atomic_json(root/'summary.json',{'elapsed_s':time.perf_counter()-started,'questions':reports})
    for r in reports:print(f'Q{r["q"]}: {r["verification"]}; see {root}/Q{r["q"]}/comparison_report.json',flush=True)
    return 0 if all(r['verification']=='OBSERVED_DIFFERENCE_PASS' for r in reports) else 2

KIND='axisymmetric_fem'


def graded_edges(n,beta,layer=.1):
    # Half of elements in the bulk, half in a smoothly graded boundary band.
    nb=n//2;u=np.linspace(0,1,n-nb+1)
    f=u if abs(beta)<1e-12 else -np.expm1(-beta*u)/(-np.expm1(-beta))
    return np.r_[np.linspace(0,1-layer,nb+1),(1-layer+layer*f)[1:]]


def q2_axis(edges):
    out=np.empty(2*len(edges)-1);out[::2]=edges;out[1::2]=.5*(edges[:-1]+edges[1:]);return out


def bernstein_children(b):
    # de Casteljau subdivision at 1/2; the convex hull gives an upper bound.
    def split(a,axis):
        a=np.moveaxis(a,axis,0);mid=(a[0]+2*a[1]+a[2])/4
        l=np.stack((a[0],(a[0]+a[1])/2,mid));r=np.stack((mid,(a[1]+a[2])/2,a[2]))
        return np.moveaxis(l,0,axis),np.moveaxis(r,0,axis)
    a,c=split(b,0)
    return (*split(a,1),*split(c,1))


class AxisymmetricQ2(GalerkinSystem):
    description='2D axisymmetric tensor Q2 continuous finite elements, positive Lobatto quadrature mass; s=(r/R)^2, eta=z/H; adaptive variable-order BDF'
    maximum_method='All Q2 elements: tensor Bernstein convex-hull enclosure with de Casteljau refinement; float64 roundoff allowance'
    def __init__(self,nr,nz,grade_r,grade_z):
        self.nr=nr;self.nz=nz
        self.se=graded_edges(nr,grade_r,.1)**2;self.ze=graded_edges(nz,grade_z,.25)
        self.s=q2_axis(self.se);self.z=q2_axis(self.ze)
        ns=len(self.s);nzv=len(self.z);n=ns*nzv;ne=nr*nz;m=9
        self.conn=np.empty((ne,m),dtype=np.int64)
        self.G0=np.empty((ne,m,m));self.G1=np.empty_like(self.G0)
        self.W0=np.empty((ne,m));self.W1=np.empty_like(self.W0)
        self.mass=np.zeros(n);self.side=np.zeros(n);self.end=np.zeros(n)
        D=np.array([[-3.,4.,-1.],[-1.,0.,1.],[1.,-4.,3.]])
        Gs=np.kron(D,np.eye(3));Gz=np.kron(np.eye(3),D)
        w=np.array([1.,4.,1.])/6;we=np.outer(w,w).ravel()
        sx,zz=np.meshgrid(self.s,self.z,indexing='ij');self.coordinates=np.column_stack((np.sqrt(sx.ravel()),zz.ravel()))
        e=0
        for i in range(nr):
            ds=self.se[i+1]-self.se[i]
            for j in range(nz):
                dz=self.ze[j+1]-self.ze[j]
                ids=((2*i+np.arange(3))[:,None]*nzv+2*j+np.arange(3)[None,:]).ravel()
                self.conn[e]=ids;self.G0[e]=Gs/ds;self.G1[e]=Gz/dz
                ww=we*ds*dz;self.mass[ids]+=ww
                self.W0[e]=ww*sx.ravel()[ids];self.W1[e]=ww;e+=1
        for j in range(nz):self.side[(ns-1)*nzv+2*j+np.arange(3)]+=w*(self.ze[j+1]-self.ze[j])
        for i in range(nr):self.end[(2*i+np.arange(3))*nzv+nzv-1]+=w*(self.se[i+1]-self.se[i])
        px,pz=np.meshgrid(np.linspace(0,1,201),np.linspace(0,1,9),indexing='ij')
        self.probe_coordinates=np.column_stack((px.ravel(),pz.ravel()))
        self.setup();self.probe_matrix=self.interpolation(self.probe_coordinates)
        self.mesh_info={'radial_Q2_elements':nr,'axial_Q2_elements_half_length':nz,'nodes_per_field':n,
                        'full_coupled_DOF':2*n,'radial_grade_beta':grade_r,'axial_grade_beta':grade_z,
                        'radial_outer_band_fraction':.1,'axial_end_band_fraction':.25,'elements_allocated_to_each_band_fraction':.5,
                        'initial_radial_node_gap_um':[float(v) for v in (np.min(np.diff(np.sqrt(self.s)))*.02e6,np.max(np.diff(np.sqrt(self.s)))*.02e6)],
                        'initial_axial_node_gap_um':[float(v) for v in (np.min(np.diff(self.z))*.125e6,np.max(np.diff(self.z))*.125e6)],
                        'domain':'s in [0,1], eta in [0,1] symmetric half-cylinder; Lobatto-integrated Q2, not Q1/FVM'}
        self.B=np.array([[1.,0.,0.],[-.5,2.,-.5],[0.,0.,1.]])

    def interpolation(self,points):
        points=np.asarray(points);s=points[:,0]**2;z=points[:,1]
        ix=np.clip(np.searchsorted(self.se,s,side='right')-1,0,self.nr-1)
        iz=np.clip(np.searchsorted(self.ze,z,side='right')-1,0,self.nz-1)
        u=(s-self.se[ix])/(self.se[ix+1]-self.se[ix]);v=(z-self.ze[iz])/(self.ze[iz+1]-self.ze[iz])
        def shape(x):return np.column_stack(((1-x)*(1-2*x),4*x*(1-x),x*(2*x-1)))
        weights=(shape(u)[:,:,None]*shape(v)[:,None,:]).reshape(-1,9)
        ids=self.conn[ix*self.nz+iz]
        rows=np.repeat(np.arange(len(s)),9)
        return csr_matrix((weights.ravel(),(rows,ids.ravel())),shape=(len(s),self.n))

    def sample(self,points,fields):
        """Direct Q2 interpolation for a small point set; avoids building a CSR matrix every minute."""
        points=np.asarray(points);s=points[:,0]**2;z=points[:,1]
        ix=np.clip(np.searchsorted(self.se,s,side='right')-1,0,self.nr-1)
        iz=np.clip(np.searchsorted(self.ze,z,side='right')-1,0,self.nz-1)
        u=(s-self.se[ix])/(self.se[ix+1]-self.se[ix]);v=(z-self.ze[iz])/(self.ze[iz+1]-self.ze[iz])
        su=np.column_stack(((1-u)*(1-2*u),4*u*(1-u),u*(2*u-1)))
        sv=np.column_stack(((1-v)*(1-2*v),4*v*(1-v),v*(2*v-1)))
        weights=(su[:,:,None]*sv[:,None,:]).reshape(-1,9)
        ids=self.conn[ix*self.nz+iz]
        return np.einsum('ij,ijk->ik',weights,fields[ids],optimize=True)

    def maximum(self,c,tol):
        vals=c[self.conn].reshape(-1,3,3)
        controls=np.einsum('ij,ejk,lk->eil',self.B,vals,self.B,optimize=True)
        lower=float(np.max(c));upper=np.max(controls,axis=(1,2))
        heap=[];counter=0
        for i in np.flatnonzero(upper>lower+tol):
            heapq.heappush(heap,(-float(upper[i]),counter,controls[i]));counter+=1
        while heap:
            ub=-heap[0][0]
            if ub<=lower+tol:break
            _,_,b=heapq.heappop(heap)
            for child in bernstein_children(b):
                lower=max(lower,float(np.max(child[::2,::2])))
                u=float(np.max(child))
                if u>lower+tol:heapq.heappush(heap,(-u,counter,child));counter+=1
            if counter>200000:raise RuntimeError('Q2 maximum enclosure not resolved; cannot certify event')
        ub=max(lower,(-heap[0][0] if heap else lower),min(float(np.max(upper)),lower+tol))
        pad=64*np.finfo(float).eps*(1+float(np.max(abs(controls))))
        return lower,ub+pad


def make_model(spec,o):return AxisymmetricQ2(spec['nr'],spec['nz'],spec['grade_r'],spec['grade_z'])


def parse_args():
    p=common_args(KIND)
    p.set_defaults(jobs=10,threads_per_case=2)
    p.add_argument('--fem-meshes',default='32x48,48x72,64x96',help='Q2 macro-elements on half-cylinder, increasing')
    p.add_argument('--grade-r',type=float,default=7.)
    p.add_argument('--grade-z',type=float,default=7.)
    p.add_argument('--end-boundary',choices=['exposed','insulated'],default='exposed')
    p.add_argument('--no-end-control',action='store_true',help='Otherwise finest insulated/exposed control also runs once')
    return p.parse_args()


def build_specs(qs,o):
    meshes=sorted(set(tuple(map(int,x.lower().split('x'))) for x in o['fem_meshes'].split(',')),key=lambda x:x[0]*x[1])
    if min(min(v) for v in meshes)<2:raise ValueError('Use at least 2x2 Q2 elements')
    if max(o['grade_r'],o['grade_z'])>8 or min(o['grade_r'],o['grade_z'])<0:raise ValueError('Grade parameters in [0,8]')
    specs=[]
    for q in qs:
        for nr,nz in meshes:
            specs.append({'q':q,'tag':f'Q2-{nr}x{nz}-{o["end_boundary"]}','nr':nr,'nz':nz,
                          'grade_r':o['grade_r'],'grade_z':o['grade_z'],'end':o['end_boundary'],
                          'resolution':nr*nz,'role':'space','method':'BDF','rtol':o['rtol'],
                          'atol_T':o['atol_T'],'atol_C':o['atol_C'],'max_step':o['max_step'],
                          'late_max_step':o['late_max_step'],'jacobian_mode':o['jacobian_mode']})
        main=specs[-1]
        if not o['no_time_check']:
            s={**main};s.update(tag=main['tag']+'-time-tight',role='time',rtol=o['rtol']/8,
                              atol_T=o['atol_T']/8,atol_C=o['atol_C']/8,max_step=o['max_step']/2,late_max_step=o['late_max_step']/2);specs.append(s)
        if not o['no_end_control']:
            other='insulated' if o['end_boundary']=='exposed' else 'exposed'
            s={**main};s.update(tag=f'Q2-{main["nr"]}x{main["nz"]}-{other}-control',role='end_control',end=other);specs.append(s)
    return specs


def test_spec():return {'q':4,'nr':3,'nz':4,'grade_r':1.,'grade_z':1.}


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:print('Interrupted; no emergency/SIGINT checkpoint is written.',file=sys.stderr);sys.exit(130)
    except Exception:traceback.print_exc();sys.exit(1)

