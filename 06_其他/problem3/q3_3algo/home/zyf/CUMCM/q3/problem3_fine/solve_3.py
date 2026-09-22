#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUMCM 2026 A, question 3: independent, high-resolution CPU solver.

DEPENDENCIES: Python>=3.10, numpy, numba. No sibling Python files, embedded
measurements, downloads, or spreadsheet template needed. External XLSX only.
A题.pdf and its two A题/附件 XLSX files are the problem data sources.

MODEL (explicit assumptions, NOT a uniqueness/experimental-accuracy claim):
- 1-D radial heat/moisture transport, no end effects, radiation or explicit
  latent-heat term; same effective Robin moisture closure as question 2.
- h=25 W/(m2 K), hm=8e-7 m/s are inherited assumptions from Appendix 2.
- Q3: R=.02 m, Appendix 3 coefficients, solve again FROM t=0.
- Q4: Appendix 4 coefficients FROM t=0, radius from attachment 2, converted
  cm -> m. ASSUME homothetic radial shrinkage of the solid: v=r*Rdot/R,
  xi=r/R(t) is a MATERIAL coordinate. C is dry-basis moisture, NOT water
  mass per volume. No artificial dilution/concentration term for C.
  In this coordinate, with u=T or C, the material transport operator is
  R^(-2)/xi * d_xi(xi*a*d_xi u). The surface condition is
  -a*u_xi/R = transfer*(u_s-u_air). This is not just cropping a fixed grid,
  nor replacing R in the initial boundary and leaving it unchanged.
  The prescribed radius is an effective kinematic closure; rho(C) is used
  in thermal storage. No claim of a fully closed deformable-mixture model.
- Environmental XLSX data are used unchanged with piecewise-linear
  interpolation throughout their support. AFTER that support, by default
  a declared 60 s linear transition reaches 50 C and .05 kg/kg, then those
  values stay constant. This extension is an assumption, NOT measured data.
- Radius: piecewise linear, no smoothing/curve fitting. Default: ERROR if
  drying is not reached before the last measured radius time. Explicit
  --radius-tail hold permits constant extension, recorded in provenance.
- Drying event: max of ALL grid-node dry-basis values crosses .15, not an
  average, not an assumed center value, and not values rounded to 4 decimals.

NUMERICS/OPTIMIZATION:
- nested, node-centered cylindrical finite volumes, 10x refinement over
  xi in [.9,1]; default 12160 intervals (12161 nodes).
- float64, Numba nogil, fastmath=False; paired O(N) positive-pivot Thomas.
- two backward-Euler half steps at startup; variable-step BDF2 thereafter.
  First-minute dt=1/128 s; gradual doubling to warm_dt=1/8 s for the
  measured-input period, and then to late_dt=1 s after its transition.
  Increases happen at most once per 60 simulated seconds.
  A step change uses the ACTUAL BDF2 step ratio, not constant-step weights.
  The two larger-step verification cases multiply ALL stage step sizes by 2/4.
- Picard: updates AND recomputed final-state local algebraic residuals must
  pass every implicit step; no clipping, unconverged acceptance, or fastmath.
- hoisted geometry, forcing interpolation, history terms and old gradients;
  preallocated buffers; a single shared face flux; shared material ratio and
  combined exponential; no invalid reuse of changing matrix factorizations.
- 5 unique coupled cases: main, two spatial and two temporal comparisons.
  At most 24 CPUs (also affinity/cgroup aware); no fake work to fill cores.
- substep drying-event bracket from quadratic BDF2 dense reconstruction;
  concentration convergence AND drying-time convergence checked separately.
  Event bisection precision is NOT global model/time-discretization accuracy.
- result3.xlsx follows the template's single Sheet1. Real distance
  columns every .1 cm and time every 60 s, plus an unrounded-time final row.
  Q4 has an additional surface column; positions outside R(t) are BLANK,
  never zero, repeated surface values, extrapolation, or normalized radii.
  Details/full-precision time and fields/profiles are exported separately.
- safe atomic caches/checkpoints keyed by code, data and settings. Each
  independent script can restart without importing question 1/2 or its peer.

Validation is conditional on these model/forcing/radius closures. A numerical
convergence pass is not a rigorous error bound or a physical validation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, platform, posixpath, sys
import threading, time, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, BadZipFile
from xml.sax.saxutils import escape
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
              "BLIS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
    os.environ[_name]="1"
try:
    import numpy as np
    import numba
    from numba import njit
except ImportError as exc:
    raise SystemExit("Install dependencies: python3 -m pip install numpy numba\n"+str(exc)) from exc
PROBLEM = 3
VERSION = "3.0.0"
DEFAULT_INPUT = Path('/home/zyf/CUMCM/problem/attachment/附件1.xlsx')
DEFAULT_RADIUS = Path('/home/zyf/CUMCM/problem/attachment/附件2.xlsx')
NS="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG="http://schemas.openxmlformats.org/package/2006/relationships"
XML_DECL='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
PRINT_LOCK=threading.Lock()


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        with PRINT_LOCK:
            print(message, flush=True)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def cpu_limit(requested: int) -> tuple[int, dict]:
    """Logical CPU cap; honors affinity and Linux cgroup CPU quotas."""
    if not 1 <= requested <= 24:
        raise ValueError("--workers must be between 1 and 24 (hard limit).")
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = list(range(os.cpu_count() or 1))
    quotas: list[float] = []
    # cgroup v2/v1 mounted roots, plus process-relative paths where exposed.
    candidates = {Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/cpu"),
                  Path("/sys/fs/cgroup/cpu,cpuacct")}
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if not controllers or "cpu" in controllers.split(","):
                for root in tuple(candidates):
                    if ".." not in Path(relative).parts:
                        path = root / relative.lstrip("/")
                        for ancestor in (path, *path.parents):
                            if ancestor == root or root in ancestor.parents:
                                candidates.add(ancestor)
    except (OSError, ValueError):
        pass
    for root in candidates:
        try:
            raw = (root / "cpu.max").read_text().split()
            if raw[0] != "max" and float(raw[0]) > 0 and float(raw[1]) > 0:
                quotas.append(float(raw[0]) / float(raw[1]))
        except (OSError, ValueError, IndexError):
            pass
        try:
            q = float((root / "cpu.cfs_quota_us").read_text())
            p = float((root / "cpu.cfs_period_us").read_text())
            if q > 0 and p > 0:
                quotas.append(q / p)
        except (OSError, ValueError):
            pass
    quota = min(quotas) if quotas else None
    available = min(len(allowed), max(1, math.ceil(quota))) if quota else len(allowed)
    cap = min(24, requested, max(1, available))
    # Linux hard affinity cap applies to subsequently created solver threads.
    # It does not expand affinity or touch any other process's affinity.
    # Prefer separate physical cores to sibling SMT threads when topology exists.
    first, siblings, seen = [], [], set()
    for cpu_id in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
        try:
            physical = ((base/"physical_package_id").read_text().strip(),
                        (base/"core_id").read_text().strip())
        except OSError:
            physical = ("unknown", str(cpu_id))
        if physical in seen:
            siblings.append(cpu_id)
        else:
            seen.add(physical)
            first.append(cpu_id)
    selected = (first+siblings)[:cap]
    pinned = False
    try:
        os.sched_setaffinity(0, selected)
        pinned = True
    except (AttributeError, OSError):
        pass
    return cap, {"requested_max": requested, "hard_max": 24,
                 "affinity_cpu_count_before_cap": len(allowed),
                 "cgroup_cpu_quota": quota, "effective_cpu_cap": cap,
                 "affinity_cap_applied": pinned, "selected_cpu_ids": selected,
                 "available_distinct_physical_cores": len(first), "nested_library_threads": 1}

def column_number(ref: str) -> int:
    value = 0
    for ch in ref:
        if not ch.isalpha():
            break
        value = 26 * value + ord(ch.upper()) - 64
    if value < 1:
        raise ValueError(f"Invalid XLSX cell reference: {ref}")
    return value - 1

def read_xlsx_rows(path: Path) -> list[tuple[int, list]]:
    """Read first worksheet values, handling both shared and inline strings."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}; no embedded-data fallback.")
    with ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt input XLSX archive.")
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.iter(f"{{{NS}}}t"))
                       for si in root.findall(f"{{{NS}}}si")]
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        sheet = wb.find(f"{{{NS}}}sheets/{{{NS}}}sheet")
        if sheet is None:
            raise ValueError("Input workbook has no worksheet.")
        rid = sheet.attrib[f"{{{REL}}}id"]
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relation = next((x for x in rels if x.get("Id") == rid), None)
        if relation is None or relation.get("TargetMode") == "External":
            raise ValueError("Invalid/external input worksheet relationship.")
        target = relation.attrib["Target"]
        member = (posixpath.normpath(target.lstrip("/")) if target.startswith("/")
                  else posixpath.normpath(posixpath.join("xl", target)))
        if not member.startswith("xl/"):
            raise ValueError("Unsafe XLSX worksheet path.")
        root = ET.fromstring(z.read(member))
        rows = []
        for row in root.findall(f"{{{NS}}}sheetData/{{{NS}}}row"):
            values = {}
            for cell in row.findall(f"{{{NS}}}c"):
                col = column_number(cell.attrib["r"])
                if col > 2:  # only the three specified environmental columns
                    continue
                if cell.find(f"{{{NS}}}f") is not None:
                    raise ValueError(f"Formula in {cell.attrib['r']}: input must contain values, not stale formula caches.")
                t = cell.get("t", "n")
                v = cell.find(f"{{{NS}}}v")
                if t == "inlineStr":
                    value = "".join(x.text or "" for x in cell.iter(f"{{{NS}}}t"))
                elif v is None or v.text is None:
                    value = None
                elif t == "s":
                    value = strings[int(v.text)]
                elif t in ("str", "e", "b"):
                    value = v.text  # headers handled below; errors/bools rejected
                    if t in ("e", "b"):
                        raise ValueError(f"Invalid value type {t} at {cell.attrib['r']}")
                else:
                    value = float(v.text)
                values[col] = value
            items = [values.get(i) for i in range(3)]
            if any(v not in (None, "") for v in items):
                rows.append((int(row.attrib["r"]), items))
        return rows

def load_environment(path: Path, end_time: int) -> np.ndarray:
    rows = read_xlsx_rows(path)
    expected = ["时间", "温度", "水分浓度"]
    if not rows or [str(v).strip() for v in rows[0][1]] != expected:
        raise ValueError("附件1 first-row headers must be 时间、温度、水分浓度.")
    data = []
    last_row = rows[0][0]
    for row_no, row in rows[1:]:
        if row_no != last_row + 1:
            raise ValueError(f"Blank/missing input row before Excel row {row_no}.")
        last_row = row_no
        try:
            values = [float(x) for x in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric or missing input at Excel row {row_no}.") from exc
        if not all(math.isfinite(x) for x in values) or values[2] <= 0 or values[1] <= -273.15:
            raise ValueError(f"Non-finite input or nonpositive moisture at row {row_no}.")
        data.append(values)
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) < 2 or not np.all(np.diff(arr[:, 0]) > 0):
        raise ValueError("Input must contain >=2 strictly increasing, unique time points.")
    if arr[0, 0] > 0 or arr[-1, 0] < end_time:
        raise ValueError(f"Input does not cover [0, {end_time}] s; extrapolation is forbidden.")
    return arr

def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

@njit(cache=True, nogil=True, fastmath=False, inline="always")
def paired_thomas(mass, cap, gT, gC, histT, histC, fluxT, fluxC,
                  bT, bC, srcT, srcC, ratT, ratC, canT, canC):
    """Two symmetric diffusion M-matrices, solved without cancelling pivots."""
    n = mass.size
    sT, sC = mass[0]*cap[0], mass[0]
    invT, invC = 1./(sT+gT[0]), 1./(sC+gC[0])
    ratT[0], ratC[0] = gT[0]*invT, gC[0]*invC
    canT[0] = (histT[0]*cap[0]+fluxT[0])*invT
    canC[0] = (histC[0]+fluxC[0])*invC
    for i in range(1,n-1):
        sT = mass[i]*cap[i]+gT[i-1]*(sT*invT)
        sC = mass[i]+gC[i-1]*(sC*invC)
        invT, invC = 1./(sT+gT[i]), 1./(sC+gC[i])
        ratT[i], ratC[i] = gT[i]*invT, gC[i]*invC
        canT[i] = (histT[i]*cap[i]+fluxT[i]-fluxT[i-1]+gT[i-1]*canT[i-1])*invT
        canC[i] = (histC[i]+fluxC[i]-fluxC[i-1]+gC[i-1]*canC[i-1])*invC
    sT = mass[-1]*cap[-1]+bT+gT[n-2]*(sT*invT)
    sC = mass[-1]+bC+gC[n-2]*(sC*invC)
    if sT <= 0 or sC <= 0 or not math.isfinite(sT+sC):
        raise ArithmeticError("Nonpositive/nonfinite diffusion pivot.")
    canT[-1] = (histT[-1]*cap[-1]-fluxT[n-2]+srcT+gT[n-2]*canT[n-2])/sT
    canC[-1] = (histC[-1]-fluxC[n-2]+srcC+gC[n-2]*canC[n-2])/sC
    for i in range(n-2,-1,-1):
        canT[i] += ratT[i]*canT[i+1]
        canC[i] += ratC[i]*canC[i+1]

def arrays_digest(arrays: dict) -> str:
    h = hashlib.sha256()
    for name,arr in sorted(arrays.items()):
        a = np.ascontiguousarray(arr)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        if a.nbytes:
            h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def atomic_npz(path: Path, key: str, arrays: dict, info: dict) -> None:
    """Atomic and non-executable cache format; no pickle or object arrays."""
    tmp=path.with_name(path.name+f".{os.getpid()}.{threading.get_ident()}.tmp")
    metadata={**info,"arrays_sha256":arrays_digest(arrays)}
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f,cache_key=np.array(key),
                                metadata=np.array(json.dumps(metadata,ensure_ascii=False,allow_nan=False)),
                                **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():
            tmp.unlink()

def atomic_csv(path: Path, header: list, rows) -> None:
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temp.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)
        os.replace(temp,path)
    finally:
        if temp.exists():
            temp.unlink()

@dataclass(frozen=True)
class Config:
    radial_cells: int = 12160
    dt: float = .0078125
    late_dt: float = 1.
    temperature_tol: float = 2e-12
    moisture_tol: float = 2e-13
    residual_rtol: float = 1e-10
    picard_max: int = 40
    event_tol: float = 1e-4
    warm_dt: float = .125

@dataclass
class Mesh:
    xi: np.ndarray
    w: np.ndarray
    g: np.ndarray
    probes: np.ndarray

# Used for independent spatial comparisons, not only the sparse submission grid.
PROBE_XI = np.r_[np.arange(181)*.005, .9+np.arange(1,101)*.001]
OUTPUT_CM = np.arange(21 if PROBLEM==3 else 20,dtype=np.float64)/10.

def make_mesh(n: int) -> Mesh:
    if n < 380 or n % 380:
        raise ValueError('radial-cells must be a positive multiple of 380, >=380.')
    nb=n*9//19
    dx=.9/nb
    x=np.r_[np.arange(nb+1)*dx,.9+np.arange(1,n-nb+1)*(dx/10)]
    x[0]=0.; x[-1]=1.
    faces=(x[:-1]+x[1:])*.5
    bounds=np.r_[0.,faces,1.]
    w=.5*np.diff(bounds)*(bounds[1:]+bounds[:-1])
    ind=np.searchsorted(x,PROBE_XI)
    ind=np.minimum(ind,n)
    left=np.maximum(ind-1,0)
    ind=np.where(abs(x[left]-PROBE_XI)<abs(x[ind]-PROBE_XI),left,ind).astype(np.int64)
    if np.max(abs(x[ind]-PROBE_XI))>1e-12 or abs(np.sum(w)-.5)>1e-14:
        raise AssertionError('Bad grid volumes or nonaligned convergence probes.')
    return Mesh(np.ascontiguousarray(x),np.ascontiguousarray(w),
                np.ascontiguousarray(faces/np.diff(x)),np.ascontiguousarray(ind))

def load_radius(path: Path) -> np.ndarray:
    rows=read_xlsx_rows(path)
    if not rows or [str(v).strip() for v in rows[0][1][:2]]!=['时间','半径']:
        raise ValueError('附件2 must have first-row headers 时间、半径.')
    out=[]; last=rows[0][0]
    for row_no, row in rows[1:]:
        if row_no!=last+1: raise ValueError('Blank/missing radius row.')
        last=row_no
        try: t,r=map(float,row[:2])
        except (TypeError,ValueError) as exc: raise ValueError(f'Invalid radius at Excel row {row_no}.') from exc
        if not math.isfinite(t+r) or r<=0: raise ValueError('Invalid/nonpositive radius.')
        out.append((t,r/100.))
    a=np.asarray(out,dtype=np.float64)
    if len(a)<2 or a[0,0]!=0 or not np.all(np.diff(a[:,0])>0):
        raise ValueError('Radius time must start at 0 and increase strictly.')
    if abs(a[0,1]-.02)>1e-12: raise ValueError('Initial radius does not equal the stated 2 cm.')
    if np.max(np.diff(a[:,1]))>1e-12:
        raise ValueError('Radius data increase: this shrinkage model will not silently smooth/repair them.')
    return a

class Controls:
    """One external forcing history, shared by all spatial jobs."""
    def __init__(self, env, radius, airT, airC, ramp):
        self.raw_env=env
        self.radius=radius
        # Do not alter any measured row. The extra row is a DECLARED extension.
        if ramp<=0 or ramp%60:
            raise ValueError('--tail-ramp must be a positive multiple of 60 s.')
        self.env=np.vstack((env,[env[-1,0]+ramp,airT,airC]))
        self.transition_end=float(self.env[-1,0])
        if self.transition_end%60: raise ValueError('This precision schedule requires the boundary tail to end on a 60 s boundary.')
    def at(self, times):
        t=np.asarray(times,dtype=np.float64)
        out=np.empty(t.shape+(3,),dtype=np.float64)
        out[...,0]=np.interp(t,self.env[:,0],self.env[:,1])
        out[...,1]=np.interp(t,self.env[:,0],self.env[:,2])
        if PROBLEM==4:
            out[...,2]=np.interp(t,self.radius[:,0],self.radius[:,1])
        else: out[...,2]=.02
        return out

def phases(c: Config, switch: float, end: int) -> list[tuple]:
    """Initial minute, measured-input period, long-time stage. Every increase
    is by 2 at most. All stages/output times match exactly across refinements.
    """
    if not (0<c.dt<=c.warm_dt<=c.late_dt<=15):
        raise ValueError('Require 0<dt<=warm-dt<=late-dt<=15 s.')
    for d in (c.dt,c.warm_dt,c.late_dt):
        if abs(60/d-round(60/d))>1e-8: raise ValueError('Each step size must divide 60 s.')
    for ratio in (c.warm_dt/c.dt,c.late_dt/c.warm_dt):
        power=int(round(math.log2(ratio)))
        if abs(ratio-2**power)>1e-9: raise ValueError('Successive stage step ratios must be powers of two.')
    out=[];t=0.;h=c.dt
    first=min(60.,float(end))
    out.append((0.,first,h));t=first
    for until,target in ((min(float(end),switch),c.warm_dt),(float(end),c.late_dt)):
        while t<until:
            h=min(2*h,target)
            stop=min(t+60.,until) if h<target else until
            # Merge adjacent equal-step phases; avoid duplicated forcing nodes.
            if out and out[-1][2]==h:
                old=out.pop();out.append((old[0],stop,h))
            else:out.append((t,stop,h))
            t=stop
    return out

def prepare_timeline(c: Config, ctrl: Controls, end: int) -> list[dict]:
    """Computed ONCE per distinct time schedule, reused by N/4,N/2,N.
    Approximately 24 bytes per time node, not a full space-time solution.
    """
    result=[]
    for begin,stop,h in phases(c,ctrl.transition_end,end):
        n=int(round((stop-begin)/h))
        times=begin+np.arange(n+1,dtype=np.float64)*h
        values=ctrl.at(times)
        values.flags.writeable=False
        result.append({'begin':begin,'stop':stop,'dt':h,'values':values})
    return result

@njit(cache=True,nogil=True,fastmath=False)
def coefficients(T,C,dT,dC,cap,k,D):
    for i in range(T.size):
        c=C[i]+dC[i]; kelvin=T[i]+dT[i]+273.15
        if c<=0 or kelvin<=0 or not math.isfinite(c+kelvin):
            raise ArithmeticError('Invalid nonlinear iterate; no clipping or fallback.')
        z=c/(1.+c)
        if PROBLEM==3:
            cap[i]=(650.+128.*c)*(1450.+2736.*z)
            k[i]=.21+.38*z
            D[i]=.0024*math.exp(-.45/c-3850./kelvin)
        else:
            cap[i]=(760.+90.*c)*(1850.+2150.*z)
            k[i]=.12+.20*z
            D[i]=.00042*math.exp(-.30/c-3850./kelvin)
        if D[i]<=0 or not math.isfinite(D[i]):
            raise ArithmeticError('Invalid/underflow diffusivity.')

@njit(cache=True,nogil=True,fastmath=False)
def advance(T,C,prevT,prevC,w,geom,R,airT,airC,h,ratio,
            ht,hm,tolT,tolC,rtol,maxit,scratch,diag):
    """One material-coordinate BE (ratio=0) or variable-step BDF2 step.
    alpha*d_new - beta*d_previous = h*f(new),
    alpha=(1+2q)/(1+q), beta=q*q/(1+q), q=h/h_previous.
    R^2 multiplies the material derivative, NOT the differentiated state.
    """
    n=T.size
    incT,incC,histT,histC,gradT,gradC,cap,k,D,gT,gC,fT,fC,ratT,ratC,canT,canC,mass=scratch
    alpha=(1.+2.*ratio)/(1.+ratio)
    beta=ratio*ratio/(1.+ratio)
    mass_factor=alpha*R*R/h; hist_factor=beta*R*R/h
    for i in range(n):
        mass[i]=w[i]*mass_factor
        histT[i]=w[i]*hist_factor*prevT[i]
        histC[i]=w[i]*hist_factor*prevC[i]
        incT[i]=ratio*prevT[i]; incC[i]=ratio*prevC[i]
        if C[i]+incC[i]<=0 or T[i]+incT[i]<=-273.15:
            incT[i]=0.; incC[i]=0.
    for i in range(n-1):
        gradT[i]=T[i+1]-T[i]; gradC[i]=C[i+1]-C[i]
    bT=ht*R; bC=hm*R
    srcT=bT*(airT-T[-1]); srcC=bC*(airC-C[-1])
    converged=False
    updT=0.;updC=0.; localT=0.;localC=0.;ratmaxT=0.;ratmaxC=0.
    for iteration in range(1,maxit+1):
        coefficients(T,C,incT,incC,cap,k,D)
        for i in range(n-1):
            gT[i]=geom[i]*(2*k[i]*k[i+1]/(k[i]+k[i+1]))
            gC[i]=geom[i]*(2*D[i]*D[i+1]/(D[i]+D[i+1]))
            fT[i]=gT[i]*gradT[i]; fC[i]=gC[i]*gradC[i]
        paired_thomas(mass,cap,gT,gC,histT,histC,fT,fC,bT,bC,srcT,srcC,
                      ratT,ratC,canT,canC)
        updT=0.;updC=0.
        for i in range(n):
            updT=max(updT,abs(canT[i]-incT[i]))
            updC=max(updC,abs(canC[i]-incC[i]))
            incT[i]=canT[i]; incC[i]=canC[i]
        if updT>tolT or updC>tolC: continue
        # An increment-only stopping test is insufficient: re-evaluate the
        # actual nonlinear coefficients/fluxes at the candidate final state.
        coefficients(T,C,incT,incC,cap,k,D)
        for i in range(n-1):
            gT[i]=geom[i]*(2*k[i]*k[i+1]/(k[i]+k[i+1]))
            gC[i]=geom[i]*(2*D[i]*D[i+1]/(D[i]+D[i+1]))
            fT[i]=gT[i]*(gradT[i]+incT[i+1]-incT[i])
            fC[i]=gC[i]*(gradC[i]+incC[i+1]-incC[i])
        boundaryT=srcT-bT*incT[-1]; boundaryC=srcC-bC*incC[-1]
        passed=True;localT=0.;localC=0.;ratmaxT=0.;ratmaxC=0.
        for i in range(n):
            st=cap[i]*(mass[i]*incT[i]-histT[i]); sc=mass[i]*incC[i]-histC[i]
            flt=fT[i-1] if i else 0.; flc=fC[i-1] if i else 0.
            frt=fT[i] if i<n-1 else boundaryT; frc=fC[i] if i<n-1 else boundaryC
            rt=abs(st-frt+flt); rc=abs(sc-frc+flc)
            at=mass[i]*cap[i]+(gT[i-1] if i else 0.)+(gT[i] if i<n-1 else bT)
            ac=mass[i]+(gC[i-1] if i else 0.)+(gC[i] if i<n-1 else bC)
            # Componentwise algebraic residual with an EXPLICIT float64
            # roundoff allowance, not a mesh-independent magic absolute limit.
            limT=rtol*(abs(st)+abs(flt)+abs(frt))+64.*2.220446049250313e-16*at*(1.+abs(T[i])+abs(incT[i]))
            limC=rtol*(abs(sc)+abs(flc)+abs(frc))+64.*2.220446049250313e-16*ac*(1.+abs(C[i])+abs(incC[i]))
            localT=max(localT,rt);localC=max(localC,rc)
            ratmaxT=max(ratmaxT,rt/max(limT,1e-300));ratmaxC=max(ratmaxC,rc/max(limC,1e-300))
            if rt>limT or rc>limC: passed=False
        if passed:
            converged=True; break
        diag[20]+=1.
    if not converged:
        raise RuntimeError('Picard updates/final-state local residual did not converge; refusing result.')
    boundaryT=srcT-bT*incT[-1]; boundaryC=srcC-bC*incC[-1]
    sumT=0.;sumC=0.;corrT=0.;corrC=0.; maxC=-1e300
    for i in range(n):
        nt=T[i]+incT[i]; nc=C[i]+incC[i]
        if nc<=0 or nt<=-273.15 or not math.isfinite(nt+nc):
            raise ArithmeticError('Invalid accepted state.')
        yt=cap[i]*(mass[i]*incT[i]-histT[i])-corrT
        yc=(mass[i]*incC[i]-histC[i])-corrC
        zt=sumT+yt; zc=sumC+yc
        corrT=(zt-sumT)-yt;corrC=(zc-sumC)-yc
        sumT=zt;sumC=zc
        T[i]=nt;C[i]=nc;prevT[i]=incT[i];prevC[i]=incC[i]
        maxC=max(maxC,nc)
        diag[10]=min(diag[10],nt);diag[11]=max(diag[11],nt)
        diag[12]=min(diag[12],nc);diag[13]=max(diag[13],nc)
    diag[0]+=1.;diag[1]+=iteration;diag[2]=max(diag[2],iteration)
    diag[3]=max(diag[3],updT);diag[4]=max(diag[4],updC)
    eT=abs(sumT-boundaryT);eC=abs(sumC-boundaryC)
    diag[5]=max(diag[5],eT);diag[6]=max(diag[6],eC)
    diag[7]=max(diag[7],eT/max(abs(boundaryT),abs(sumT),1e-30))
    diag[8]=max(diag[8],eC/max(abs(boundaryC),abs(sumC),1e-30))
    diag[9]=max(diag[9],abs(boundaryT));diag[19]=max(diag[19],abs(boundaryC))
    diag[14]=max(diag[14],localT);diag[15]=max(diag[15],localC)
    diag[16]=max(diag[16],ratmaxT);diag[17]=max(diag[17],ratmaxC);diag[18]+=1.
    diag[21]=maxC
    return maxC

@njit(cache=True,nogil=True,fastmath=False)
def integrate_block(first,last,phase_start,dt,forcing,state,prev,w,geom,
                    tolT,tolC,rtol,maxit,scratch,diag,initial_prev_dt,
                    threshold,ht,hm):
    """This block never crosses an output (60 s) boundary. Dependent time
    steps are serial; independent refinement jobs are the parallel unit.
    """
    for j in range(first,last+1):
        R=forcing[j,2]; airT=forcing[j,0]; airC=forcing[j,1]
        oldh=initial_prev_dt if j==first else dt
        if phase_start==0. and j==1:
            mid=(forcing[0]+forcing[1])*.5
            advance(state[0],state[1],prev[0],prev[1],w,geom,mid[2],mid[0],mid[1],
                    dt*.5,0.,ht,hm,tolT,tolC,rtol,maxit,scratch,diag)
            half=prev.copy()  # Exactly once, startup only.
            maxC=advance(state[0],state[1],prev[0],prev[1],w,geom,R,airT,airC,
                         dt*.5,0.,ht,hm,tolT,tolC,rtol,maxit,scratch,diag)
            for i in range(state.shape[1]):
                prev[0,i]+=half[0,i];prev[1,i]+=half[1,i]
        else:
            maxC=advance(state[0],state[1],prev[0],prev[1],w,geom,R,airT,airC,
                         dt,dt/oldh,ht,hm,tolT,tolC,rtol,maxit,scratch,diag)
        if maxC<threshold:
            return j,True,oldh
    return last,False,dt

def event_bracket(state,prev,scratch,w,t_right,h,hprev,R,threshold,tol):
    """Continuous piecewise-quadratic temporal reconstruction at ALL nodes.
    Bisection finds a strictly-below upper bracket, without resolving the PDE
    from t=0 for every event trial. This is a discretization-dependent root.
    """
    q=h/hprev; beta=q*q/(1+q)
    if beta<=0: raise ValueError('Event before a usable BDF2 history exists.')
    old_delta=np.stack((scratch[2],scratch[3]))/(w*(beta*R*R/h))
    base=state-prev
    a=(hprev*prev/h+h*old_delta/hprev)/(h+hprev)
    b=(prev/h-old_delta/hprev)/(h+hprev)
    def value(s): return base+s*a+s*s*b
    lo=0.;hi=h
    if np.max(base[1])<threshold or np.max(state[1])>=threshold:
        raise ArithmeticError('Invalid drying-event endpoint signs.')
    while hi-lo>tol:
        mid=.5*(lo+hi)
        if np.max(value(mid)[1])<threshold: hi=mid
        else: lo=mid
    final=value(hi)
    if np.max(final[1])>=threshold: raise ArithmeticError('Upper event endpoint is not strictly dry.')
    idx=int(np.argmax(final[1])); t0=t_right-h
    return final,{'lower_s':float(t0+lo),'upper_s':float(t0+hi),
                  'midpoint_s':float(t0+.5*(lo+hi)),
                  'bracket_width_s':float(hi-lo),'last_full_step_s':float(h),
                  'upper_max_C':float(np.max(final[1])),
                  'lower_max_C':float(np.max(value(lo)[1])),
                  'controlling_node':idx,
                  'dCdt_at_controlling_node':float(a[1,idx]+2*hi*b[1,idx]),
                  'method':'quadratic three-level all-node maximum; strict upper bracket',
                  'limitation':'event bracket is not a bound on PDE discretization/model error'}

def sample_physical(state,mesh,R):
    x=OUTPUT_CM/(R*100.)
    valid=x<=1.+1e-12
    out=np.full((2,len(x)+(1 if PROBLEM==4 else 0)),np.nan)
    for f in range(2):
        out[f,:len(x)][valid]=np.interp(np.minimum(x[valid],1.),mesh.xi,state[f])
        if PROBLEM==4: out[f,-1]=state[f,-1]
    return out

def read_npz_safe(path: Path,key: str):
    with np.load(path,allow_pickle=False) as f:
        if str(f['cache_key'].item())!=key: raise ValueError('Cache key mismatch.')
        meta=json.loads(str(f['metadata'].item()))
        arrays={k:f[k] for k in f.files if k not in ('cache_key','metadata')}
    if arrays_digest(arrays)!=meta['arrays_sha256']: raise ValueError('Cache checksum mismatch.')
    # NaN is intentional ONLY in out-of-material physical-distance columns.
    for k,a in arrays.items():
        if a.dtype.kind not in 'fiu': raise ValueError('Non-numeric cache array.')
        if k=='physical':
            if np.any(np.isinf(a)): raise ValueError('Infinite physical output.')
        elif not np.all(np.isfinite(a)): raise ValueError('Non-finite cache data.')
    return arrays,meta

def diagnose(diag):
    return {'implicit_steps':int(diag[0]),'picard_total':int(diag[1]),
            'picard_mean':float(diag[1]/max(diag[0],1)), 'picard_max':int(diag[2]),
            'max_final_update_T':float(diag[3]),'max_final_update_C':float(diag[4]),
            'max_global_heat_PDE_balance_abs':float(diag[5]),
            'max_global_moisture_PDE_balance_abs':float(diag[6]),
            'max_global_heat_PDE_balance_rel':float(diag[7]),
            'max_global_moisture_PDE_balance_rel':float(diag[8]),
            'temperature_range':[float(diag[10]),float(diag[11])],
            'moisture_range':[float(diag[12]),float(diag[13])],
            'max_local_heat_residual_abs':float(diag[14]),
            'max_local_moisture_residual_abs':float(diag[15]),
            'max_local_heat_residual_to_allowed':float(diag[16]),
            'max_local_moisture_residual_to_allowed':float(diag[17]),
            'final_state_local_residual_audits':int(diag[18]),
            'candidate_residual_rejections':int(diag[20]),
            'all_node_max_C_at_last_full_step':float(diag[21]),
            'residual_roundoff_allowance':'64*eps*row_diagonal*(1+abs(old_state)+abs(increment))',
            'balance_meaning':'chosen material-coordinate PDE; not full thermodynamic energy audit'}

def run_case(c,mesh,timeline,ctrl,end,args,key,cache_dir,cancel):
    label=f'Q{PROBLEM} N={c.radial_cells} dt={c.dt:g}/{c.warm_dt:g}/{c.late_dt:g}'
    finished_path=cache_dir/(key+'.npz'); partial=cache_dir/(key+'.partial.npz')
    if finished_path.is_file() and not args.no_cache:
        try:
            arrays,info=read_npz_safe(finished_path,key)
            if info['config']!=asdict(c) or arrays['state'].shape!=(2,c.radial_cells+1):
                raise ValueError('Cached configuration/shape mismatch.')
            info['cache_hit']=True
            log(f'[cache] {label}: dried={info["dried"]}',args.quiet)
            return arrays,info
        except (OSError,ValueError,KeyError,BadZipFile,EOFError) as exc:
            log(f'[warning] {label}: invalid completed cache: {exc}',args.quiet)
    n=c.radial_cells+1; rows=end//60
    state=np.empty((2,n));state[0].fill(28.);state[1].fill(2.55)
    prev=np.zeros((2,n));scratch=np.empty((18,n))
    diag=np.zeros(22);diag[10:14]=[28.,28.,2.55,2.55];diag[21]=2.55
    probe=np.empty((rows,2,len(mesh.probes)))
    phy=np.empty((rows,2,len(OUTPUT_CM)+(1 if PROBLEM==4 else 0)))
    # time, R(m), min C, max C, dry-mass-weighted mean, max index
    stats=np.empty((rows,6))
    snapshots=[];snapshot_times=[];done=0.;prior=0.;prev_dt=c.dt
    resumed_from=0.; started=time.perf_counter(); event=None
    if partial.is_file() and not args.no_cache:
        try:
            saved,meta=read_npz_safe(partial,key)
            d=float(meta['done_s']);count=int(round(d/60))
            if not 0<d<end or d%60 or saved['state'].shape!=(2,n) or saved['probes'].shape!=(count,2,len(mesh.probes)):
                raise ValueError('Partial checkpoint shape/time mismatch.')
            state[:]=saved['state'];prev[:]=saved['prev'];diag[:]=saved['diag']
            probe[:count]=saved['probes'];phy[:count]=saved['physical'];stats[:count]=saved['stats']
            snapshot_times=list(saved['snapshot_times']);snapshots=list(saved['snapshots'])
            done=resumed_from=d;prev_dt=float(meta['prev_dt']);prior=float(meta['runtime_s'])
            log(f'[resume] {label}: {done/3600:.3f} h',args.quiet)
        except (OSError,ValueError,KeyError,BadZipFile,EOFError) as exc:
            raise RuntimeError(f'Invalid checkpoint; remove only {partial} and rerun: {exc}') from exc
    def pack(count, final_state=None):
        snaps=np.asarray(snapshots,dtype=np.float64) if snapshots else np.empty((0,2,n))
        return {'state':state.copy() if final_state is None else final_state.copy(),
                'prev':prev.copy(),'diag':diag.copy(),'probes':probe[:count].copy(),
                'physical':phy[:count].copy(),'stats':stats[:count].copy(),
                'snapshot_times':np.asarray(snapshot_times,dtype=np.float64),'snapshots':snaps,
                'xi':mesh.xi,'probe_xi':mesh.xi[mesh.probes]}
    count=int(round(done/60));crossed=False
    for phase in timeline:
        if phase['stop']<=done: continue
        h=phase['dt']; begin=phase['begin']; forc=phase['values']
        start=max(done,begin)
        # Python overhead/I/O only once per output minute; native inner loops.
        for stop in np.arange(start+60,phase['stop']+.5,60):
            if cancel.is_set(): raise InterruptedError('Cancelled; checkpoints retained.')
            first=int(round((done-begin)/h))+1;last=int(round((stop-begin)/h))
            try:
                used,crossed,oldh=integrate_block(first,last,begin,h,forc,state,prev,mesh.w,mesh.g,
                              c.temperature_tol,c.moisture_tol,c.residual_rtol,c.picard_max,
                              scratch,diag,prev_dt,args.threshold,args.heat_transfer,args.mass_transfer)
            except Exception as exc:
                raise RuntimeError(f'{label}, interval ({done},{stop}] s: {exc}') from exc
            actual=begin+used*h; R=float(forc[used,2]);prev_dt=h
            if crossed:
                final_state,event=event_bracket(state,prev,scratch,mesh.w,actual,h,oldh,R,args.threshold,c.event_tol)
                done=event['upper_s'];event['upper_h']=done/3600.
                event['rounded_up_h_4dp']=math.ceil(done/3600.*1e4)/1e4
                event['controlling_xi']=float(mesh.xi[event['controlling_node']])
                event['radius_at_upper_m']=float(ctrl.at([done])[0,2])
                event['last_full_step_time_s']=float(actual)
                break
            done=float(stop)
            probe[count]=state[:,mesh.probes]
            phy[count]=sample_physical(state,mesh,R)
            stats[count]=[done,R,np.min(state[1]),np.max(state[1]),2*np.dot(mesh.w,state[1]),np.argmax(state[1])]
            count+=1
            if int(round(done))%21600==0:
                snapshot_times.append(done);snapshots.append(state.copy())
            if int(round(done))%1800==0 or done==end:
                elapsed=time.perf_counter()-started
                log(f'[{label}] t={done/3600:.3f} h; max C={diag[21]:.8f}; '
                    f'R={R*100:.6f} cm; this-run={elapsed:.1f}s; '
                    f'Picard={diag[1]/diag[0]:.3f}/{int(diag[2])}',args.quiet)
            if not args.no_cache and args.checkpoint_seconds and done<end and (int(round(done))%args.checkpoint_seconds==0 or (done<=ctrl.transition_end and int(round(done))%1800==0)):
                atomic_npz(partial,key,pack(count),{'done_s':done,'prev_dt':prev_dt,
                           'runtime_s':prior+time.perf_counter()-started})
        if crossed: break
    if not crossed: final_state=state.copy()
    snapshot_times.append(done);snapshots.append(final_state.copy())
    arrays=pack(count,final_state)
    arrays['final_physical']=sample_physical(final_state,mesh,float(ctrl.at([done])[0,2]))
    # Blanks need a separate allowed key in the cache checker.
    arrays['final_physical_mask']=np.isfinite(arrays['final_physical']).astype(np.int64)
    arrays['final_physical']=np.nan_to_num(arrays['final_physical'],nan=0.)
    info={'config':asdict(c),'dried':bool(crossed),'end_s':float(done),'event':event,
          'threshold':args.threshold,'diagnostics':diagnose(diag),
          'wall_time_this_run_s':time.perf_counter()-started,
          'cumulative_runtime_s':prior+time.perf_counter()-started,
          'resumed_from_s':resumed_from,'cache_hit':False,'normal_output_rows':count}
    if int(diag[18])!=int(diag[0]) or diag[16]>1 or diag[17]>1:
        raise ArithmeticError('An implicit step escaped the residual gate.')
    lowerC=min(2.55,float(np.min(ctrl.env[:,2])))
    lowerT=min(28.,float(np.min(ctrl.env[:,1])));upperT=max(28.,float(np.max(ctrl.env[:,1])))
    if diag[12]<lowerC-1e-7 or diag[13]>max(2.55,float(np.max(ctrl.env[:,2])))+1e-7 or diag[10]<lowerT-1e-7 or diag[11]>upperT+1e-7:
        raise ArithmeticError('Solution violates physical range; BDF2 is not blindly assumed monotone.')
    if not args.no_cache:
        atomic_npz(finished_path,key,arrays,info)
        if partial.exists(): partial.unlink()
    log(f'[done] {label}: '+(f'dry time={done/3600:.9f} h' if crossed else f'NOT DRIED before {end/3600:g} h'),args.quiet)
    return arrays,info

def estimate_triplet(a,b,c,noise_floor=0.):
    d1=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    d2=float(np.max(np.abs(np.asarray(b)-np.asarray(c))))
    order=math.log2(d1/d2) if d1>0 and d2>0 else None
    if max(d1,d2)<=4*noise_floor and noise_floor>0:
        e=1.5*(d1+d2)+2*noise_floor;method='resolution_limited_observed_envelope'
    elif order is not None and .5<=order<=3.:
        e=1.5*d2/(2**min(order,2.)-1);method='Richardson_with_1.5_factor'
    else: e=None;method='unreliable_order_or_difference'
    return {'coarse_medium_max_abs':d1,'medium_fine_max_abs':d2,
            'observed_order':order,'estimated_error':e,'method':method,
            'is_rigorous_bound':False}

def convergence(results, keys, args):
    main=results[keys['main']][1]
    rep={'performed':not args.skip_convergence,
         'field_absolute_target':args.field_target,'event_time_target_s':args.event_target,
         'model_validation':False,'status':'NOT_CHECKED',
         'all_cases_dried':all(info['dried'] for _,info in results.values())}
    if not rep['all_cases_dried']:
        rep['status']='DRYING_NOT_REACHED';return rep
    if args.skip_convergence: return rep
    r=[results[keys[k]] for k in ('space_coarse','space_medium','main','time_medium','time_coarse')]
    count=min(len(a['probes']) for a,_ in r)
    if count==0:
        rep['status']='INSUFFICIENT_COMPARISON_OUTPUT';return rep
    rep['comparison_output_times_s']=[60,60*count]
    rep['comparison_normalized_radii']=PROBE_XI.tolist()
    rep['fields']={};passed=True
    for j,f in enumerate(('temperature','moisture')):
        s=estimate_triplet(r[0][0]['probes'][:count,j],r[1][0]['probes'][:count,j],r[2][0]['probes'][:count,j],2e-12 if j==0 else 2e-13)
        t=estimate_triplet(r[4][0]['probes'][:count,j],r[3][0]['probes'][:count,j],r[2][0]['probes'][:count,j],2e-12 if j==0 else 2e-13)
        total=s['estimated_error']+t['estimated_error'] if s['estimated_error'] is not None and t['estimated_error'] is not None else None
        ok=total is not None and total<=args.field_target
        rep['fields'][f]={'space':s,'time':t,'estimated_total_abs_error':total,'pass':ok};passed &=ok
    times=[info['event']['midpoint_s'] for _,info in r]
    noise=max(info['event']['bracket_width_s'] for _,info in r)
    se=estimate_triplet(times[0],times[1],times[2],noise)
    te=estimate_triplet(times[4],times[3],times[2],noise)
    total=se['estimated_error']+te['estimated_error']+noise if se['estimated_error'] is not None and te['estimated_error'] is not None else None
    ok=total is not None and total<=args.event_target
    rep['drying_time']={'space':se,'time':te,'event_bracket_noise_s':noise,
                        'estimated_total_error_s':total,'pass':ok}
    passed &=ok
    rep['status']='NUMERICAL_ESTIMATE_PASS' if passed else 'NEEDS_REFINEMENT_OR_REVIEW'
    return rep

def col_name(k):
    s='';k+=1
    while k: k,m=divmod(k-1,26);s=chr(65+m)+s
    return s

def write_result_xlsx(path,times,values):
    """Portable, dependency-free OOXML output. No attachment bytes embedded.
    One sheet, numeric radius headers + optional literal surface header.
    Missing material positions are omitted XML cells (true Excel blanks).
    """
    headers=['时间\\到药材中心的距离']+OUTPUT_CM.tolist()+(['药材表面'] if PROBLEM==4 else [])
    ncol=len(headers);endcell=col_name(ncol-1)+str(len(times)+1)
    p=[XML_DECL,f'<worksheet xmlns="{NS}"><dimension ref="A1:{endcell}"/>',
       '<sheetViews><sheetView workbookViewId="0"><pane xSplit="1" ySplit="1" topLeftCell="B2" activePane="bottomRight" state="frozen"/></sheetView></sheetViews>',
       '<sheetFormatPr defaultRowHeight="17"/><cols><col min="1" max="1" width="26" customWidth="1"/>',
       f'<col min="2" max="{ncol}" width="12" customWidth="1"/></cols><sheetData><row r="1" ht="34" customHeight="1">']
    for j,v in enumerate(headers):
        cell=col_name(j)+'1'
        if isinstance(v,str): p.append(f'<c r="{cell}" s="1" t="inlineStr"><is><t>{escape(v)}</t></is></c>')
        else: p.append(f'<c r="{cell}" s="3"><v>{v:.1f}</v></c>')
    p.append('</row>')
    for i,(t,row) in enumerate(zip(times,values),2):
        p.append(f'<row r="{i}"><c r="A{i}" s="2"><v>{t:.4f}</v></c>')
        for j,v in enumerate(row,1):
            if np.isnan(v): continue
            if not np.isfinite(v): raise ValueError('Non-finite Excel result.')
            p.append(f'<c r="{col_name(j)}{i}" s="2"><v>{v:.4f}</v></c>')
        p.append('</row>')
    p.append('</sheetData></worksheet>')
    styles=XML_DECL+f'''<styleSheet xmlns="{NS}"><numFmts count="2"><numFmt numFmtId="164" formatCode="0.0000"/><numFmt numFmtId="165" formatCode="0.0"/></numFmts><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFE8EEF3"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="4"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center"/></xf><xf numFmtId="165" fontId="1" fillId="2" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    parts={
        '[Content_Types].xml':XML_DECL+'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
        '_rels/.rels':XML_DECL+f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" Type="{REL}/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        'xl/workbook.xml':XML_DECL+f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels':XML_DECL+f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" Type="{REL}/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="{REL}/styles" Target="styles.xml"/></Relationships>',
        'xl/styles.xml':styles,'xl/worksheets/sheet1.xml':''.join(p)}
    tmp=path.with_name(path.name+'.tmp')
    with ZipFile(tmp,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for name,text in parts.items():
            info=ZipInfo(name,date_time=(2000,1,1,0,0,0));info.compress_type=ZIP_DEFLATED
            z.writestr(info,text.encode('utf-8'))
    os.replace(tmp,path)
    # Every physical value and every intended blank is checked on read-back.
    with ZipFile(path) as z:
        if z.testzip() is not None: raise AssertionError('ZIP integrity failed.')
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        names=[s.attrib['name'] for s in wb.findall(f'{{{NS}}}sheets/{{{NS}}}sheet')]
        if names!=['Sheet1']: raise AssertionError('Wrong template sheet name.')
        root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows=root.findall(f'{{{NS}}}sheetData/{{{NS}}}row')
        if len(rows)!=len(times)+1: raise AssertionError('Wrong output row count.')
        for i,rr in enumerate(rows[1:]):
            cells={column_number(cc.attrib['r']):cc for cc in rr}
            if cells[0].find(f'{{{NS}}}v').text!=f'{times[i]:.4f}': raise AssertionError('Wrong time.')
            for j,v in enumerate(values[i],1):
                if np.isnan(v):
                    if j in cells: raise AssertionError('Out-of-body position is not blank.')
                elif cells[j].find(f'{{{NS}}}v').text!=f'{v:.4f}': raise AssertionError('Round-trip value mismatch.')
    return {'sheet':'Sheet1','dimensions':f'A1:{endcell}','zip_integrity':'ok',
            'all_values_and_blanks_checked':True,'result_format':'0.0000'}

def export_outputs(outdir,results,keys,report,meta,args,ctrl):
    arr,info=results[keys['main']]
    normal_times=arr['stats'][:,0]
    end=info['end_s'];Rf=float(ctrl.at([end])[0,2])
    final=arr['final_physical'].copy()
    final[arr['final_physical_mask']==0]=np.nan
    times=normal_times.copy();physical=arr['physical'].copy();stats=arr['stats'].copy()
    if not len(times) or end>times[-1]+1e-8:
        times=np.r_[times,end];physical=np.concatenate((physical,final[None]),axis=0)
        Cf=arr['state'][1];mesh=make_mesh(args.radial_cells)
        extra=[end,Rf,float(np.min(Cf)),float(np.max(Cf)),float(2*np.dot(mesh.w,Cf)),int(np.argmax(Cf))]
        stats=np.vstack((stats,extra))
    complete=info['dried']
    filename=f'result{PROBLEM}.xlsx' if complete else f'result{PROBLEM}_NOT_DRIED.xlsx'
    meta['xlsx']=write_result_xlsx(outdir/filename,times,physical[:,1])
    coord=['r_'+f'{x:.1f}'+'_cm' for x in OUTPUT_CM]+(['surface'] if PROBLEM==4 else [])
    for f,name in enumerate(('temperature','moisture')):
        def rows():
            for i,t in enumerate(times):
                yield [format(t,'.17g')]+[('' if np.isnan(v) else format(v,'.17g')) for v in physical[i,f]]
        atomic_csv(outdir/(name+'_full_precision.csv'),['time_s']+coord,rows())
    atomic_csv(outdir/'radius_and_maximum.csv',
               ['time_s','radius_m','min_C','max_C_all_grid_nodes','mean_C_dry_mass_weighted','max_node'],
               ([format(x,'.17g') for x in row] for row in stats))
    profile_key=hashlib.sha256((meta['source_sha256']+'profiles').encode()).hexdigest()
    atomic_npz(outdir/'fine_grid_profiles.npz',profile_key,
               {'xi':arr['xi'],'time_s':arr['snapshot_times'],'fields_T_C':arr['snapshots'],
                'radius_m':ctrl.at(arr['snapshot_times'])[:,2]},
               {'meaning':'At each time r_m=xi*radius_m; T in Celsius, C in kg/kg dry basis.'})
    # Paper table 5/6: every 6 h, and the non-grid-aligned terminal event row.
    pcols=[0,5,10,15,20] if PROBLEM==3 else [0,5,10,15,len(OUTPUT_CM)]
    if PROBLEM==4:
        pcols=[j for j in pcols if j==len(OUTPUT_CM) or np.any(np.isfinite(physical[np.isclose(np.mod(times,21600),0,atol=1e-8),1,j]))]
        if not pcols: pcols=[0,5,10,len(OUTPUT_CM)]
    labels=[('药材表面' if PROBLEM==4 and j==len(OUTPUT_CM) else f'{OUTPUT_CM[j]:g} cm') for j in pcols]
    text=[f'# 问题 {PROBLEM}：表 {5 if PROBLEM==3 else 6}','',
          '模型数值结果，不是实测数据。空白表示该实际位置已在药材之外。',
          '终止条件使用全网格未舍入含水率；显示为 0.1500 不表示原始值未低于阈值。','',
          '| 时间/h | '+' | '.join(labels)+' |','|---:|'+'---:|'*len(labels)]
    select=[i for i,t in enumerate(times) if abs(t/21600-round(t/21600))<1e-9]
    if len(times) and (not select or select[-1]!=len(times)-1):select.append(len(times)-1)
    for i in select:
        text.append('| '+f'{times[i]/3600:.4f}'+' | '+' | '.join('' if np.isnan(physical[i,1,j]) else f'{physical[i,1,j]:.4f}' for j in pcols)+' |')
    if not complete:text.insert(2,'**尚未达到烘干条件；末行不是烘干结束时间。**')
    (outdir/'summary_tables.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    meta['main_result']=info
    meta['cases']={name:results[key][1] for name,key in keys.items()}
    meta['verification']=report
    atomic_json(outdir/'run_metadata.json',meta)
    event_out={'problem':PROBLEM,'dried':complete,'event':info['event'],
               'verification':report,'model_closures':meta['model_closures']}
    atomic_json(outdir/'drying_time.json',event_out)
    rep=[f'# 第 {PROBLEM} 问数值验证报告','',f'状态：{report["status"]}',
         f'主网格：{args.radial_cells} 个区间。首分钟 dt={args.dt:g} s；附件期间上限 dt={args.warm_dt:g} s；后期上限 dt={args.late_dt:g} s。',
         '换步长采用真实步长比的 BDF2 权重，最大增大比为 2；不是用旧权重强行换步。',
         '首分钟后逐级放宽到附件期间上限，过渡后逐级放宽到后期上限。稳态的是外界设定值，并不把药材温度强制设成 50°C。','',
         '## 题给信息与明示闭合假设',*meta['model_closures'],'',
         '附件 1 的有效数据范围：'+str(meta['input_environment']['time_range_s'])+' s。',
         '附件 1 SHA-256：'+meta['input_environment']['sha256']]
    if PROBLEM==4:
        rep += ['附件 2 的有效数据范围：'+str(meta['input_radius']['time_range_s'])+' s。',
                '附件 2 SHA-256：'+meta['input_radius']['sha256'],
                'xi=r/R(t) 为等比例收缩的随体坐标；干基 C 不添加体积浓度的压缩/稀释项。',
                '在该假设下：rho cp T_t|xi = div_xi(k grad_xi T)/R²，C_t|xi = div_xi(D grad_xi C)/R²。',
                '表面边界在每步使用当时 R：-k T_xi/R=h(T_s-T_air)，-D C_xi/R=hm(C_s-C_air)。',
                '输出按实际厘米坐标采样，额外给出真正的移动表面；超出半径的格子留空。']
    if complete:
        ev=info['event']
        rep+=['','## 烘干事件',f'离散解事件区间：[{ev["lower_s"]:.12f}, {ev["upper_s"]:.12f}] s。',
              f'区间上端：{ev["upper_h"]:.12f} h；操作上向上保留四位小数：{ev["rounded_up_h_4dp"]:.4f} h。',
              f'上端全网格最大 C：{ev["upper_max_C"]:.16g}；下端：{ev["lower_max_C"]:.16g}。',
              f'控制节点 xi={ev["controlling_xi"]:.12g}；最后完整时间步={ev["last_full_step_s"]:g} s。',
              '事件二分区间只是所选时空离散解的根定位精度，不是烘干时间的总误差。']
    else:
        rep+=['','**计算范围内未达到烘干条件；没有生成可冒充最终结果的 result 文件。**',
              '第四问如因半径数据范围终止，不应自动外推。经建模确认后才可显式使用 --radius-tail hold。']
    rep+=['','## 空间和时间的独立检验','每 60 s、281 个共同随体位置进行两个场的空间/时间比较。',
          '主解只求解一次，空间检验固定时间计划，时间检验固定网格并同时加密初期/后期步长。',
          '烘干时刻独立比较；不能用含水率误差替代事件时间误差。误差估计不是严格上界。',
          '```json',json.dumps(report,ensure_ascii=False,indent=2),'```','',
          '## 非线性与离散收支','每个隐式步都重新计算候选最终状态的局部残差，并与增量判据一起验收。',
          '局部残差允许值=rtol×该行物理项绝对值之和 + 64 eps×行对角元×(1+|旧状态|+|增量|)。',
          '第二项显式表示 float64 舍入底噪；并不是声称残差能在任意细网格上无限小。',
          '总体收支以 Kahan 累加，检验所选 PDE；不声称完整热力学能量守恒或实验验证。',
          '```json',json.dumps(info['diagnostics'],ensure_ascii=False,indent=2),'```','',
          '## 性能、输出和恢复','Numba native/nogil/float64，关闭 fastmath 与嵌套线程；最多 24 核。',
          '几何/共同控制输入复用，工作数组预分配；每步旧梯度与历史项外提；物性随场更新，不缓存错误矩阵。',
          '并行单位是 5 组独立耦合收敛任务；不重复任务，不错误并行时间步或解耦温湿两个场。',
          '缓存包含代码/输入/配置/版本标识。只有成功写入的检查点可恢复；运行时长不是实验依据。',
          'Excel 按题给模板展开为完整列，每 60 秒输出并加最后事件行；CSV 保留原始数值。',
          'Excel 每个数值和物体外空白均已回读检查。',
          '极细网格只能降低所选模型的数值误差；无法消除输入的毫米/温湿度采样误差或闭合假设误差。']
    (outdir/'validation_report.md').write_text('\n'.join(rep)+'\n',encoding='utf-8')
    return filename

def self_tests():
    """Synthetic diagnostics ONLY; never used instead of actual attachments."""
    tests={}
    rng=np.random.default_rng(20260304);n=17
    m=rng.uniform(.2,2.,n);cap=rng.uniform(1,3,n)
    gt=rng.uniform(.01,3,n-1);gc=rng.uniform(.01,3,n-1)
    ht=rng.normal(size=n);hc=rng.normal(size=n)
    ft=rng.normal(size=n-1);fc=rng.normal(size=n-1)
    rt=np.empty(n);rc=np.empty(n);ut=np.empty(n);uc=np.empty(n)
    paired_thomas(m,cap,gt,gc,ht,hc,ft,fc,.7,.3,.2,-.1,rt,rc,ut,uc)
    errs=[]
    for mass,g,hist,flux,bd,src,u in ((m*cap,gt,ht*cap,ft,.7,.2,ut),(m,gc,hc,fc,.3,-.1,uc)):
        A=np.diag(mass+np.r_[g,0]+np.r_[0,g])+np.diag(-g,1)+np.diag(-g,-1);A[-1,-1]+=bd
        rhs=hist+np.r_[flux,src]-np.r_[0,flux]
        errs.append(float(np.max(abs(np.linalg.solve(A,rhs)-u))))
    if max(errs)>2e-12: raise AssertionError('Paired tridiagonal vs dense solve failed.')
    tests['paired_tridiagonal_dense_max_errors']=errs
    mesh=make_mesh(380);N=381;state=np.array([np.full(N,28.),np.full(N,2.55)])
    prev=np.zeros_like(state);scratch=np.empty((18,N));diag=np.zeros(22);diag[10:14]=[28,28,2.55,2.55]
    for R in (.02,.019,.016,.012):
        advance(state[0],state[1],prev[0],prev[1],mesh.w,mesh.g,R,28.,2.55,.125,1.,0.,0.,
                2e-12,2e-13,1e-10,40,scratch,diag)
    if np.max(abs(state[0]-28))>1e-13 or np.max(abs(state[1]-2.55))>1e-13:
        raise AssertionError('Uniform sealed shrinking-body invariance failed.')
    tests['uniform_dry_basis_no_artificial_shrinkage_source']=True
    coefficients(state[0],state[1],prev[0],prev[1],scratch[6],scratch[7],scratch[8])
    c=2.55;z=c/(1+c)
    ex=((650+128*c)*(1450+2736*z),.21+.38*z,.0024*math.exp(-.45/c-3850/301.15)) if PROBLEM==3 else ((760+90*c)*(1850+2150*z),.12+.20*z,.00042*math.exp(-.30/c-3850/301.15))
    if max(abs(scratch[6+i,0]/ex[i]-1) for i in range(3))>1e-13:raise AssertionError('Material law/Kelvin test failed.')
    tests['appendix_coefficients_and_kelvin']=True
    # Independent scalar BDF2 convergence including two-half-step startup.
    errors=[]
    for nsteps in (32,64,128):
        h=1./nsteps;u=1.;prevdelta=0.
        for j in range(nsteps):
            if j==0:
                v=u/(1+h/2)**2;delta=v-u
            else:
                delta=(.5*prevdelta-h*u)/(1.5+h);v=u+delta
            u=v;prevdelta=delta
        errors.append(abs(u-math.exp(-1)))
    orders=[math.log2(errors[i]/errors[i+1]) for i in range(2)]
    if not all(1.8<p<2.2 for p in orders):raise AssertionError('BDF2 order test failed.')
    tests['scalar_BDF2_orders']=orders
    # Max NOT mean, with an intentionally off-center controlling node.
    w=np.array([.1,.2,.2]);R=.02;h=10.;hp=10.
    new=np.array([[30.,30.,30.],[.14,.145,.12]])
    delta=np.array([[0.,0.,0.],[-.02,-.02,-.02]])
    s=np.zeros((18,3));s[2]=0.;s[3]=w*.5*R*R/h*delta[1]
    final,ev=event_bracket(new,delta,s,w,20.,h,hp,R,.15,1e-5)
    if not ev['lower_s']<=17.5<=ev['upper_s'] or ev['controlling_node']!=1 or np.max(final[1])>=.15:
        raise AssertionError('All-node drying event test failed.')
    tests['all_node_event_not_average_or_assumed_center']=True
    tests['every_step_residual_gating']=int(diag[0])==int(diag[18])
    return tests

class OutputLock:
    def __init__(self,path):self.path=path;self.held=False
    def __enter__(self):
        if self.path.exists():
            try:
                data=json.loads(self.path.read_text());pid=int(data['pid'])
                if data['host']!=platform.node():raise RuntimeError('Output directory is locked by another host.')
                try:os.kill(pid,0)
                except ProcessLookupError:self.path.unlink()
                else:raise RuntimeError(f'Output directory is locked by running PID {pid}.')
            except (ValueError,KeyError,json.JSONDecodeError) as exc:
                raise RuntimeError('Invalid lock file; inspect it before deleting: '+str(self.path)) from exc
        fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'host':platform.node()},f)
        self.held=True;return self
    def __exit__(self,*_):
        if self.held:
            try:self.path.unlink()
            except FileNotFoundError:pass

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    if PROBLEM==4:
        p.add_argument('--radius-input',type=Path,default=DEFAULT_RADIUS)
        p.add_argument('--radius-tail',choices=('error','hold'),default='error')
    p.add_argument('--outdir',type=Path,default=Path(f'/home/zyf/CUMCM/problem/result_problem{PROBLEM}_fine'))
    p.add_argument('--workers',type=int,default=24)
    p.add_argument('--radial-cells',type=int,default=12160)
    p.add_argument('--dt',type=float,default=.0078125)
    p.add_argument('--warm-dt',type=float,default=.125)
    p.add_argument('--late-dt',type=float,default=1.)
    p.add_argument('--max-hours',type=float,default=240.)
    p.add_argument('--air-temperature',type=float,default=50.)
    p.add_argument('--air-moisture',type=float,default=.05)
    p.add_argument('--tail-ramp',type=int,default=60)
    p.add_argument('--threshold',type=float,default=.15)
    p.add_argument('--heat-transfer',type=float,default=25.)
    p.add_argument('--mass-transfer',type=float,default=8e-7)
    p.add_argument('--temperature-tol',type=float,default=2e-12)
    p.add_argument('--moisture-tol',type=float,default=2e-13)
    p.add_argument('--residual-rtol',type=float,default=1e-10)
    p.add_argument('--picard-max',type=int,default=40)
    p.add_argument('--event-tol',type=float,default=1e-4,help='Root bracket width, seconds; NOT global error.')
    p.add_argument('--field-target',type=float,default=1e-6)
    p.add_argument('--event-target',type=float,default=.05,help='Space+time+root empirical event error target, seconds.')
    p.add_argument('--checkpoint-seconds',type=int,default=21600)
    p.add_argument('--skip-convergence',action='store_true')
    p.add_argument('--no-cache',action='store_true')
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--self-test-only',action='store_true')
    return p.parse_args(argv)

def main(argv=None):
    args=parse_args(argv)
    cap,cpu=cpu_limit(args.workers)
    if args.self_test_only:
        print(json.dumps(self_tests(),ensure_ascii=False,indent=2));return 0
    for key in ('dt','warm_dt','late_dt','max_hours','air_temperature','air_moisture','threshold',
                'heat_transfer','mass_transfer','temperature_tol','moisture_tol',
                'residual_rtol','event_tol','field_target','event_target'):
        if not math.isfinite(getattr(args,key)):raise ValueError(f'Nonfinite --{key}.')
    if not 0<args.threshold<2.55 or not 0<args.air_moisture<args.threshold:
        raise ValueError('Require 0<air-moisture<threshold<initial 2.55.')
    if args.air_temperature<=-273.15 or min(args.heat_transfer,args.mass_transfer)<=0:
        raise ValueError('Invalid thermal environment or transfer coefficient.')
    if min(args.temperature_tol,args.moisture_tol,args.residual_rtol,args.event_tol,args.field_target,args.event_target)<=0 or args.picard_max<2:
        raise ValueError('Invalid numerical tolerances.')
    if args.event_tol>=args.dt: raise ValueError('event-tol must be smaller than dt.')
    if args.max_hours<=0 or args.max_hours>720:raise ValueError('max-hours must be in (0,720].')
    end=int(round(args.max_hours*3600))
    if end<60 or end%60 or abs(end-args.max_hours*3600)>1e-6:raise ValueError('max-hours must end on a full minute.')
    if args.checkpoint_seconds<0 or args.checkpoint_seconds%60:raise ValueError('Checkpoint period must be zero or a multiple of 60 s.')
    if not args.skip_convergence and (args.radial_cells<1520 or args.radial_cells%1520):
        raise ValueError('For three nested grids, radial-cells must be a positive multiple of 1520.')
    env=load_environment(args.input,0)
    if env[0,0]!=0:raise ValueError('附件1 must start at t=0.')
    radius=load_radius(args.radius_input) if PROBLEM==4 else np.array([[0.,.02],[float(end),.02]])
    if PROBLEM==4 and args.radius_tail=='error':
        radius_end=int(round(radius[-1,0]))
        if radius_end%60:raise ValueError('Radius coverage must end at a full minute in this schedule.')
        end=min(end,radius_end)
    ctrl=Controls(env,radius,args.air_temperature,args.air_moisture,args.tail_ramp)
    base=Config(args.radial_cells,args.dt,args.late_dt,args.temperature_tol,args.moisture_tol,
                args.residual_rtol,args.picard_max,args.event_tol,args.warm_dt)
    definitions={'main':base}
    if not args.skip_convergence:
        definitions.update({
            'space_coarse':Config(args.radial_cells//4,args.dt,args.late_dt,args.temperature_tol,args.moisture_tol,args.residual_rtol,args.picard_max,args.event_tol,args.warm_dt),
            'space_medium':Config(args.radial_cells//2,args.dt,args.late_dt,args.temperature_tol,args.moisture_tol,args.residual_rtol,args.picard_max,args.event_tol,args.warm_dt),
            'time_medium':Config(args.radial_cells,args.dt*2,args.late_dt*2,args.temperature_tol,args.moisture_tol,args.residual_rtol,args.picard_max,args.event_tol,args.warm_dt*2),
            'time_coarse':Config(args.radial_cells,args.dt*4,args.late_dt*4,args.temperature_tol,args.moisture_tol,args.residual_rtol,args.picard_max,args.event_tol,args.warm_dt*4)})
    for cfg in definitions.values(): phases(cfg,ctrl.transition_end,end)
    args.outdir.mkdir(parents=True,exist_ok=True)
    cache_dir=args.outdir/'cache';cache_dir.mkdir(exist_ok=True)
    status_path=args.outdir/'status.json';status={}
    with OutputLock(args.outdir/'.running.lock'):
        start=time.perf_counter();cancel=threading.Event()
        hashes={'environment':sha256(args.input)}
        if PROBLEM==4:hashes['radius']=sha256(args.radius_input)
        source_hash=sha256(Path(__file__))
        closures=[
            f'物性：全程使用附录 {3 if PROBLEM==3 else 4}，温度指数使用 T_C+273.15；初始 T=28°C、C=2.55 kg/kg。',
            '一维径向，忽略端部效应、显式潜热和辐射。此为明示简化，不是题面已证明其效应为零。',
            f'继承假设：h={args.heat_transfer:g} W/(m² K)，hm={args.mass_transfer:g} m/s。',
            '有效 Robin 传质边界直接使用题给环境 kg/kg 变量；没有声称药材和空气质量基准热力学等价。',
            f'附件1完整区间采用分段线性插值；之后 {args.tail_ramp} s 过渡到 {args.air_temperature:g}°C/{args.air_moisture:g} kg/kg，再恒定延拓；这些延拓不是测量值。',
            'rho(C)*cp(C)*material_dT/dt 为有效热储存模型，不是 d(rho cp T)/dt 或完整组分焓模型。',
            '终止判据：全网格未舍入最大干基含水率严格低于阈值；连续物理模型的阈值时刻以收敛估计近似。']
        if PROBLEM==4:
            closures += ['附件2半径先 cm→m，再分段线性插值；假设固体沿径向等比例随体收缩、轴向作用忽略。',
                         '干基含水率不使用体积浓度的几何压缩项。密度经验式用于热物性，不用于反推另一个与附件冲突的半径。',
                         f'半径数据末端处理：{args.radius_tail}；error 表示未干即停止，hold 表示显式冻结最后实测半径。']
        else:closures+=['第三问半径固定 0.02 m。']
        physics={k:getattr(args,k) for k in ('threshold','air_temperature','air_moisture','tail_ramp','heat_transfer','mass_transfer')}
        if PROBLEM==4:physics['radius_tail']=args.radius_tail
        common={'problem':PROBLEM,'source':source_hash,'input_hashes':hashes,'physics':physics,'end_limit_s':end,
                'numpy':np.__version__,'numba':numba.__version__,'python':platform.python_version()}
        keys={};configs={}
        for name,cfg in definitions.items():
            key=hashlib.sha256(json.dumps({**common,'config':asdict(cfg)},sort_keys=True).encode()).hexdigest()
            keys[name]=key;configs[key]=cfg
        status={'status':'RUNNING','pid':os.getpid(),'problem':PROBLEM,'started_utc':datetime.now(timezone.utc).isoformat()}
        atomic_json(status_path,status)
        try:
            tests=self_tests()
            log(f'Q{PROBLEM}: {len(configs)} distinct coupled cases; CPU cap={cap}; '
                f'end limit={end/3600:g} h; initial/warm/late dt={args.dt:g}/{args.warm_dt:g}/{args.late_dt:g} s.',args.quiet)
            log(f'External tail: keep all measurements, then {args.tail_ramp}s transition to '
                f'{args.air_temperature:g} C / {args.air_moisture:g} kg/kg. No body-temperature clamp.',args.quiet)
            if PROBLEM==4:
                log(f'Radius data: {radius[0,0]:g}..{radius[-1,0]:g}s; {radius[0,1]*100:g}..{radius[-1,1]*100:g}cm; tail={args.radius_tail}.',args.quiet)
            meshes={n:make_mesh(n) for n in {c.radial_cells for c in configs.values()}}
            timelines={}
            for cfg in configs.values():
                k=(cfg.dt,cfg.warm_dt,cfg.late_dt)
                if k not in timelines:timelines[k]=prepare_timeline(cfg,ctrl,end)
            forcing_bytes=sum(p['values'].nbytes for line in timelines.values() for p in line)
            log(f'Shared forcing arrays: {forcing_bytes/2**20:.1f} MiB; all spatial cases reuse one timeline.',args.quiet)
            results={};pool=ThreadPoolExecutor(max_workers=min(cap,len(configs)))
            pending={}
            try:
                # Longest expected job first; no duplicate main solve.
                for key,cfg in sorted(configs.items(),key=lambda kv:kv[1].radial_cells/kv[1].dt,reverse=True):
                    future=pool.submit(run_case,cfg,meshes[cfg.radial_cells],timelines[(cfg.dt,cfg.warm_dt,cfg.late_dt)],
                                      ctrl,end,args,key,cache_dir,cancel)
                    pending[future]=key
                for f in as_completed(pending):results[pending[f]]=f.result()
            except BaseException:
                cancel.set()
                for f in pending:f.cancel()
                raise
            finally:pool.shutdown(wait=True,cancel_futures=True)
            if sha256(args.input)!=hashes['environment'] or (PROBLEM==4 and sha256(args.radius_input)!=hashes['radius']):
                raise RuntimeError('Input changed during solving; refusing mixed-source outputs.')
            report=convergence(results,keys,args)
            meta={'problem':PROBLEM,'source_sha256':source_hash,'version':VERSION,'cpu':cpu,
                  'numpy':np.__version__,'numba':numba.__version__,'self_tests':tests,
                  'input_environment':{'path':str(args.input.resolve()),'sha256':hashes['environment'],
                    'rows':len(env),'time_range_s':[float(env[0,0]),float(env[-1,0])]},
                  'input_radius':({'path':str(args.radius_input.resolve()),'sha256':hashes['radius'],'rows':len(radius),
                    'time_range_s':[float(radius[0,0]),float(radius[-1,0])]} if PROBLEM==4 else None),
                  'model_closures':closures,'main_config':asdict(base),
                  'shared_forcing_memory_bytes':forcing_bytes,'total_wall_s':time.perf_counter()-start,
                  'completed_utc':datetime.now(timezone.utc).isoformat()}
            filename=export_outputs(args.outdir,results,keys,report,meta,args,ctrl)
            event=results[keys['main']][1]['event']
            code=0 if report['status'] in ('NUMERICAL_ESTIMATE_PASS','NOT_CHECKED') and results[keys['main']][1]['dried'] else 2
            status.update({'status':'COMPLETED' if code==0 else 'COMPLETED_NEEDS_REVIEW',
                           'numerical_verification':report['status'],'wall_s':time.perf_counter()-start})
            atomic_json(status_path,status)
            print('Output:',args.outdir/filename,flush=True)
            if event:print(f'Drying estimate: {event["upper_h"]:.12f} h; strict upper root bracket; see drying_time.json.',flush=True)
            print('Verification:',report['status'],'; see validation_report.md.',flush=True)
            if code:print('WARNING: no high-precision pass claimed; inspect drying/convergence coverage before using the results.',flush=True)
            return code
        except BaseException as exc:
            status.update({'status':'FAILED','error':str(exc),'ended_utc':datetime.now(timezone.utc).isoformat()})
            atomic_json(status_path,status)
            raise

if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Only successfully saved checkpoints can be resumed.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(1)

