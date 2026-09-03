"""
Cube spinodal overlays -- run from src/:  python cube_verification_figure.py

Cube spinodal overlays: published avgE vs rebuilt-cache avgE verification,
against MC.  Parser, confirmed_boundary rule and Gaussian smoothing are taken
verbatim from notebooks/threed_plots.ipynb; style is the notebook's PRL/APS block.
"""
import re, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from matplotlib.colors import rgb_to_hsv

H = Path(__file__).resolve().parent.parent          # repo root (script lives in src/)
LOGS, DATA, FIGS = H/"results_logs", H/"data", H/"figures"
FIGS_INSET = FIGS
FPP_TOL_CONFIRM, CONFIRM_K, MAX_RES_CUTOFF = 0.03, 2, 1.0

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|-inf|INF|-INF"
POINT_RE = re.compile(
    rf"POINT .*?eps_a=({NUM})\s+eps_c=({NUM}).*?event=(\d+)\s+spin=(\d+)\s+bin=(\d+).*?"
    rf"min_fpp=({NUM})\s+phi_min_fpp=({NUM}).*?n_valid=(\d+)/(\d+).*?"
    rf"n_success=(\d+)\s+max_res=({NUM})", re.DOTALL)
def tof(x):
    try: return float(x)
    except Exception: return np.nan
def parse_log(p):
    rows=[dict(eps_a=tof(m.group(1)),eps_c=tof(m.group(2)),event=int(m.group(3)),
               spin=int(m.group(4)),bin=int(m.group(5)),min_fpp=tof(m.group(6)),
               phi_min_fpp=tof(m.group(7)),n_valid=int(m.group(8)),n_phi=int(m.group(9)),
               n_success=int(m.group(10)),max_res=tof(m.group(11)))
          for m in POINT_RE.finditer(Path(p).read_text(errors="ignore"))]
    return pd.DataFrame(rows).drop_duplicates().sort_values(["eps_c","eps_a"]).reset_index(drop=True)
def classify(df):
    df=df.copy()
    df["valid_solver"]=(np.isfinite(df["min_fpp"])&np.isfinite(df["max_res"])
                        &(df["max_res"]<=MAX_RES_CUTOFF)&(df["n_success"]>0))
    return df
def confirmed_boundary(df, fpp_tol=FPP_TOL_CONFIRM, confirm_k=CONFIRM_K):
    rows=[]
    for eps_c,col in df.groupby("eps_c"):
        col=col.sort_values("eps_a").reset_index(drop=True)
        uns=((col["valid_solver"])&(col["spin"]==1)&np.isfinite(col["min_fpp"])
             &(col["min_fpp"]<-abs(fpp_tol))).to_numpy()
        got=np.nan
        for i in range(len(col)):
            if i+confirm_k+1<=len(col) and np.all(uns[i:i+confirm_k+1]):
                got=float(col.iloc[i]["eps_a"]); break
        rows.append(dict(eps_c=float(eps_c), eps_a_boundary=got))
    b=pd.DataFrame(rows)
    return b[np.isfinite(b["eps_a_boundary"])]
def drop_redundant(b):
    """
    Keep only points that strictly lower the boundary as eps_c increases.

    The critical eps_a falls monotonically with eps_c, so once the boundary has
    reached a value, a later eps_c sitting at the same or a higher eps_a is not
    a boundary point -- it is already inside the two-phase region.  This removes
    the trailing run along eps_a = 0 (e.g. the stick verification has boundary
    points at eps_c = 1.0, 1.1 and 1.2 all at eps_a = 0; only 1.0 is the
    boundary) and any non-monotone excursions from isolated noisy points.
    """
    b=b.sort_values("eps_c").reset_index(drop=True)
    keep=[]; best=np.inf
    for r in b.itertuples():
        if r.eps_a_boundary < best-1e-12:
            keep.append(r.Index); best=r.eps_a_boundary
    return b.loc[keep].reset_index(drop=True)

def _slope(x,y,k=10,side="left"):
    k=min(int(k),len(x))
    if k<2: return 0.0
    xx,yy=(x[:k],y[:k]) if side=="left" else (x[-k:],y[-k:])
    return float(np.linalg.lstsq(np.vstack([xx,np.ones_like(xx)]).T,yy,rcond=None)[0][0])
def gsmooth(x,y,*,sigma=50.0,n=900,pad_mult=4,kfit=10):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    i=np.argsort(x); x,y=x[i],y[i]
    t=pd.DataFrame({"x":x,"y":y}).groupby("x",as_index=False)["y"].mean()
    x=t["x"].to_numpy(float); y=t["y"].to_numpy(float)
    if len(x)<2: return x,y
    xg=np.linspace(x.min(),x.max(),int(n)); yg=np.interp(xg,x,y)
    pad=int(max(3,pad_mult*sigma)); dx=xg[1]-xg[0]
    mL=_slope(xg,yg,kfit,"left"); mR=_slope(xg,yg,kfit,"right")
    lx=xg[0]-dx*np.arange(pad,0,-1); rx=xg[-1]+dx*np.arange(1,pad+1)
    ygp=np.concatenate([yg[0]+mL*(lx-xg[0]),yg,yg[-1]+mR*(rx-xg[-1])])
    return xg, gaussian_filter1d(ygp,sigma=float(sigma),mode="nearest")[pad:-pad]
def trim(x,y,floor=0.0,eps=2e-2):
    x=np.asarray(x); y=np.asarray(y); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    i=np.where(y<=(floor+eps))[0]
    return (x[:i[0]+1],y[:i[0]+1]) if i.size else (x,y)

# ---- MC for 000011: digitized from jacobs_curve.png (notebook cell 1) ----
arr=np.array(Image.open(FIGS/"jacobs_curve.png").convert("RGB"))
xl,xr,yt,yb=158,737,59,645
hsv=rgb_to_hsv(arr/255.0)
org=((hsv[...,0]>0.03)&(hsv[...,0]<0.08)&(hsv[...,1]>0.40)&(hsv[...,2]>0.50))
x0,x1,y0,y1=486,670,450,645
roi=org[y0:y1+1,x0:x1+1]
e=np.array([(x0+j,y0+np.where(roi[:,j])[0].max()) for j in range(roi.shape[1])
            if np.where(roi[:,j])[0].size],float)
c=e.copy(); c[:,1]-=7.0
main=c[c[:,0]<=648].copy(); tail=main[main[:,0]>=620]
mm,bb=np.polyfit(tail[:,0],tail[:,1],1); xe=np.arange(main[-1,0]+1,669)
cl=np.vstack([main,np.column_stack([xe,mm*xe+bb])]) if len(xe) else main
mcc=(cl[:,0]-xl)/(xr-xl); mca=(yb-cl[:,1])/(yb-yt)*8.0
xd=np.linspace(mcc.min(),mcc.max(),300)
MC000011=pd.DataFrame({"eps_c":xd,"eps_a":gaussian_filter1d(np.interp(xd,mcc,mca),sigma=2.0)})
# ---- SAFT for 000011 ----
sp=np.array(json.load(open(DATA/"spinodal_du_saft.json")))
EA=np.linspace(0,8,21); EC=np.linspace(0,1.5,31)
sc,sa=[],[]
for j in range(len(EC)):
    w=np.where(sp[:,j]==1)[0]
    if len(w) and EC[j]<1.22: sa.append(EA[w.min()]); sc.append(EC[j])
SAFT000011=pd.DataFrame({"eps_c":sc,"eps_a":sa})
# ---- MC / SAFT for 111000 ----
# MC: the digitized overlay shipped in data/ (an output of threed_plots.ipynb cell 5).
MC111000=pd.read_csv(DATA/"planar3patch_11469_overlay_mc_digitized.csv")
# SAFT: cell 5 builds this from spinodal_tri_saft.json on its OWN eps_a grid,
# linspace(0, 5, 21) -- not the 0..8 grid the stick panel (cell 4) uses, and not
# the digitized CSV.  Getting this wrong rescales the whole SAFT curve.
sp_tri=np.array(json.load(open(DATA/"spinodal_tri_saft.json")))
EA_TRI=np.linspace(0,5,21); EC_TRI=np.linspace(0,1.5,31)
tc,ta=[],[]
for j in range(len(EC_TRI)):
    w=np.where(sp_tri[:,j]==1)[0]
    if len(w) and EC_TRI[j]<1.22: ta.append(EA_TRI[w.min()]); tc.append(EC_TRI[j])
SAFT111000=pd.DataFrame({"eps_c":tc,"eps_a":ta})

# Published SAFT-P boundary for the coplanar case ships as a CSV in data/
# (planar3patch_11469_overlay_saftp_boundary.csv) -- that is the curve behind the
# published overlay figure, so use it rather than re-deriving one from a log.
PUB_SAFTP_111000=pd.read_csv(DATA/"planar3patch_11469_overlay_saftp_boundary.csv")\
                   .rename(columns={"eps_a":"eps_a_boundary"})[["eps_c","eps_a_boundary"]]

PANELS=[dict(tag="000011", title=r"(a)  two opposite patches", inset="patchy_3d_stick.png",
             pub="cube_scan_bethe.11439.out", pub_csv=None, new="cube_scan_bethe.16513.out",
             mc=MC000011, saft=SAFT000011, sigma=70.0, xlim=(0,1.5), ylim=(0,6)),
        dict(tag="111000", title=r"(b)  three coplanar patches", inset="patchy_3d_planar3.png",
             pub=None, pub_csv=PUB_SAFTP_111000, new="cube_scan_bethe.16514.out",
             mc=MC111000, saft=SAFT111000, sigma=100.0, xlim=(0,1.5), ylim=(0,6))]

mpl.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","STIXGeneral","DejaVu Serif"],
    "mathtext.fontset":"stix","font.size":8,"axes.labelsize":10,"legend.fontsize":6.5,
    "xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":0.8,"lines.linewidth":1.8,
    "lines.solid_capstyle":"round","lines.dash_capstyle":"round","xtick.direction":"in",
    "ytick.direction":"in","xtick.major.size":3.5,"ytick.major.size":3.5,"xtick.minor.size":2.0,
    "ytick.minor.size":2.0,"xtick.major.width":0.7,"ytick.major.width":0.7,"xtick.minor.width":0.6,
    "ytick.minor.width":0.6,"legend.frameon":False,"savefig.bbox":"tight","savefig.pad_inches":0.02,
    "pdf.fonttype":42,"ps.fonttype":42})

fig,axes=plt.subplots(1,2,figsize=(6.9,2.6),dpi=300,constrained_layout=True)
for ax,P in zip(axes,PANELS):
    bp = P["pub_csv"] if P["pub_csv"] is not None else confirmed_boundary(classify(parse_log(LOGS/P["pub"])))
    bn = confirmed_boundary(classify(parse_log(LOGS/P["new"])))
    bp, bn = drop_redundant(bp), drop_redundant(bn)
    sg = P["sigma"]
    print(f"{P['tag']}: published {len(bp)} pts, verification {len(bn)} pts, sigma={sg}")
    xs,ys=trim(*gsmooth(P["saft"]["eps_c"],P["saft"]["eps_a"],sigma=50.0))
    xm,ym=trim(*gsmooth(P["mc"]["eps_c"],  P["mc"]["eps_a"],  sigma=50.0))
    ax.plot(xs,ys,color="#a8ddb5",zorder=1,label="SAFT")
    ax.plot(xm,ym,color="#ff7f0e",lw=2.0,ls=(0,(4,2,1.2,2)),zorder=2,label="MC simulation")
    ax.plot(*trim(*gsmooth(bp["eps_c"],bp["eps_a_boundary"],sigma=sg)),color="#1f77b4",zorder=3,
            label="SAFT-P published")
    ax.plot(*trim(*gsmooth(bn["eps_c"],bn["eps_a_boundary"],sigma=sg)),color="#d62728",zorder=4,
            ls=(0,(5,1.6)),label="SAFT-P verification")
    ax.plot(bp["eps_c"],bp["eps_a_boundary"],"o",ms=2.4,mfc="none",mew=0.7,color="#1f77b4",alpha=.75,zorder=3)
    ax.plot(bn["eps_c"],bn["eps_a_boundary"],"s",ms=2.9,mfc="none",mew=0.85,color="#d62728",alpha=.95,zorder=4)
    iax=ax.inset_axes((0.70,0.60,0.26,0.36),transform=ax.transAxes)
    iax.imshow(Image.open(FIGS_INSET/P["inset"]).convert("RGBA")); iax.set_axis_off(); iax.set_zorder(20)
    ax.set_xlabel(r"Non-directional interaction, $\epsilon_{nd}/k_{\mathrm{B}}T$")
    ax.set_ylabel(r"Directional interaction, $\epsilon_d/k_{\mathrm{B}}T$")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2)); ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both",top=True,right=True)
    ax.set_xlim(*P["xlim"]); ax.set_ylim(*P["ylim"])
    ax.set_title(P["title"],fontsize=8,loc="left",pad=3)
axes[0].legend(loc="lower left",bbox_to_anchor=(0.02,0.02))
fig.savefig(FIGS/"cube_spinodal_verification_panels.png",dpi=600)
fig.savefig(FIGS/"cube_spinodal_verification_panels.pdf")
plt.close(fig)

# --- the same panels, standalone, at the notebook's single-panel size --------
NAME={"000011":"cube_scan_stick3d_verification","111000":"cube_scan_planar3patch_verification"}
for P in PANELS:
    bp = P["pub_csv"] if P["pub_csv"] is not None else confirmed_boundary(classify(parse_log(LOGS/P["pub"])))
    bn = confirmed_boundary(classify(parse_log(LOGS/P["new"])))
    bp, bn = drop_redundant(bp), drop_redundant(bn); sg=P["sigma"]
    f2=plt.figure(figsize=(3.375,2.55),dpi=300,constrained_layout=True); a2=f2.add_subplot(111)
    a2.plot(*trim(*gsmooth(P["saft"]["eps_c"],P["saft"]["eps_a"],sigma=50.0)),color="#a8ddb5",zorder=1,label="SAFT")
    a2.plot(*trim(*gsmooth(P["mc"]["eps_c"],P["mc"]["eps_a"],sigma=50.0)),color="#ff7f0e",lw=2.0,
            ls=(0,(4,2,1.2,2)),zorder=2,label="MC simulation")
    a2.plot(*trim(*gsmooth(bp["eps_c"],bp["eps_a_boundary"],sigma=sg)),color="#1f77b4",zorder=3,label="SAFT-P published")
    a2.plot(*trim(*gsmooth(bn["eps_c"],bn["eps_a_boundary"],sigma=sg)),color="#d62728",zorder=4,
            ls=(0,(5,1.6)),label="SAFT-P verification")
    a2.plot(bp["eps_c"],bp["eps_a_boundary"],"o",ms=2.4,mfc="none",mew=0.7,color="#1f77b4",alpha=.75,zorder=3)
    a2.plot(bn["eps_c"],bn["eps_a_boundary"],"s",ms=2.9,mfc="none",mew=0.85,color="#d62728",alpha=.95,zorder=4)
    ia=a2.inset_axes((0.70,0.60,0.26,0.36),transform=a2.transAxes)
    ia.imshow(Image.open(FIGS_INSET/P["inset"]).convert("RGBA")); ia.set_axis_off(); ia.set_zorder(20)
    a2.set_xlabel(r"Non-directional interaction, $\epsilon_{nd}/k_{\mathrm{B}}T$")
    a2.set_ylabel(r"Directional interaction, $\epsilon_d/k_{\mathrm{B}}T$")
    a2.xaxis.set_minor_locator(AutoMinorLocator(2)); a2.yaxis.set_minor_locator(AutoMinorLocator(2))
    a2.tick_params(which="both",top=True,right=True)
    a2.set_xlim(*P["xlim"]); a2.set_ylim(*P["ylim"]); a2.legend(loc="lower left",bbox_to_anchor=(0.02,0.02))
    f2.savefig(FIGS/(NAME[P["tag"]]+".png"),dpi=600); f2.savefig(FIGS/(NAME[P["tag"]]+".pdf")); plt.close(f2)
    print("saved", NAME[P["tag"]])
print("saved")
