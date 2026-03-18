import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.linewidth": 1.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
})

G = 6.67430e-11
M_sun = 1.98847e30
R_sun = 6.957e8
M_jup = 1.89813e27
R_jup = 7.1492e7
k_B = 1.380649e-23
m_H = 1.6735575e-27

def separation_a(Mtot_Msun, P_min):
    P = P_min * 60.0
    return (G * (Mtot_Msun * M_sun) * P**2 / (4*np.pi**2))**(1/3)

def roche_L1(M1, M2):
    x1 = -M2
    x2 =  M1
    def dUdx(x):
        r1 = abs(x-x1)
        r2 = abs(x-x2)
        return x - M1*(x-x1)/r1**3 - M2*(x-x2)/r2**3
    return brentq(dUdx, x1+1e-6, x2-1e-6)

def potential(x, y, M1, M2, x1, x2):
    r1 = np.sqrt((x-x1)**2 + y**2)
    r2 = np.sqrt((x-x2)**2 + y**2)
    return -M1/r1 - M2/r2 - 0.5*(x**2 + y**2)

def accel(x, y, vx, vy, M1, M2, x1, x2):
    r1 = ((x-x1)**2 + y**2)**1.5
    r2 = ((x-x2)**2 + y**2)**1.5
    ax = -M1*(x-x1)/r1 - M2*(x-x2)/r2 + x + 2*vy
    ay = -M1*y/r1 - M2*y/r2 + y - 2*vx
    return ax, ay

def integrate_stream(x0, y0, vx0, vy0,
                     M1, M2, x1, x2, Racc,
                     stop_at_impact, dt=2e-3, tmax=12.0):

    x, y = x0, y0
    vx, vy = vx0, vy0
    xs, ys = [x], [y]
    impact = None
    min_d = 1e9

    for step in range(int(tmax/dt)):
        d = np.hypot(x-x1, y)
        min_d = min(min_d, d)

        if impact is None and d <= Racc:
            impact = (x, y, step, min_d)
            if stop_at_impact:
                break

        ax, ay = accel(x, y, vx, vy, M1, M2, x1, x2)
        vxh, vyh = vx + 0.5*dt*ax, vy + 0.5*dt*ay
        xn, yn = x + dt*vxh, y + dt*vyh
        axn, ayn = accel(xn, yn, vxh, vyh, M1, M2, x1, x2)
        vx, vy = vxh + 0.5*dt*axn, vyh + 0.5*dt*ayn
        x, y = xn, yn
        xs.append(x); ys.append(y)

        if x*x + y*y > 6:
            break

    return np.array(xs), np.array(ys), impact, min_d

def impact_longitude_offset_deg(impact_xy, x_acc):
    ix, iy = impact_xy
    dx, dy = ix - x_acc, iy
    return np.degrees(np.arctan2(dy, dx))

def make_figs(case_name, P_min,
              Macc_Msun, Mdon_Msun, Racc_Rsun,
              T_gas=2500.0, mu_gas=2.3,
              alpha=2.0, eta=0.5,
              n_family=200,
              dt=2e-3, tmax=12.0):

    q = Mdon_Msun / Macc_Msun
    M1 = 1/(1+q)
    M2 = q/(1+q)
    x1, x2 = -M2, M1

    a = separation_a(Macc_Msun + Mdon_Msun, P_min)
    Racc = (Racc_Rsun * R_sun) / a

    cs = np.sqrt(k_B * T_gas / (mu_gas * m_H))
    Omega = 2*np.pi / (P_min * 60.0)
    eps = cs / (Omega * a)

    xL1 = roche_L1(M1, M2)

    print(f"  eps = cs/(Omega a) = {eps:.4e}")
    print(f"  alpha = {alpha}, eta = {eta}")

    family = []

    for i in range(n_family):
        vx0 = -alpha * eps
        vy0 = np.clip(np.random.normal(0.0, eps), -3*eps, 3*eps)
        x0  = xL1 - eta * eps
        y0  = 0.0

        xs, ys, impact, min_d = integrate_stream(
            x0, y0, vx0, vy0,
            M1, M2, x1, x2, Racc,
            stop_at_impact=False,
            dt=dt, tmax=tmax
        )
        family.append((xs, ys, impact))

    xs_rep, ys_rep, impact_rep, _ = integrate_stream(
        xL1 - eta*eps, 0.0,
        -alpha*eps, 0.0,
        M1, M2, x1, x2, Racc,
        stop_at_impact=True,
        dt=dt, tmax=tmax
    )

    if impact_rep:
        lon = impact_longitude_offset_deg(impact_rep[:2], x1)
        phase = lon / 360.0
        print(f"  Δλ = {lon:+.1f} deg")
        print(f"  Δφ = φ_IP − φ_SP = {phase:+.2f}")

    UL1 = potential(xL1, 0.0, M1, M2, x1, x2)
    
    xlim = (-1.15, 1.10)
    ylim = (-1.05, 1.05)
    grid = 800
    xx = np.linspace(xlim[0], xlim[1], grid)
    yy = np.linspace(ylim[0], ylim[1], grid)
    X, Y = np.meshgrid(xx, yy)
    R1 = np.sqrt((X-x1)**2 + Y**2)
    R2 = np.sqrt((X-x2)**2 + Y**2)
    
    mask = (R1 < 0.02) | (R2 < 0.02)
    UU = np.full_like(X, np.nan)
    UU[~mask] = -M1/R1[~mask] - M2/R2[~mask] - 0.5*(X[~mask]**2 + Y[~mask]**2)

    theta = np.linspace(0, 2*np.pi, 600)
    substellar = (x1 + Racc, 0.0)

    def mask_to_no_donor_side(xs, ys):
        keep = xs <= xL1
        return xs[keep], ys[keep]

    def draw_base(ax):
        ax.contour(X, Y, UU, levels=[UL1], colors="k", linestyles="dashed", linewidths=1.8)
        ax.plot(x1 + Racc*np.cos(theta), Racc*np.sin(theta), 'k', lw=3)
        ax.plot([x1], [0], "ko", ms=6)
        ax.plot([x2], [0], "ko", ms=6)
        ax.plot([xL1], [0], "k.", ms=10)
        ax.text(x1, -0.06, "M dwarf", ha="center", va="top", fontsize=15)
        ax.text(x2, -0.06, "brown dwarf", ha="center", va="top", fontsize=15)
        ax.minorticks_on()
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("X / a", fontsize=20)
        ax.set_ylabel("Y / a", fontsize=20)
        ax.tick_params(which="both", direction="in", top=True, right=True)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    draw_base(ax)

    for xs, ys, _ in family:
        xs_m, ys_m = mask_to_no_donor_side(xs, ys)
        ax.plot(xs_m, ys_m, color='#b11226', alpha=0.25, lw=1.2)

    xs_m, ys_m = mask_to_no_donor_side(xs_rep, ys_rep)
    ax.plot(xs_m, ys_m, color='#b11226', lw=4)

    ax.plot(substellar[0], substellar[1], "ko", ms=7, zorder=6)
    ax.annotate(
        "Substellar\npoint",
        xy=substellar,
        xytext=(substellar[0] + 0.32, substellar[1] - 0.50),
        arrowprops=dict(arrowstyle="->", lw=1.8),
        ha="center", va="center", fontsize=16
    )

    if impact_rep:
        ix, iy = impact_rep[:2]
        ax.plot(ix, iy, "o", color="#1f77b4", ms=9, zorder=7)
        ax.annotate(
            "Impact point", color="#1f77b4",
            xy=(ix, iy),
            xytext=(substellar[0]+0.35, iy + 0.18),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f77b4"),
            ha="center", va="bottom", fontsize=16
        )
        ax.text(
            0.03, 0.97,
            rf"$\Delta\lambda = {lon:+.1f}^\circ$" + "\n" +
            rf"$\Delta\phi = \phi_{{\rm IP}}-\phi_{{\rm SP}} = {phase:+.2f}$",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=16
        )

    fig.tight_layout()
    fig.savefig(f"{case_name}_ballistic_A.png", dpi=300)
    fig.savefig(f"{case_name}_ballistic_A.pdf", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    draw_base(ax)

    for xs, ys, impact in family:
        if impact is not None:
            ix, iy, istep, _ = impact
            xs_plot = xs[:istep+1]
            ys_plot = ys[:istep+1]
        else:
            xs_plot = xs
            ys_plot = ys
        
        xs_m, ys_m = mask_to_no_donor_side(xs_plot, ys_plot)
        ax.plot(xs_m, ys_m, color='#b11226', alpha=0.25, lw=1.2)

    xs_m, ys_m = mask_to_no_donor_side(xs_rep, ys_rep)
    ax.plot(xs_m, ys_m, color='#b11226', lw=4)

    ax.plot(substellar[0], substellar[1], "ko", ms=7, zorder=6)
    ax.annotate(
        "Substellar\npoint",
        xy=substellar,
        xytext=(substellar[0] + 0.32, substellar[1] - 0.40),
        arrowprops=dict(arrowstyle="->", lw=1.8),
        ha="center", va="center", fontsize=18
    )

    if impact_rep:
        ix, iy = impact_rep[:2]
        ax.plot(ix, iy, "o", color="#1f77b4", ms=9, zorder=7)
        ax.annotate(
            "Impact point", color="#1f77b4",
            xy=(ix, iy),
            xytext=(substellar[0]+0.35, iy + 0.26),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f77b4"),
            ha="center", va="bottom", fontsize=18
        )
        ax.text(
            0.03, 0.97,
            rf"$\lambda_{{\rm IP}}-\lambda_{{\rm SP}} = {lon:+.1f}^\circ$" + "\n" +
            rf"$\Delta\phi = \phi_{{\rm IP}}-\phi_{{\rm SP}} = {phase:+.2f}$",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=16
        )

    fig.tight_layout()
    fig.savefig(f"{case_name}_ballistic_B.png", dpi=300)
    fig.savefig(f"{case_name}_ballistic_B.pdf", dpi=300)
    plt.close(fig)


make_figs(
    "67min",
    P_min=67.16,
    Macc_Msun=0.099,
    Mdon_Msun=0.0397,
    Racc_Rsun=0.119,
    T_gas=3000
)

make_figs(
    "",
    P_min=86.65,
    Macc_Msun=0.106, 
    Mdon_Msun=0.0288,
    Racc_Rsun=0.126,
    T_gas=3000
)
