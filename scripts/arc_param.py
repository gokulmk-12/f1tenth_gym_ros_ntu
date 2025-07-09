import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def interpolate(waypoints):
    n = len(waypoints)
    pointvals = np.arange(n+1)
    waypoints = np.vstack((waypoints, waypoints[0, :].reshape(-1, 1).T))
    x_int = CubicSpline(pointvals, waypoints[:,0], bc_type='periodic')
    y_int = CubicSpline(pointvals, waypoints[:,1], bc_type='periodic')
    return x_int, y_int 

def getwaypoints(track):
    waypoints = np.genfromtxt(track, delimiter=',')
    return waypoints[:, :2]

def eval_raw(x_int, y_int, t):
    x_vals = x_int(t)
    y_vals = y_int(t)
    coords = np.array([x_vals, y_vals])
    return coords

def getangle_raw(x_int, y_int, t):
    der = eval_raw(x_int, y_int, t+0.1) - eval_raw(x_int, y_int, t)
    phi = np.arctan2(der[1], der[0])
    return phi

def fit_st(waypoints, x_int, y_int):
    nwp = len(waypoints)
    npoints = 20 * nwp

    tvals = np.linspace(0, nwp, npoints+1)
    coords = []
    for t in tvals:
        coords.append(eval_raw(x_int, y_int, t))
    coords = np.array(coords)

    dists = []
    dists.append(0)
    for idx in range(npoints):
        dists.append(np.sqrt(np.sum(np.square(coords[idx, :]-coords[np.mod(idx+1, npoints-1), :]))))
    dists = np.cumsum(np.array(dists))
    smax = dists[-1]

    npoints = 2 * 20 * nwp

    tvals = np.linspace(0, 2*nwp, npoints+1)

    coords = []
    for t in tvals:
        coords.append(eval_raw(x_int, y_int, np.mod(t, nwp)))
    coords = np.array(coords)

    distsr = []
    distsr.append(0)
    for idx in range(npoints):
        distsr.append(np.sqrt(np.sum(np.square(coords[idx, :] - coords[np.mod(idx+1, npoints-1), :]))))
    dists = np.cumsum(np.array(distsr))

    ts_inverse = CubicSpline(dists, tvals)
    svals = np.linspace(0, 2*smax, npoints)
    t_corr = ts_inverse(svals)

    return ts_inverse, smax

def generatelookuptable(track, r=0.1):
    track = os.path.join("/sim_ws/src/f1tenth_gym_ros/tracks/", f"{track}.csv")
    waypoints = getwaypoints(track)

    x_int, y_int = interpolate(waypoints)
    ts_inverse, smax = fit_st(waypoints, x_int, y_int)

    lutable_density = 100

    npoints = np.int32(np.floor(2 * smax * lutable_density))
    print(f"table generated with npoints: {npoints}")
    svals = np.linspace(0, 2*smax, npoints)
    tvals = ts_inverse(svals)

    names_table = ['sval', 'tval', 'xtrack', 'ytrack', 'phitrack', 'cos(phi)', 'sin(phi)', 'g_upper', 'g_lower']
    table = []
    for idx in range(npoints):
        track_point = eval_raw(x_int, y_int, tvals[idx])
        phi = getangle_raw(x_int, y_int, tvals[idx])
        n = [-np.sin(phi), np.cos(phi)]
        g_upper = r + track_point[0]*n[0] + track_point[1]*n[1]
        g_lower = -r + track_point[0]*n[0] + track_point[1]*n[1]
        table.append([svals[idx], tvals[idx], track_point[0], track_point[1], phi, np.cos(phi), np.sin(phi), g_upper, g_lower])
    
    table = np.array(table)
    plot_track(table)
    print("Variables stored in following order = ", names_table)
    np.savetxt(str(track)+'_lutab.csv', table, delimiter=', ')

    dict = {'smax': float(smax), 'ppm': lutable_density}
    with open(r''+track+'_params.yaml','w') as file:
        documents = yaml.dump(dict, file)
    return table, smax

def plot_track(table):
    downsampling = 20
    coords = table[:, 2:4]
    phis = table[::downsampling, 4]
    svals = table[::downsampling, 0]
    tvals = table[::downsampling, 1]
    cos_phi = table[::downsampling, 5]
    sin_phi = table[::downsampling, 6]
    gvals = table[::downsampling, 7]

    dists = []
    dists.append(0)
    npoints = len(coords)
    for idx in range(npoints-1):
        dists.append(np.sqrt(np.sum(np.square(coords[idx, :] - coords[np.mod(idx+1, npoints-1), :]))))
    dists = np.cumsum(np.array(dists))
    dists = dists[::downsampling]
    coords = coords[::downsampling]
    npoints = len(coords)

    plt.figure()
    plt.plot(svals, dists)
    plt.plot([0, svals[-1]], [0, svals[-1]])
    plt.xlabel("t (Bezier param corrected) [m]")
    plt.ylabel("s (approx. distance traveled) [m] ")
    plt.legend(["arclength vs t_corr","x=y"])

    plt.figure()
    len_indicator = 0.05
    downsampling = 10
    plt.plot(table[:,2], table[:,3])
    plt.scatter(table[::downsampling, 2], table[::downsampling, 3], marker='o')
    for idx in range(npoints):
        base = coords[idx, :]
        end = len_indicator * np.array([cos_phi[idx], sin_phi[idx]]) + base
        plt.plot([base[0], end[0]], [base[1], end[1]], color='r')
    plt.show()

if __name__ == "__main__":
    generatelookuptable("levine_closed")