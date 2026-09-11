"""
Routines to build and query a reusable static Green's function database
using fk.pl, structured to parallel CSI's existing EDKS static GF
We need to prior install FKs (Lupei Zhu code)
GF pipeline (Fault.py's edksGFs / EDKSmp.py).

"""

import os
import subprocess
import numpy as np
from shapely.ops import substring
from scipy.interpolate import RegularGridInterpolator

# The 9 static components fk.f writes, in this exact order. Confirmed
# directly from fk.f's static write statement:
#     write(*,'(f5.1,9e11.3)') x(ix), (real(sum(ix,l,1)), l=1,nCom)
# which reads from the same `sum` array, same index l, as the dynamic
# SAC-writing branch -- so the static line uses the same ordering as
# fk.f's own header comment: "Z0, R0, T0, Z1, ..." for n=0,1,2.
FK_DC_COMPONENTS = ['Z0', 'R0', 'T0', 'Z1', 'R1', 'T1', 'Z2', 'R2', 'T2']


def read_fk_model(model_file):
    """
    Cumulative depth (km) of each real interface in an fk-format model
    file -- everything except the surface and the bottom half-space.
 
    Only reads the thickness column (column 1), so this works
    regardless of whether the model's 3rd column is Vp or a Vp/Vs
    ratio.
 
    Args:
        model_file : path to the fk-format velocity model file.
 
    Returns:
        1D array of cumulative depths (km) at each layer boundary,
        excluding depth 0 and excluding the infinite bottom half-space.
    """
    thicknesses = []
    with open(model_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            thicknesses.append(float(line.split()[0]))
 
    cum_depth = np.cumsum(np.asarray(thicknesses, dtype=float))
 
    # the last layer's thickness is 0 (half-space) by convention, so
    # cum_depth[:-1] gives the depth of every real interface
    return cum_depth[:-1]


def nudge_depths_off_interfaces(depths, model_file, eps=1e-2):
    """
    Push any depth landing exactly on a velocity interface `eps` km
    deeper.
 
    fk.pl refuses to run a source placed exactly on an interface (its
    own vp[src_layer] != vp[src_layer-1] check exits(0) and prints
    "The source is located at a real interface" to stderr) -- exit
    code 0 means our returncode check never catches it, so it shows up
    downstream as an empty/short result instead of an obvious failure.
 
    Args:
        depths     : 1D array_like of requested source depths, km.
        model_file : path to the fk-format velocity model file.
        eps        : how far to nudge, km.
 
    Returns:
        1D array of depths, same length as input, with any
        interface-exact values nudged by eps. Nudges are printed so
        they don't happen invisibly.
    """
    depths = np.asarray(depths, dtype=float).copy()
    interfaces = read_fk_model(model_file)
 
    for i, d in enumerate(depths):
        if np.any(np.isclose(d, interfaces, atol=eps / 2)):
            print('nudging depth {} km -> {} km (sits on a model interface)'
                  .format(d, d + eps))
            depths[i] = d + eps
 
    return depths

def unit_moment(mu_pa, area_m2, slip_m=1.0):
    """
    Scalar seismic moment for a unit-slip point source, in the same
    pre-scaled convention fk.f's own Green's functions and syn.c's own
    -M handling use -- multiply this directly against dc_radiat's
    coefficients and fk's Z/R/T static outputs.
    It corresponds to the case 4 of FK 
        (magnitude/strike/dip/rake double-couple):
            m0 = pow(10.,1.5*m0+16.1-20);
    
 
    Takes mu and area in CSI's own native SI units (Pa, m^2) --
    matching fault.mu from Fault.py's setmu(model_file, tents=True,
    format='FK') and Areas from Patches2Sources once converted to m^2
    (Areas *= 1e6 from km^2, same conversion edksGFs itself applies).
 
    """
    M0_Nm = mu_pa * area_m2 * slip_m           # scalar moment, N*m
    M0_dyne_cm = M0_Nm * 1e7                   # 1 N*m = 1e7 dyne*cm
    Mw = (np.log10(M0_dyne_cm) - 16.1) / 1.5   # Kanamori (1977), inverted
    m0 = 10.0 ** (1.5 * Mw + 16.1 - 20.0)      # syn.c's own scaling, verbatim
    return m0,Mw


def build_fk_static_database(model_file, model_3rd_col, depths, distances, output_path,
                              fk_pl_path=None, fk_bin_dir=None,
                              nft=2, dk=0.1, kc=15.0, pmin=0.0, pmax=1.0,
                              src_type=2, verbose=True):
    """
    Build a reusable static Green's function database over a grid of
    source depths and distances, using fk.pl's static mode.


    Args:
        model_file  : path to the fk-format velocity model file
                      (columns: thickness, Vs, Vp_or_Vp/Vs, rho, Qs, Qp).
        model_3rd_col : None (or ''), 'k', or 'f' -- these map directly
                        onto fk.pl's own -M third field, and are NOT a
                        simple Vp-vs-ratio binary choice:
                          None / '' -- 3rd column of model_file is
                                       literal Vp, no transform applied.
                                       This is almost certainly what you
                                       want for a local/regional
                                       flat-earth model.
                          'k'       -- 3rd column is a Vp/Vs ratio, not Vp.
                          'f'       -- apply an earth-flattening
                                       transformation (spherical-Earth
                                       correction, only relevant at
                                       teleseismic distances).
        depths      : 1D array_like of source depths to tabulate, km.
                      Must be strictly increasing.
        distances   : 1D array_like of distances to tabulate, km.
                      Must be strictly increasing. Shared across all
                      depths.
        output_path : path to write the resulting .npz database to.
        fk_pl_path  : full path to fk.pl.
        fk_bin_dir  : directory containing fk / st_fk / trav. Passed
                      explicitly into the subprocess's PATH rather than
                      relying on the caller's shell already having it
                      set up, since fk.pl invokes those by bare name.

    Kwargs:
        nft      : 1 or 2. 
                    nt=1 will compute static displacements (require st_fk compiled).
                    nt=2 will compute static displacements using the dynamic solution.
        dk       : wavenumber sampling, in pi/x. fk.pl's own default is
                   0.3; we default (0.1).
        kc       : max wavenumber at zero frequency, in 1/h.
        pmin,pmax: phase velocity window, in 1/vs.
        src_type : 0=explosion, 1=single force, 2=double couple.
        verbose  : print progress per depth.

    Returns:
        None. Writes output_path as an .npz file with keys:
            depths, distances       : the input grids
            grn                     : (n_depth, n_dist, n_comp) array
            components              : component name per last axis
            model_file, src_type, nft, dk, kc, pmin, pmax : provenance
            units                   : the physical units of `grn`,
                                       straight from fk.f's header
    """
    if model_3rd_col not in (None, '', 'k', 'f'):
        raise ValueError(
            "model_3rd_col must be None/'' (literal Vp, no transform), "
            "'k' (3rd column is Vp/Vs ratio), or 'f' (earth-flattening -- "
            "a separate, unrelated choice from the ratio question). "
            "Got: {!r}".format(model_3rd_col))

    depths = np.asarray(depths, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if not np.all(np.diff(depths) > 0):
        raise ValueError('depths must be strictly increasing (required '
                          'for the regular-grid interpolation in stage 3)')
    if not np.all(np.diff(distances) > 0):
        raise ValueError('distances must be strictly increasing (required '
                          'for the regular-grid interpolation in stage 3)')
    
    # avoid the silent "source is located at a real interface" exit
    depths = nudge_depths_off_interfaces(depths, model_file)

    ndepth = len(depths)
    ndist = len(distances)
    ncomp = 9 if src_type == 2 else (3 if src_type == 1 else 1)

    # Allocate the full grid up front, filled in one depth at a time.
    # Left as NaN until written -- lets us catch any depth/distance
    # combination that silently failed to come back, instead of
    # shipping a database with a zero sitting where a real value
    # should be.
    grn = np.full((ndepth, ndist, ncomp), np.nan, dtype=float)

    env = dict(os.environ)
    if fk_bin_dir:
        env['PATH'] = fk_bin_dir + os.pathsep + env.get('PATH', '')
 
    fk_pl_path = 'fk.pl' if fk_pl_path is None else fk_pl_path


    for i, depth in enumerate(depths):

        dist_args = ['{:.4f}'.format(d) for d in distances]

        cmd = [
            fk_pl_path,
            '-M{}/{}/{}'.format(model_file, depth, model_3rd_col),
            '-N{}/1000/1/{}'.format(nft, dk),
            '-P{}/{}/{}'.format(pmin, pmax, kc),
            '-S{}'.format(src_type),
        ] + dist_args

        if verbose:
            print('[{}/{}] depth={} km'.format(i + 1, ndepth, depth))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                'fk.pl failed at depth={} km (exit {}):\n{}'.format(
                    depth, result.returncode, result.stderr))

        lines = [ln for ln in result.stdout.strip().split('\n') if ln.strip()]

        if len(lines) != ndist:
            raise RuntimeError(
                'fk.pl at depth={} km returned {} lines, expected {} (one '
                'per distance). Check that -N{} is actually static '
                '(nft<=2) and that fk/st_fk is reachable via fk_bin_dir.'
                .format(depth, len(lines), ndist, nft))

        for j, line in enumerate(lines):

            values = [float(v) for v in line.split()]

            got_dist = values[0]
            if abs(got_dist - distances[j]) > 1e-2:
                raise RuntimeError(
                    'Distance mismatch at depth={} km, line {}: expected '
                    '{} km, fk.pl reported {} km.'.format(
                        depth, j, distances[j], got_dist))

            grn[i, j, :] = values[1:1 + ncomp]

    if np.isnan(grn).any():
        raise RuntimeError('Database has unfilled entries after the build '
                            '-- something did not come back as expected.')

    np.savez(
        output_path,
        depths=depths,
        distances=distances,
        grn=grn,
        components=np.array(FK_DC_COMPONENTS if src_type == 2 else []),
        model_file=os.path.abspath(model_file),
        src_type=src_type,
        nft=nft, dk=dk, kc=kc, pmin=pmin, pmax=pmax,
        units=('1e-20 cm/(dyne*cm), per fk.f header comment '
               '(double-couple/explosion source)'
               if src_type in (0, 2) else
               '1e-15 cm/dyne, per fk.f header comment (single-force source)'),
    )

    if verbose:
        print('Saved database: {} depths x {} distances x {} components -> {}'
              .format(ndepth, ndist, ncomp, output_path))

def dc_radiat(stk, dip, rak):
    """
    Direct port of Lupei Zhu's dc_radiat() from radiats.c.
 
    Args:
        stk : station azimuth measured from the fault STRIKE, clockwise,
              degrees. This is (station_azimuth - strike), NOT the
              fault's strike itself -- matches how syn.c calls it:
              dc_radiat(az-mt[0][0], mt[0][1], mt[0][2], rad).
        dip, rak : dip and rake of the fault, degrees.
 
        All three can be scalars or arrays that broadcast together
        (e.g. an array of per-receiver `stk` values against a scalar
        dip/rak for one point source).
 
    Returns:
        rad : (..., 3, 3) array, rad[..., n, comp] for azimuthal order
              n=0,1,2 and component comp=0 (Z), 1 (R), 2 (T). Combines
              with fk's own Z0,R0,T0,Z1,R1,T1,Z2,R2,T2 static outputs:
 
                  Z = m0 * (rad[0,0]*Z0 + rad[1,0]*Z1 + rad[2,0]*Z2)
                  R = m0 * (rad[0,1]*R0 + rad[1,1]*R1 + rad[2,1]*R2)
                  T = m0 * (rad[0,2]*T0 + rad[1,2]*T1 + rad[2,2]*T2)
 
              Confirmed against syn.c's own combination loop: for
              i in 0..2 (azimuthal order), j in 0..2 (component),
              coef = m0*rad[i][j] multiplies the i-th triplet's j-th
              Green's function file, read in the sequence
              Z0,R0,T0,Z1,R1,T1,Z2,R2,T2.
    """
    stk = np.radians(np.asarray(stk, dtype=float))
    dip = np.radians(np.asarray(dip, dtype=float))
    rak = np.radians(np.asarray(rak, dtype=float))
    stk, dip, rak = np.broadcast_arrays(stk, dip, rak)
 
    sstk, cstk = np.sin(stk), np.cos(stk)
    sdip, cdip = np.sin(dip), np.cos(dip)
    srak, crak = np.sin(rak), np.cos(rak)
    sstk2, cstk2 = 2 * sstk * cstk, cstk * cstk - sstk * sstk
    sdip2, cdip2 = 2 * sdip * cdip, cdip * cdip - sdip * sdip
 
    rad = np.zeros(stk.shape + (3, 3))
 
    rad[..., 0, 0] = 0.5 * srak * sdip2
    rad[..., 0, 1] = rad[..., 0, 0]
    rad[..., 0, 2] = 0.0
 
    rad[..., 1, 0] = -sstk * srak * cdip2 + cstk * crak * cdip
    rad[..., 1, 1] = rad[..., 1, 0]
    rad[..., 1, 2] = cstk * srak * cdip2 + sstk * crak * cdip
 
    rad[..., 2, 0] = -sstk2 * crak * sdip - 0.5 * cstk2 * srak * sdip2
    rad[..., 2, 1] = rad[..., 2, 0]
    rad[..., 2, 2] = cstk2 * crak * sdip - 0.5 * sstk2 * srak * sdip2
 
    return rad

 
def rotate_rt_to_ne(R, T, az_deg):
    """
    Rotate radial/transverse static displacement to North/East.
 
    Confirmed against syn.c's own component-orientation assignment
    (cmpaz[1]=az for radial, cmpaz[2]=az+90 for transverse, both
    compass bearings clockwise from north, shared by every source
    type syn.c supports):
 
        N = R*cos(az) - T*sin(az)
        E = R*sin(az) + T*cos(az)
 
    Args:
        R, T   : radial / transverse static displacement, any matching
                 shape.
        az_deg : station azimuth (source -> receiver), degrees,
                 broadcastable against R/T.
 
    Returns:
        N, E : same shape as R/T.
    """
    az_rad = np.radians(az_deg)
    N = R * np.cos(az_rad) - T * np.sin(az_rad)
    E = R * np.sin(az_rad) + T * np.cos(az_rad)
    return N, E
 
 
def synthesize_static(db, xs, ys, zs, strike, dip, rake, mu_pa, area_km2, xr, yr):
    """
    Static Z (up), N, E displacement at every receiver from every point
    source, for ONE rake value (scalar, or one per source), with unit
    (1 m) slip on each source. NOT yet summed over point sources within
    a patch -- that grouping (by Ids, mirroring _edks_chunk_worker) is
    the caller's job, same division of responsibility edksGFs uses.
 
    Args:
        db         : FKGreenFunctionDB instance.
        xs, ys, zs : (n_source,) point source position, km. xs/ys are
                     local east/north (matching Patches2Sources' raw,
                     un-converted km output -- do NOT apply edksGFs's
                     *1000 meters conversion here).
        strike,dip : (n_source,) degrees (Patches2Sources returns
                     radians -- convert before calling this).
        rake       : scalar or (n_source,) degrees.
        mu_pa      : (n_source,) shear modulus, Pa -- from fault.mu,
                     via setmu(model_file, tents=True, format='FK').
        area_km2   : (n_source,) point source area, km^2 (raw
                     Patches2Sources output, unconverted).
        xr, yr     : (n_receiver,) receiver position, km.
 
    Returns:
        Z, N, E : each (n_source, n_receiver).
    """
    xs, ys, zs = (np.asarray(a, dtype=float) for a in (xs, ys, zs))
    strike, dip = np.asarray(strike, dtype=float), np.asarray(dip, dtype=float)
    rake = np.broadcast_to(np.asarray(rake, dtype=float), strike.shape)
    mu_pa = np.asarray(mu_pa, dtype=float)
    area_km2 = np.asarray(area_km2, dtype=float)
    xr, yr = np.asarray(xr, dtype=float), np.asarray(yr, dtype=float)
 
    # az: compass bearing (clockwise from north) FROM source TO
    # receiver, matching syn.c's own -A convention.
    dx = xr[None, :] - xs[:, None]   # east difference, km
    dy = yr[None, :] - ys[:, None]   # north difference, km
    distance = np.sqrt(dx**2 + dy**2)
    az = np.degrees(np.arctan2(dx, dy)) % 360.
 
    depth_grid = np.broadcast_to(zs[:, None], distance.shape)
    raw = db.lookup(depth_grid.ravel(), distance.ravel())
    raw = raw.reshape(distance.shape + (raw.shape[-1],))
    Z0, R0, T0, Z1, R1, T1, Z2, R2, T2 = (raw[..., k] for k in range(9))
 
    stk = az - strike[:, None]
    rad = dc_radiat(stk, dip[:, None], rake[:, None])
 
    m0 = unit_moment(mu_pa, area_km2 * 1e6, slip_m=1.0)   # km^2 -> m^2
 
    Z = m0[:, None] * (rad[..., 0, 0] * Z0 + rad[..., 1, 0] * Z1 + rad[..., 2, 0] * Z2)
    R = m0[:, None] * (rad[..., 0, 1] * R0 + rad[..., 1, 1] * R1 + rad[..., 2, 1] * R2)
    T = m0[:, None] * (rad[..., 0, 2] * T0 + rad[..., 1, 2] * T1 + rad[..., 2, 2] * T2)
 
    N, E = rotate_rt_to_ne(R, T, az)
 
    return Z, N, E


class FKGreenFunctionDB:
    """
    Wraps a saved fk static Green's function database (from
    build_fk_static_database) with a linear interpolator over
    (depth, distance) -- matching the interpolation method EDKS's own
    python-path uses (confirmed from Fault.py's edksGFs:
    inter.interpolate(..., method='linear', ...)).
 
    Unlike EDKS's per-point-source subprocess call to a Fortran binary,
    this whole lookup lives in memory: the interpolator is built once
    at load time and reused for every point source and every rake.
    """
 
    def __init__(self, npz_path, bounds_error=True):
 
        db = np.load(npz_path)
 
        self.depths = db['depths']
        self.distances = db['distances']
        self.grn = db['grn']              # (n_depth, n_dist, n_comp)
        self.components = list(db['components'])
        self.model_file = db['model_file'].item()
        self.src_type = int(db['src_type'].item())
        self.units = db['units'].item()
 
        db.close()
 
        # bounds_error=True on purpose: silently extrapolating a fault
        # patch deeper than the database covers, or a receiver farther
        # than it covers, should raise loudly rather than hand back a
        # quietly wrong number.
        self._interp = RegularGridInterpolator(
            (self.depths, self.distances), self.grn,
            method='linear', bounds_error=bounds_error)
 
    def lookup(self, depth, distance):
        """
        Interpolate the static Green's function components at
        arbitrary (depth, distance) points, both in km.
 
        Args:
            depth, distance : scalars, or 1D array_likes of equal
                               length (one (depth, distance) pair per
                               point source / receiver combination).
 
        Returns:
            array of shape (n_comp,) for scalar input, or
            (n, n_comp) for array input, in the order given by
            self.components (Z0, R0, T0, Z1, ...).
        """
        scalar_input = np.isscalar(depth)
        points = np.column_stack([np.atleast_1d(depth),
                                   np.atleast_1d(distance)])
        result = self._interp(points)
        return result[0] if scalar_input else result
    
if __name__ == '__main__':

    # Example only -- replace with your fault's actual depth extent and
    # your network's actual distance range before running for real.
    example_depths = np.arange(2.0, 40.0, 2.0)      # km
    example_distances = np.arange(5.0, 300.0, 5.0)  # km

    build_fk_static_database(
        model_file=os.path.expanduser('~/models/test.d'),
        model_3rd_col='k',
        depths=example_depths,
        distances=example_distances,
        output_path='fk_static_db.npz'
    )
