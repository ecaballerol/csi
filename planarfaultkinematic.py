'''
A class that deals planar kinematic faults

Written by Z. Duputel, January 2014
'''

## Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
import scipy.signal as signal
from scipy.linalg import block_diag
import copy
import sys
import os
import shutil
import sacpy

## Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok

# Rectangular patches Fault class
from .planarfault import planarfault



class planarfaultkinematic(planarfault):

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None):
        '''
        Args:
            * name      : Name of the fault.
            * f_strike: strike angle in degrees (from North)
            * f_dip:    dip angle in degrees (from horizontal)
            * f_length: length of the fault (i.e., along strike)
            * f_width: width of the fault (i.e., along dip)        
            * utmzone   : UTM zone.
        '''
        
        # Parent class init
        super(planarfaultkinematic,self).__init__(name,
                                                  utmzone=utmzone,
                                                  ellps=ellps,
                                                  lon0=lon0,
                                                  lat0=lat0)

        # Hypocenter coordinates
        self.hypo_x   = None
        self.hypo_y   = None
        self.hypo_z   = None
        self.hypo_lon = None
        self.hypo_lat = None
                
        # Fault size
        self.f_length  = None
        self.f_width   = None
        self.f_nstrike = None
        self.f_ndip    = None
        self.f_strike  = None
        self.f_dip     = None
        
        # Patch objects
        self.patch = None
        self.grid  = None
        self.vr    = None
        self.tr    = None
        
        # All done
        return


    def getHypoToCenter(self, p, sd_dist=False):
        ''' 
        Get patch center coordinates from hypocenter
        Args:
            * p      : Patch number.
            * sd_dist: If true, will return along dip and along strike distances
        '''

        # Check strike/dip/hypo assigmement
        assert self.f_strike != None, 'Fault strike must be assigned'
        assert self.f_dip    != None, 'Fault dip    must be assigned'
        assert self.hypo_x   != None, 'Hypocenter   must be assigned'
        assert self.hypo_y   != None, 'Hypocenter   must be assigned'
        assert self.hypo_z   != None, 'Hypocenter   must be assigned'

        # Get center
        p_x, p_y, p_z = self.getcenter(self.patch[p])
        x = p_x - self.hypo_x
        y = p_y - self.hypo_y
        z = p_z - self.hypo_z

        # Along dip and along strike distance to hypocenter
        if sd_dist:
            dip_d = z / np.sin(self.f_dip)
            strike_d = x * np.sin(self.f_strike) + y * np.cos(self.f_strike)
            return dip_d, strike_d
        else:
            return x,y,z
            


    def setHypoXY(self,x,y, UTM=True):
        '''
        Set hypocenter attributes from x,y
        Outputs: East/West UTM/Lon coordinates, depth attributes
        Args:
            * x:   east  coordinates 
            * y:   north coordinates
            * UTM: default=True, x and y is in UTM coordinates (in km)
                   if    ==False x=lon and y=lat (in deg)
        '''
        
        # Check strike/dip assigmement
        assert self.f_strike != None, 'Fault strike must be assigned'
        assert self.f_dip    != None, 'Fault dip    must be assigned'

        # If UTM==False, convert x=lon/y=lat to UTM
        if not UTM:
            self.hypo_x,self.hypo_y = self.ll2xy(x,y)
        else:
            self.hypo_x = x
            self.hypo_y = y

        # Get distance from the fault trace axis (in km)
        dist_from_trace = (self.hypo_x-self.xi[0]) * np.cos(self.f_strike) - (self.hypo_y-self.yi[0]) * np.sin(self.f_strike)

        # Get depth on the fault
        self.hypo_z = dist_from_trace * np.tan(self.f_dip) + self.top
        
        # UTM to lat/lon conversion        
        self.hypo_lon,self.hypo_lat = self.xy2ll(self.hypo_x,self.hypo_y)

        # All done
        return
    
    def setMu(self,model_file,modelformat='CPS'):
        '''
        Set shear modulus values for seismic moment calculation
        from model_file:

        +------------------+--------------------------+
        | if format = 'CPS'|Thickness, Vp, Vs, Rho    |
        +==================+==========================+
        | if format = 'KK' |file from Kikuchi Kanamori|
        +------------------+--------------------------+

        Args:
            * model_file        : Input file

        Kwargs:
            * modelformat       : Format of the model file

        Returns:
            * None
        '''

        # Check modelformat
        assert modelformat == 'CPS' or modelformat == 'KK', 'Incorrect model format (CPS or KK)'
        
        # Read model file
        mu = []
        depth  = 0.
        depths = []
        L = open(model_file).readlines()
        hdr = L[:12]
        NL = 0
        H = []; Vp = []; Vs = []; Rho = []; Qp = []; Qs = []
        for l in L[12:]:
            if not len(l.strip().split()):
                break
            items = l.strip().split()
            H.append(float(items[0]))
            Vp.append(float(items[1]))
            Vs.append(float(items[2]))
            Rho.append(float(items[3]))
            Qp.append(float(items[4]))
            Qs.append(float(items[5]))
            mu.append(Vs[-1]*Vs[-1]*Rho[-1]*1.0e9)
            depths.append([depth,depth+H[-1]])
            depth += H[-1]
            NL += 1
                   
        Nd = len(depths)
        Np = len(self.patch)        
        # Set Mu for each patch
        self.mu = np.zeros((Np,))
        for p in range(Np):
            p_x, p_y, p_z, width, length, strike_rad, dip_rad = self.getpatchgeometry(p,center=True)
            for d in range(Nd):
                if p_z>=depths[d][0] and p_z<depths[d][1]:
                    self.mu[p] = mu[d]

        # All done
        return
        
    def buildFault(self, lon, lat, dep, f_strike, f_dip, f_length, f_width, grid_size, p_nstrike, p_ndip,leading='strike'):
        '''
        Build fault patches/grid
        Args:
            * lat,lon,dep:  coordinates at the center of the top edge of the fault (in degrees)
            * strike:       strike angle in degrees (from North)
            * dip:          dip angle in degrees (from horizontal)
            * f_length: Fault length, km
            * f_width:  Fault width, km
            * grid_size:    Spacing between point sources within each patch
            * p_nstrike:   Number of subgrid points per patch along strike (multiple pts src per patch)
            * p_ndip:      Number of subgrid points per patch along dip    (multiple pts src per patch)
        '''
        
        # Orientation
        self.f_strike = f_strike * np.pi/180.
        self.f_dip    = f_dip    * np.pi/180.

        # Patch size = nb of pts along dip/strike * spacing
        patch_length  = grid_size * p_nstrike
        patch_width   = grid_size * p_ndip

        # Number of patches along strike / along dip
        self.f_nstrike = int(np.round(f_length/patch_length))
        self.f_ndip    = int(np.round(f_width/patch_width))

        # Correct the fault size to match n_strike and n_dip
        self.f_length = self.f_nstrike * patch_length
        self.f_width  = self.f_ndip    * patch_width
        if self.f_length != f_length or self.f_width != f_width:
            sys.stderr.write('!!! Fault size changed to %.2f x %.2f km'%(self.f_length,self.f_width))

                    
        # build patches
        self.buildPatches(lon, lat, dep, f_strike, f_dip, self.f_length, self.f_width, self.f_nstrike, self.f_ndip,leading)
        
        # build subgrid
        self.buildSubGrid(grid_size,p_nstrike,p_ndip)

        # All done
        return

        
    
    def buildSubGrid(self,grid_size,nbp_strike,nbp_dip):
        '''
        Define a subgrid of point sources on the fault (multiple point src per patches)
        Args: 
            * grid_size:    Spacing between point sources within each patch
            * p_nstrike:   Number of subgrid points per patch along strike 
            * p_ndip:      Number of subgrid points per patch along dip            
        '''
        
        # Check prescribed assigments
        assert self.f_strike != None, 'Fault strike must be assigned'
        assert self.f_dip    != None, 'Fault dip must be assigned'
        assert self.patch    != None, 'Patch objects must be assigned'
        
        dipdir = (self.f_strike+np.pi/2.)%(2.*np.pi)
        
        # Loop over patches
        self.grid = []
        for p in range(len(self.patch)):
            # Get patch location/size
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)

            # Set grid points coordinates on fault
            grid_strike = np.arange(0.5*grid_size,p_length,grid_size) - p_length/2.
            grid_dip    = np.arange(0.5*grid_size,p_width, grid_size) - p_width/2.

            # Check that everything is correct
            assert np.round(p_strike,2) == np.round(self.f_strike,2), 'Fault must be planar' 
            assert np.round(p_dip,2)    == np.round(self.f_dip,2)   , 'Fault must be planar' 
            assert nbp_strike == len(grid_strike), 'Incorrect length for patch %d'%(p)
            assert nbp_dip    == len(grid_dip),    'Incorrect width for patch  %d'%(p)

            # Get grid points coordinates in UTM  
            xt = p_x + grid_strike * np.sin(self.f_strike)
            yt = p_y + grid_strike * np.cos(self.f_strike)
            zt = p_z * np.ones(xt.shape)
            g  = []
            for i in range(nbp_dip):
                x = xt + grid_dip[i] * np.cos(self.f_dip) * np.sin(dipdir)
                y = yt + grid_dip[i] * np.cos(self.f_dip) * np.cos(dipdir)
                z = zt + grid_dip[i] * np.sin(self.f_dip)
                for j in range(x.size):                    
                    g.append([x[j],y[j],z[j]])
            self.grid.append(g)
                
        # All done
        return
    
    # ----------------------------------------------------------------------
    def setFaultMap(self,Nstrike,Ndip,leading='strike',check_depth=True):
        '''
        Set along dip and along strike indexing for patches

        Args:
            * Nstrike       : number of patches along strike
            * Ndip          : number of patches along dip

        Kwargs:
            * leading       : leadinf index of self.patch (can be 'strike' or 'dip')
            * check_depth   : CHeck patch depths and indexes are consistent

        Returns:
            * None
        '''

        # Check input parameters
        if leading=='strike':
            Nx=Nstrike
            Ny=Ndip
        else:
            Nx=Ndip
            Ny=Nstrike
        assert Nx*Ny==len(self.patch), 'Incorrect Nstrike and Ndip'
        
        # Loop over patches
        self.fault_map = []
        self.fault_inv_map = np.zeros((Nstrike,Ndip),dtype='int')
        for ny in range(Ny):
            for nx in range(Nx):
                p = ny * Nx + nx
                if leading=='strike':
                    self.fault_map.append([nx,ny])
                    self.fault_inv_map[nx,ny] = p
                elif leading=='dip':
                    self.fault_map.append([ny,nx])
                    self.fault_inv_map[ny,nx] = p
        self.fault_map = np.array(self.fault_map)
        
        for n in range(Ndip):
            i = np.where(self.fault_map[:,1]==n)[0]
            assert len(i)==Nstrike, 'Mapping error'

        for n in range(Nstrike):
            i = np.where(self.fault_map[:,0]==n)[0]
            assert len(i)==Ndip, 'Mapping error'

        if check_depth:
            for n in range(Ndip):
                indexes = np.where(self.fault_map[:,1]==n)[0]
                flag = True
                for i in indexes:
                    x,y,z = self.getcenter(self.patch[i])
                    if flag:
                        depth = np.round(z,1)
                        flag  = False
                    assert depth==np.round(z,1), 'Mapping error: inconsistent depth'

        # All done
        return


    def buildKinGFs(self, data, Mu, rake, slip=1., rise_time=2., stf_type='triangle', 
                    rfile_name=None, out_type='D', filter_coef=None, verbose=True, ofd=sys.stdout, efd=sys.stderr):
        '''
        Build Kinematic Green's functions based on the discretized fault. Green's functions will be calculated 
        for a given shear modulus and a given slip (cf., slip) along a given rake angle (cf., rake)
        Args:
            * data: Seismic data object
            * Mu:   Shear modulus
            * rake: Rake used to compute Green's functions
            * slip: Slip amplitude used to compute Green's functions (in m)
            * rise_time:  Duration of the STF in each patch
            * stf_type:   Type of STF pulse
            * rfile_name: User specified stf file name if stf_type='rfile'
            * out_type:   'D' for displacement, 'V' for velocity, 'A' for acceleration
            * filter_coef   : Array or dictionnary of second-order filter coefficients (optional), see scipy.signal.sosfilt
            * verbose:    True or False
        '''

        # Check the Waveform Engine
        assert self.patch != None, 'Patch object should be assigned'

        # Verbose on/off        
        if verbose:
            import sys
            print ("Building Green's functions for the data set {} of type {}".format(data.name, data.dtype))
            print ("Using waveform engine: {}".format(data.waveform_engine.name))
        
        
        # Loop over each patch
        Np = len(self.patch)
        rad2deg = 180./np.pi
        if data.name not in self.G:
            self.G[data.name] = {}
        self.G[data.name][rake] = []
        G = self.G[data.name][rake]
        for p in range(Np):
            if verbose:
                sys.stdout.write('\r Patch: {} / {} '.format(p+1,Np))
                sys.stdout.flush()  

            # Get point source location and patch geometry
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)
            src_loc = [p_x, p_y, p_z]

            # Angles in degree
            p_strike_deg = p_strike * rad2deg
            p_dip_deg    = p_dip    * rad2deg
            if isinstance (Mu,np.ndarray):
                Mupatch = Mu[p]
            else:
                Mupatch = Mu
            # Seismic moment
            M0 = Mupatch * slip * p_width * p_length * 1.0e13 # M0 assuming 1m slip
            
            # Compute Green's functions using data waveform engine
            data.calcSynthetics('GF_tmp',p_strike_deg,p_dip_deg,rake,M0,rise_time,stf_type,rfile_name,
                                out_type,src_loc,cleanup=True,ofd=ofd,efd=efd)
        
            # GFs filtering
            if filter_coef is not None:
                sos = filter_coef
                statmp = copy.deepcopy(data.waveform_engine.synth)
                for ist in statmp.keys():
                   for ic in statmp[ist]:
                       print(ist,ic)
                       statmp[ist][ic].depvar = signal.sosfilt(sos,statmp[ist][ic].depvar)
                # Assemble GFs
                G.append(statmp)
            else:
                # Assemble GFs
                G.append(copy.deepcopy(data.waveform_engine.synth))
                
            
            
        sys.stdout.write('\n')

        # All done
        return

    def buildKinDataTriangleMRF(self, data, eik_solver, Mu, rake_para=0., out_type='D', 
                                verbose=True, ofd=sys.stdout, efd=sys.stderr):
        '''
        Build Kinematic Green's functions based on the discretized fault. Green's functions will be calculated 
        for a given shear modulus and a given slip (cf., slip) along a given rake angle (cf., rake)
        Args:
            * data: Seismic data object
            * eik_solver: eikonal solver
            * Mu:   Shear modulus
            * rake_para: Rake of the slip parallel component in deg (default=0. deg)
            * out_type:   'D' for displacement, 'V' for velocity, 'A' for acceleration (default='D')
            * verbose:    True or False (default=True)

        WARNING: ONLY VALID FOR HOMOGENEOUS RUPTURE VELOCITY

        '''

        # Check the Waveform Engine
        assert self.patch  != None, 'Patch object must be assigned'
        assert self.hypo_x != None, 'Hypocenter location must be assigned'
        assert self.hypo_y != None, 'Hypocenter location must be assigned'
        assert self.hypo_z != None, 'Hypocenter location must be assigned'
        assert self.slip   != None, 'Slip values must be assigned'
        assert self.vr     != None, 'Rupture velocities must be assigned'
        assert self.tr     != None, 'Rise times must be assigned'
        assert len(self.patch)==len(self.slip)==len(self.vr)==len(self.tr), 'Patch attributes must have same length'
        
        # Verbose on/off        
        if verbose:
            import sys
            print ("Building predictions for the data set {} of type {}".format(data.name, data.dtype))
            print ("Using waveform engine: {}".format(data.waveform_engine.name))

        # Max duration
        max_dur = np.sqrt(self.f_length*self.f_length + self.f_width*self.f_width)/np.min(self.vr)
        Nt      = np.ceil(max_dur/data.waveform_engine.delta)

        # Calculate timings using eikonal solver
        print('-- Compute rupture front')
        eik_solver.setGridFromFault(self,0.3)
        eik_solver.fastSweep()

        # Loop over each patch
        print('-- Compute and sum-up synthetics')
        Np = len(self.patch)
        rad2deg = 180./np.pi
        if not self.d.has_key(data.name):
            self.d[data.name] = {}
        D = self.d[data.name]   
        for p in range(Np):
            if verbose:
                sys.stdout.write('\r Patch: {} / {} '.format(p+1,Np))
                sys.stdout.flush()  

            # Get point source location and patch geometry
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)
            src_loc = [p_x, p_y, p_z] 

            # Angles in degree
            p_strike_deg = p_strike * rad2deg
            p_dip_deg    = p_dip    * rad2deg
            
            # Total slip
            s_para = self.slip[p][0]
            s_perp = self.slip[p][1]
            total_slip = np.sqrt(s_para*s_para + s_perp*s_perp)

            # Compute Rake
            rake = rake_para + np.arctan2(s_perp,s_para)*rad2deg
            
            # Seismic moment
            M0  = Mu * total_slip * p_width * p_length * 1.0e13 # M0 assuming 1m slip
            
            # Moment rate function
            rfile = 'rfile.p%03d'%(p)
            MRF = np.zeros((Nt,),dtype='np.float64')
            t   = np.arange(Nt,dtype='np.float64')*data.waveform_engine.delta
            hTr = 0.5 * self.tr[p]
            for g in range(len(self.grid[p])):
                g_t0 = eik_solver.getT0FromFault(self,self.grid[p][g][0],self.grid[p][g][1],
                                                 self.grid[p][g][2])
                g_tc = g_t0 + hTr
                g_t1 = g_t0 + 2*hTr
                g_i  = np.where((t>=g_t0)*(t<=g_t1))
                MRF[g_i] += (1.0 - np.abs(t[g_i]-g_tc)/hTr)*(1.0/hTr)/len(self.grid[p])
            data.waveform_engine.writeRfile(rfile,MRF)
            rfile = os.path.abspath(rfile)
            # Compute Green's functions using data waveform engine
            data.calcSynthetics('GF_tmp',p_strike_deg,p_dip_deg,rake,M0,None,'rfile',rfile,
                                out_type,src_loc,cleanup=True,ofd=ofd,efd=efd)
                        
            
            # Assemble GFs
            for stat in data.sta_name:
                if not D.has_key(stat):
                    D[stat] = copy.deepcopy(data.waveform_engine.synth[stat])
                else:
                    for c in data.waveform_engine.synth[stat].keys():
                        D[stat][c].depvar += data.waveform_engine.synth[stat][c].depvar
        sys.stdout.write('\n')
        print('-- Done')
        
        # All done
        return

    def creaWav(self,data,include_G=True,include_d=True):
        '''
        Create a list of Waveform dictionaries
        Args:
            * data: Data object 
            * include_G: if True, include G (default=True)
            * include_d: if True, include d (default=True)
        '''
        # Create a list of waveform dictionaries
        Wav = []
        if include_G==True:
            assert self.G.has_key(data.name), 'G must be implemented for {}'.format(data.name)
            for r in self.G[data.name].keys():
                for p in range(len(self.patch)):
                    Wav.append(self.G[data.name][r][p])
        if include_d==True:
            assert self.d.has_key(data.name), 'd must be implemented for {}'.format(data.name)
            Wav.append(self.d[data.name])
        
        # All done
        return Wav

    def trim(self,data,mint,maxt,trim_G=True,trim_d=True):
        '''
        Waveform windowing
        Args:
            * data: Data object 
            * mint: Minimum time
            * maxt: Maximum time
            * trim_G: if True, trim G (default=True)
            * trim_d: if True, trim d (default=True)
        '''

        # Create waveform dictionary list
        Wav = self.creaWav(data,include_G=trim_G,include_d=trim_d)

        # Trim waveforms
        for w in Wav:
            for s in data.sta_name:
                for c in w[s].keys():
                    t = np.arange(w[s][c].npts,dtype='np.float64') * w[s][c].delta + w[s][c].o + w[s][c].b
                    ta = np.abs(t-mint)
                    tb = np.abs(t-maxt)
                    ita = np.where(ta==ta.min())[0][0]
                    itb = np.where(tb==tb.min())[0][0]
                    w[s][c].b      = t[ita]- w[s][c].o
                    w[s][c].depvar = w[s][c].depvar[ita:itb+1]
                    w[s][c].npts   = len(w[s][c].depvar)

        # All done
        return

    def filter(self,data,a,b,filtFunc,mean_npts=None,filter_G=True,filter_d=True):
        '''
        Waveform filtering
        Args:
            * data: Data object 
            * a: numerator polynomial of the IIR filter
            * b: denominator polynomial of the IIR filter
            * filtFunc: filter function
            * mean_npts: remove mean over the leading mean_npts points (default=None)
            * filter_G: if True, filter G (default=True)
            * filter_d: if True, filter d (default=True)        
        '''
        # Create waveform dictionary list
        Wav = self.creaWav(data,include_G=filter_G,include_d=filter_d)

        # Trim waveforms
        for w in Wav:
            for s in data.sta_name:
                for c in w[s].keys():
                    if mean_npts != None:
                        w[s][c].depvar -= np.mean(w[s][c].depvar[:mean_npts])
                    w[s][c].depvar = filtFunc(b,a,w[s][c].depvar)

        # All done
        return      


    def GFwindow(self,data):
        '''
        Function to window the GFs from fault, according to 
        data
        data:  Seismic object 
        '''  
        print ("Windowing GFs according to data {}"\
               .format(data.name))
        
        assert data.name in self.G.keys(), 'GFs not build for this dataset'

        G = self.G[data.name]
        for irake in G.keys():
            for ip in range(len(self.patch)):
                for ista in G[irake][ip]:
                    
                    o_sac = G[irake][ip][ista]
                    b = data.d[ista].b - data.d[ista].o
                    npts = data.d[ista].npts
                    t = np.arange(o_sac.npts)*o_sac.delta+o_sac.b-o_sac.o
                    dtb = np.absolute(t-b)
                    ib  = np.where(dtb==dtb.min())[0][0]
                    assert np.absolute(dtb[ib])<o_sac.delta,'Incomplete GFs'                
                    o_sac.depvar = o_sac.depvar[ib:ib+npts]
                    # Sac headers
                    o_sac.kstnm  = data.d[ista].kstnm
                    o_sac.kcmpnm = data.d[ista].kcmpnm
                    o_sac.knetwk = data.d[ista].knetwk
                    o_sac.khole  = data.d[ista].khole
                    o_sac.stlo   = data.d[ista].stlo
                    o_sac.stla   = data.d[ista].stla
                    o_sac.npts   = npts
                    o_sac.b      = t[ib]+o_sac.o

        return
    
    def setBigDmap(self,seismic_data):
            '''
            Assign data_idx map for kinematic data

            Args:
                * seismic_data      : Data to take care of

            Returns:
                * None
            '''
            if type(seismic_data) != list:
                data_list = [seismic_data]
            else:
                data_list = seismic_data
                
            # Set the data index map
            d1 = 0
            d2 = 0        
            self.bigD_map = {}
            for data in data_list:
                for dkey in data.sta_name:
                    d2 += data.d[dkey].npts
                self.bigD_map[data.name]=[d1,d2]
                d1 = d2
            # All done
            return
    # ----------------------------------------------------------------------
    def buildBigGD(self,eik_solver,seismic_data,rakes,vmax,Nt,Dt,
                        rakes_key=None,dtype='np.float64',
                        fastsweep=False,indexing='Altar'):
        '''
        Build BigG and bigD matrices from Green's functions and data dictionaries

        Args:
            * eik_solver: Eikonal solver (e.g., FastSweep or None)
            * data:       Seismic data object or list of objects
            * rakes:      List of rake angles
            * vmax:       Maximum rupture velocity
            * Nt:         Number of rupture time-steps
            * Dt:         Rupture time-steps

        Kwargs:
            * rakes_key:  If GFs are stored under different keywords than rake value, provide them here
            * fastsweep:  If True and vmax is set, solves min arrival time using fastsweep algo. If false, uses analytical solution.

        Returns:
            * tmin:       Array of ???
        '''

        if type(seismic_data) != list:
            data_list = [seismic_data]
        else:
            data_list = seismic_data
           
        # set rake keywords for dictionnary
        if rakes_key is None:
            rakes_key = rakes
         
        # Set eikonal solver grid for vmax
        Np = len(self.patch)
        if vmax != np.inf and vmax > 0.:
            vr = copy.deepcopy(self.vr)
            self.vr[:] = vmax 
            if fastsweep and (eik_solver is not None): # Uses fastsweep
                eik_solver.setGridFromFault(self,1.0)
                eik_solver.fastSweep()
                self.vr[:] = copy.deepcopy(vr)
        
                # Get tmin for each patch
                tmin = []
                for p in range(Np):
                    # Location at the patch center
                    dip_c, strike_c = self.getHypoToCenter(p,True)
                    tmin.append(eik_solver.getT0([dip_c],[strike_c])[0])

            else: # Uses analytical solution
                # Get tmin for each patch
                tmin = []
                for p in range(Np):
                    dip_c, strike_c = self.getHypoToCenter(p,True)
                    tmin.append(np.sqrt(dip_c**2+strike_c**2)/vmax)                    
        else:
            tmin = np.zeros((Np,))

        
        # Build up bigD
        self.bigD = []
        for data in data_list:
            for dkey in data.sta_name:
                self.bigD.extend(data.d[dkey].depvar)
        self.bigD = np.array(self.bigD)
        
        # Build Big G matrix
        self.bigG = np.zeros((len(self.bigD),Nt*Np*len(rakes_key)))
        j  = 0
        if indexing == 'Altar':
            for nt in range(Nt):
                #print('Processing %d'%(nt))
                for r in rakes_key:
                    for p in range(Np):                    
                        di = 0
                        for data in data_list:
                            for dkey in data.sta_name:
                                if isinstance (self.G[data.name][r][p][dkey],sacpy.sac.Sac):
                                    depvar = self.G[data.name][r][p][dkey].depvar
                                    npts   = self.G[data.name][r][p][dkey].npts
                                    delta  = self.G[data.name][r][p][dkey].delta
                                    its = int(np.round((tmin[p] + nt * Dt)/delta,0)) 
                                    i = its + di     
                                    l = npts - its
                                    if l>0:
                                        self.bigG[i:i+l,j] = depvar[:l]                        
                                    di += npts
                        j += 1

        return tmin
    def buildBigCd(self,seismic_data):
            '''
            Assemble Cd from multiple kinematic datasets

            Args:
                * seismic_data      : Data to take care of

            Returns:
                * None
            '''
            assert self.bigD is not None, 'bigD must be assigned'
            assert self.bigD_map is not None, 'bigD_map must be assigned (use setbigDmap)'
            self.bigCd = np.zeros((self.bigD.size,self.bigD.size))

            if type(seismic_data) != list:
                data_list = [seismic_data]
            else:
                data_list = seismic_data
                
            for data in data_list:
                i = self.bigD_map[data.name]
                self.bigCd[i[0]:i[1],i[0]:i[1]] = data.Cd
            # All done return
            return
    
    def saveKinGFs(self, data, o_dir='gf_kin'):
        '''
        Writing Green's functions (1 sac file per channel per patch for each rake)
        Args:
            data  : Data object corresponding to the Green's function to be saved
            o_dir : Output directory name
        '''
        
        # Print stuff
        print('Writing Kinematic Greens functions in directory {} for fault {} and dataset {}'.format(o_dir,self.name,data.name))

        # Write Green's functions
        G = self.G[data.name]
        for r in G: # Slip direction: Rake (integer)
            for p in range(len(self.patch)): # Patch number (integer)
                for s in G[r][p]: # station name (string)
                    for c in G[r][p][s]: # component name (string)
                        o_file = os.path.join(o_dir,'gf_rake%d_patch%d_%s_%s.sac'%(r,p,s,c))
                        G[r][p][s][c].write(o_file)
        
        # Write list of stations
        f = open(os.path.join(o_dir,'stat_list'),'w')
        for s in G[r][p]:
            f.write('%s\n'%(s))
        f.close()

        # All done
        return

    def loadKinGFs(self, data, rake=[0,90],i_dir='gf_kin',station_file=None,components='all'):
        '''
        Reading Green's functions (1 sac file per channel per patch for each rake)
        Args:
            data       : Data object corresponding to the Green's function to be loaded
            rake       : List of rake values (default=0 and 90 deg)
            i_dir      : Output directory name (default='gf_kin')
            station_file: read station list from 'station_file'
            components : Decide whether or not load all the components, 
                        if not, it reads the component and name from data.d (use for real data)
        '''
        
        # Import sac for python (ask Zach)
        import sacpy

        # Print stuff
        print('Loading Kinematic Greens functions from directory {} for fault {} and dataset {}'.format(i_dir,self.name,data.name))

        # Init G
        self.G[data.name] = {}
        G = self.G[data.name]
        
        # Read list of station names
        if station_file != None:
            sta_name = []
            f = open(os.path.join(o_dir,'stat_list'),'r')
            for l in f:
                sta_name.append(l.strip().split()[0])
            f.close()
        else:
            sta_name = data.sta_name

        # Read Green's functions
        for r in rake: # Slip direction: Rake (integer)
            G[r] = []
            for p in range(len(self.patch)): # Patch number (integer)
                G[r].append({})
                for s in sta_name: # station name (string)
                    G[r][p][s] = {}
                    if components=='all':
                        for c in ['Z','N','E']: # component name (string)
                            i_file = os.path.join(i_dir,'gf_rake%d_patch%d_%s_%s.sac'%(r,p,s,c))
                            if os.path.exists(i_file):                            
                                G[r][p][s][c] = sacpy.Sac()
                                G[r][p][s][c].read(i_file)
                            else:
                                print('Skipping GF for {} {}'.format(s,c))
                    elif components=='individual':
                        c = data.d[s].kcmpnm[-1]
                        sGF= data.d[s].kstnm
                        i_file = os.path.join(i_dir,'gf_rake%d_patch%d_%s_%s.sac'%(r,p,sGF,c))
                        if os.path.exists(i_file):                            
                            G[r][p][s] = sacpy.Sac()
                            G[r][p][s].read(i_file)
                        else:
                            print('Skipping GF for {} {}'.format(s,c))

        # All done
        return

    def saveKinData(self, data, o_dir='data_kin'):
        '''
        Write Data (1 sac file per channel)
        Args:
            data  : Data object corresponding to the Green's function to be saved
            o_dir : Output file name
        '''

        # Print stuff
        print('Writing Kinematic Data to file {} for fault {} and dataset {}'.format(o_dir,self.name,data.name))

        # Write data in sac file
        d = self.d[data.name]
        f = open(os.path.join(o_dir,'stat_list'),'w') # List of stations
        for s in d: # station name (string)
            f.write('%s\n'%(s))
            for c in d[s]: # component name (string)
                o_file = os.path.join(o_dir,'data_%s_%s.sac'%(s,c))
                d[s][c].write(o_file)
        f.close()
        
        # All done
        return

    def loadKinData(self, data, i_dir='data_kin', station_file=None):
        '''
        Read Data (1 sac file per channel)
        Args:
            data  : Data object corresponding to the Green's function to be loaded
            i_dir : Input directory
            station_file: read station list from 'station_file'
        '''

        # Import sac for python (ask Zach)
        import sacpy

        # Print stuff
        print('Loading Kinematic Data from directory {} for fault {} and dataset {}'.format(i_dir,self.name,data.name))

        # Check list of station names
        if station_file != None:
            sta_name = []
            f = open(os.path.join(o_dir,'stat_list'),'r')
            for l in f:
                sta_name.append(l.strip().split()[0])
            f.close()
        else:
            sta_name = data.sta_name

        # Read data from sac files
        self.d[data.name] = {}
        d = self.d[data.name]
        for s in sta_name: # station name (string)
            d[s]={}
            for c in ['Z','N','E']:      # component name (string)
                o_file = os.path.join(i_dir,'data_%s_%s.sac'%(s,c))
                if os.path.exists(o_file):
                    d[s][c] = sacpy.sac()
                    d[s][c].read(o_file)
                else:
                    print('Skipping Data for {} {}'.format(s,c))                    
        # All done
        return
    
    def writeFault2File(self, filename):
        '''
        Write the patch center coordinates in an ascii file with the format use 
        in RectangularPatchesKin
        the file format is so that it can by used directly in psxyz (GMT).

        Args:
            * filename      : Name of the file.

        Kwargs:
            * slip          : Put the slip as a value for the color. Can be None, strikeslip, dipslip, total, coupling
            * scale         : Multiply the slip value by a factor.

        Retunrs:
            * None
        '''

        # Write something
        print('Writing geometry to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        #Read Header
        fout.write('#lon lat E[km] N[km] Dep[km] strike dip Area ID \n')
        # Loop over the patches
        nPatches = len(self.patch)
        for patch in self.patch:

            # Get patch index
            pIndex = self.getindex(patch)

            # Get patch center
            xc, yc, zc = self.getcenter(patch)
            lonc, latc = self.xy2ll(xc, yc)

            # Write the string to file
            fout.write('{0:5.3f} {1:5.3f} {2:5.3f} {3:5.3f} {4:5.2f} {5:5.2f} {6:5.2f} {7:5.2f} {8:5.2f} \n'.format(lonc, latc,xc, yc, zc, np.rad2deg(self.f_strike),np.rad2deg(self.f_dip),self.area[pIndex],pIndex))

        # Close the file
        fout.close()

    def saveBigGD(self, bigDfile='kinematicG.data', bigGfile='kinematicG.gf', 
                            dtype='np.float64'):
            '''
            Save bigG and bigD to binary file

            Kwargs:
                * bigDfile  : bigD filename (optional)
                * bigGfile  : bigG gilename (optional)
                * dtype     : Data binary type 

            Returns:
                * None
            '''
            
            # Check bigG and bigD
            assert self.bigD is not None or self.bigG is not None
            assert bigDfile is not None or bigGfile is not None

            # Write files
            if bigDfile != None:
                self.bigD.astype(dtype).tofile(bigDfile)
            if bigGfile != None:
                self.bigG.astype(dtype).T.tofile(bigGfile)
            
            # All done
            return
    
    # ----------------------------------------------------------------------
    def saveBigCd(self, bigCdfile = 'kinematicG.Cd', dtype='np.float64'):
        '''
        Save bigCd matrix

        Kwargs:
            * bigCdfile     : Output filename
            * dtype         : binary type for output

        Returns:    
            * None
        '''

        # Check if Cd exists
        assert self.bigCd is not None, 'bigCd must be assigned'
        
        # Convert Cd to dtype
        Cd = self.bigCd.astype(dtype)

        # Write t file
        Cd.tofile(bigCdfile)

        # All done
        return

    def castbigM(self,n_ramp_param,eik_solver,npt=4,Dtriangles=1.,grid_space=1.0):
            '''
            Cast kinematic model into bigM for forward modeling using bigG
            (model should be specified in slip, tr and vr attributes, hypocenter 
            must be specified)

            Args:
                * n_ramp_param  : number of nuisance parameters (e.g., InSAR orbits)
                * eik_solver    : eikonal solver

            Kwargs:
                * npt**2        : numper of point sources per patch 
                * Dtriangles    : ??
                * grid_space    : ??

            Returns:
                * bigM matrix
            '''

            print('Casting model into bigM')        
            
            # Eikonal resolution
            eik_solver.setGridFromFault(self,grid_space)
            eik_solver.fastSweep()
            
            # BigG x BigM (on the fly time-domain convolution)
            Np = len(self.patch)  
            Ntriangles = int(self.bigG.shape[1]/(2*Np))
            bigM = np.zeros((self.bigG.shape[1],))
            for p in range(Np):
                # Location at the patch center
                p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)
                dip_c, strike_c = self.getHypoToCenter(p,True)
                # Grid location
                grid_size_dip = p_length/npt
                grid_size_strike = p_length/npt
                grid_strike = strike_c+np.arange(0.5*grid_size_strike,p_length,grid_size_strike) - p_length/2.
                grid_dip    = dip_c+np.arange(0.5*grid_size_dip   ,p_width ,grid_size_dip   )    - p_width/2.
                time = np.arange(Ntriangles)*Dtriangles#+Dtriangles
                T    = np.zeros(time.shape)
                Tr2  = self.tr[p]/2.            
                for i in range(npt):
                    for j in range(npt):
                        t = eik_solver.getT0([grid_dip[i]],[grid_strike[j]])[0]
                        tc = t+Tr2
                        ti = np.where(np.abs(time-tc)<Tr2)[0]            
                        T[ti] += (1/Tr2 - np.abs(time[ti]-tc)/(Tr2*Tr2))*Dtriangles
                for nt in range(Ntriangles):
                    bigM[2*nt*Np+p]     = T[nt] * self.slip[p,0]/float(npt*npt)
                    bigM[(2*nt+1)*Np+p] = T[nt] * self.slip[p,1]/float(npt*npt)

            # All done 
            return bigM  
#EOF
