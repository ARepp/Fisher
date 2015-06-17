import numpy as N
#import pylab as M
#import utils
import os.path
#import hankel

sqrt2 = N.sqrt(2.)

def pk(d, boxsize=500.,bin2fact=1./16., filename='',getnuminbin=False,overwrite=True,
       checkmean=False,getxi=False):

    if (os.path.isfile(filename))*(overwrite == False):
        #print 'not calculating'
        p = N.loadtxt(filename)
        kmean = p[:,0]
        pk = p[:,1]
        numinbin = p[:,2]

    else:
        #print 'calculating'

        if (checkmean):
            meanden = N.mean(d.flatten())
            if (meanden != 0.):
                print 'Mean = ',meanden,'.  Subtracting it off.'
                d -= meanden
        dk = N.fft.fftn(d)
        s = d.shape
        sk = dk.shape
        dk2 = (dk*N.conjugate(dk)).astype(N.float32)
        #M.pcolor(N.abs(dk[:,:,0]))        

        # need to double-count along the z-axis
        
        kmin = 2.*N.pi/boxsize

        if (len(s) == 3):
            a = N.fromfunction(lambda x,y,z:x, sk).astype(N.float32)
            a[N.where(a > s[0]/2)] -= s[0]
            b = N.fromfunction(lambda x,y,z:y, sk).astype(N.float32)
            b[N.where(b > s[1]/2)] -= s[1]
            c = N.fromfunction(lambda x,y,z:z, sk).astype(N.float32)
            c[N.where(c > s[2]/2)] -= s[2]
            # half-count cells on the z-axis
            k = kmin*N.sqrt((a**2+b**2+c**2).flatten()).astype(N.float32)
            dk2 = dk2*(N.sinc(N.fft.fftfreq(sk[0]))*N.sinc(N.fft.fftfreq(sk[1]))*N.sinc(N.fft.fftfreq(sk[2])))**(-2.)
        elif (len(s) == 2):
            b = N.fromfunction(lambda y,z:y, sk).astype(N.float32)
            b[N.where(b > s[0]/2)] -= s[0]
            c = N.fromfunction(lambda y,z:z, sk).astype(N.float32)
            c[N.where(c > s[1]/2)] -= s[1]

            k = kmin*N.sqrt((b**2+c**2).flatten()).astype(N.float32)
        

        dk2 = dk2.flatten()
        index = N.argsort(k)

        k = k[index]
        dk2 = dk2[index]

        c0 = 0.*c.flatten() + 1.
        c0[N.where(c.flatten() == 0.)] -= 0.5
        c0 = c0[index]

        log2 = N.log(2.)
    
        binedges = kmin*2.**N.arange(-bin2fact/2.,N.log(k[-1]/kmin)/log2,bin2fact)
        cuts = N.searchsorted(k,binedges)
        numinbin = 0.*binedges
        pk = 0.*binedges
        kmean = 0.*binedges
        nbins = len(binedges)

        for i in N.arange(0,nbins-1):
            if (cuts[i+1] > cuts[i]):
                numinbin[i] = N.sum(c0[cuts[i]:cuts[i+1]])
                pk[i] = N.sum(c0[cuts[i]:cuts[i+1]]*dk2[cuts[i]:cuts[i+1]])
                kmean[i] = N.sum(c0[cuts[i]:cuts[i+1]]*k[cuts[i]:cuts[i+1]])

        wn0 = N.where(numinbin > 0.)[0]
        pk = pk[wn0]; kmean = kmean[wn0]; numinbin = numinbin[wn0]
        pk /= numinbin
        kmean /= numinbin

        pk *= boxsize**3/N.prod(N.array(s).astype(float))**2

        if filename != '':
            N.savetxt(filename, N.transpose([kmean,pk,numinbin]))

    if (getnuminbin):
        return kmean,pk,numinbin
    else:
        return kmean,pk

def cross(d1, d2, boxsize=500.,bin2fact=1./16., filename='',getnuminbin=False,overwrite=False):
    ngrid = len(d1[:,0,0])

    if (os.path.isfile(filename))*(overwrite == False):
        p = N.loadtxt(filename)
        kmean = p[:,0]
        pk = p[:,1]
        numinbin = p[:,2]

    if (ngrid != len(d2[:,0,0])):
        print 'arrays not the same size; quit'
        return

    d1k = N.fft.rfftn(d1)
    d2k = N.fft.rfftn(d2)
    s = d1.shape
    sk = d1k.shape
    dk2 = N.real(d1k*N.conjugate(d2k) + N.conjugate(d1k)*d2k)/2.

    #print N.abs(dk2[0,0,:])/N.sqrt(N.abs(d1k[0,0,:])**2*N.abs(d2k[0,0,:])**2)
    dk2 = dk2.flatten()

    #pokj

    # need to double-count along the z-axis

    a = N.fromfunction(lambda x,y,z:x, sk)
    a[N.where(a > s[0]/2)] -= s[0]
    b = N.fromfunction(lambda x,y,z:y, sk)
    b[N.where(b > s[1]/2)] -= s[1]
    c = N.fromfunction(lambda x,y,z:z, sk)
    c[N.where(c > s[2]/2)] -= s[2]


    kmin = 2.*N.pi/boxsize

    k = kmin*N.sqrt((a**2+b**2+c**2).flatten())
    index = N.argsort(k)

    k = k[index]
    dk2 = dk2[index]

    c0 = 0.*c.flatten() + 1.
    c0[N.where(c.flatten() == 0.)] -= 0.5
    c0 = c0[index]
    # half-count cells on the z-axis

    log2 = N.log(2.)
    
    binedges = kmin*2.**N.arange(-bin2fact/2.,N.log(k[-1]/kmin)/log2,bin2fact)
    cuts = N.searchsorted(k,binedges)
    numinbin = 0.*binedges
    pk = 0.*binedges
    kmean = 0.*binedges
    nbins = len(binedges)

    for i in N.arange(0,nbins-1):
        if (cuts[i+1] > cuts[i]):
            numinbin[i] = N.sum(c0[cuts[i]:cuts[i+1]])
            pk[i] = N.sum(c0[cuts[i]:cuts[i+1]]*dk2[cuts[i]:cuts[i+1]])
            kmean[i] = N.sum(c0[cuts[i]:cuts[i+1]]*k[cuts[i]:cuts[i+1]])

    wn0 = N.where(numinbin > 0.)[0]
    pk = pk[wn0]; kmean = kmean[wn0]; numinbin = numinbin[wn0]
    pk /= numinbin
    kmean /= numinbin

    pk *= boxsize**3/N.prod(N.array(s).astype(float))**2

    if filename != '':
        N.savetxt(filename, N.transpose([kmean,pk,numinbin]))

    return kmean,pk

def r(d1, d2, boxsize=500.,bin2fact=1./16., filename='',getnuminbin=False,overwrite=False,getnumbin=False,special='',plotty=False):
    ngrid = len(d1[:,0,0])

    if (os.path.isfile(filename))*(overwrite == False):
        p = N.loadtxt(filename)
        kmean = p[:,0]
        pk = p[:,1]
        numinbin = p[:,2]

    if (ngrid != len(d2[:,0,0])):
        print 'arrays not the same size; quit'
        return

    d1k = N.fft.rfftn(d1)
    d2k = N.fft.rfftn(d2)
    s = d1.shape
    sk = d1k.shape
    #d1k = d1k.flatten()
    #d2k = d2k.flatten()
    if special == '':
        dk2 = 0.5*(d1k*N.conjugate(d2k) + N.conjugate(d1k)*d2k)/\
            N.sqrt(d1k*N.conjugate(d1k)*d2k*N.conjugate(d2k))
    elif special == 'propagator':
        #d1 must be initial conditions!
        dk2 = 0.5*N.real((d1k*N.conjugate(d2k) + N.conjugate(d1k)*d2k)/\
            (d1k*N.conjugate(d1k)))
    elif special == 'amplitude':
        dk2 = 0.5*(d1k*N.conjugate(d2k) + N.conjugate(d1k)*d2k)/\
            (d1k*N.conjugate(d1k))

    if (plotty):
        M.clf()
        M.pcolor(N.real(dk2[:,0,:]))
        M.colorbar()

    dk2 = dk2.flatten()

    a = N.fromfunction(lambda x,y,z:x, sk)
    a[N.where(a > s[0]/2)] -= s[0]
    b = N.fromfunction(lambda x,y,z:y, sk)
    b[N.where(b > s[1]/2)] -= s[1]
    c = N.fromfunction(lambda x,y,z:z, sk)
    c[N.where(c > s[2]/2)] -= s[2]


    kmin = 2.*N.pi/boxsize

    k = kmin*N.sqrt((a**2+b**2+c**2).flatten())
    index = N.argsort(k)

    k = k[index]
    dk2 = dk2[index]

    c0 = 0.*c.flatten() + 1.
    c0[N.where(c.flatten() == 0.)] -= 0.5
    c0 = c0[index]
    # half-count cells on the z-axis

    log2 = N.log(2.)
    
    binedges = kmin*2.**N.arange(-bin2fact/2.,N.log(k[-1]/kmin)/log2,bin2fact)
    cuts = N.searchsorted(k,binedges)
    numinbin = 0.*binedges
    pk = 0.*binedges
    kmean = 0.*binedges
    nbins = len(binedges)

    for i in N.arange(0,nbins-1):
        if (cuts[i+1] > cuts[i]):
            numinbin[i] = N.sum(c0[cuts[i]:cuts[i+1]])
            pk[i] = N.sum(c0[cuts[i]:cuts[i+1]]*dk2[cuts[i]:cuts[i+1]])
            kmean[i] = N.sum(c0[cuts[i]:cuts[i+1]]*k[cuts[i]:cuts[i+1]])

    wn0 = N.where(numinbin > 0.)[0]
    pk = pk[wn0]; kmean = kmean[wn0]; numinbin = numinbin[wn0]
    pk /= numinbin
    kmean /= numinbin

    #pk *= boxsize**3/N.prod(N.array(s).astype(float))**2

    if filename != '':
        N.savetxt(filename, N.transpose([kmean,pk,numinbin]))

    if (getnumbin):
        return kmean,pk, getnumbin
    else:
        return kmean,pk

def weightings(d, maxorder1d=2, maxorder2=None, outroot='', setmean0=True,include000=False,boxsize=500.):
    phase1, phase2 = 1./16., 5./16.
    s = d.shape
    if (setmean0):
        d -= N.mean(d.flatten())

    if (s[0] <= 256):
        x=N.fromfunction(lambda x,y,z:x, s, dtype=N.uint8)
        y=N.fromfunction(lambda x,y,z:y, s, dtype=N.uint8)
        z=N.fromfunction(lambda x,y,z:z, s, dtype=N.uint8)
    else:
        x=N.fromfunction(lambda x,y,z:x, s, dtype=N.uint16)
        y=N.fromfunction(lambda x,y,z:y, s, dtype=N.uint16)
        z=N.fromfunction(lambda x,y,z:z, s, dtype=N.uint16)
    
    k,pmean,numinbin = pk(d, getnuminbin = True)
    powerarray = N.zeros((((2*maxorder1d+1)**3-1+int(include000))*2,len(pmean)),N.float32)
    print (2*maxorder1d+1)**3-1+int(include000)
    numspectra = 0
    for xi in N.arange(-maxorder1d,maxorder1d+1):
        for yi in N.arange(-maxorder1d,maxorder1d+1):
            for zi in N.arange(-maxorder1d,maxorder1d+1):
                if include000 or ((xi!=0) or (yi!=0) or (zi!=0)):
                    print xi,yi,zi,numspectra
                    if maxorder2 != None:
                        if xi**2+yi**2+zi**2 > maxorder2:
                            continue
                    for phase in [1./16., 5./16.]:
                        k,powerarray[numspectra,:] = \
                            pk(d*sqrt2*N.cos(2.*N.pi*(xi*x+yi*y+zi*z)/N.float(s[0]) + phase),boxsize=boxsize)
                        numspectra += 1

                    
    N.savetxt(outroot+'.p',N.transpose([k,pmean,numinbin]),fmt='%e')
    N.savetxt(outroot+'.hrs',powerarray,fmt='%e')


def plogbias(d, boxsize=500., measfact = 4.,filename=''):
    """ d = density, not delta """
    ngrid = len(d[:,0,0])
    dk = N.fft.rfftn(d-1.)
    s = d.shape
    sk = dk.shape
    dk2 = (dk*N.conjugate(dk)).real.flatten()
    dlog = N.log(d)
    dlog -= N.mean(d)
    dlogk = N.fft.rfftn(dlog)
    dlogk2 = (dlogk*N.conjugate(dlogk)).real.flatten()

    a = N.fromfunction(lambda x,y,z:x, sk)
    a[N.where(a > s[0]/2)] -= s[0]
    b = N.fromfunction(lambda x,y,z:y, sk)
    b[N.where(b > s[1]/2)] -= s[1]
    c = N.fromfunction(lambda x,y,z:z, sk)
    c[N.where(c > s[2]/2)] -= s[2]

    c0 = 0.*c.flatten() + 1.
    c0[N.where(c.flatten() == 0.)] -= 0.5
    # half-count cells on the z-axis

    kmin = 2.*N.pi/boxsize

    k = kmin*N.sqrt((a**2+b**2+c**2).flatten())
    w = N.where((k <= kmin*measfact)*(k > 0.))[0]
    print len(w), N.sum(c0[w])

    M.loglog(k[w],dk2[w],'b.')
    M.loglog(k[w],dlogk2[w],'g.')
    M.show()

    #print dlogk2[w]
    #print dk2[w]

    nkeff = N.sum(c0[w])
    p = N.sum(dk2[w]*c0[w])/nkeff
    plog = N.sum(dlogk2[w]*c0[w])/nkeff
    bias = plog/p
    vard = N.sum(dk2[w]**2*c0[w])/N.sum(c0[w]) - p**2
    vardlog = N.sum(dlogk2[w]**2*c0[w])/N.sum(c0[w]) - plog**2

    stdbias = N.sqrt(vard/p**2 + vardlog/plog**2)*bias

    return bias,stdbias

def window2(kx,knyq,p=1):
    knyq2 = 2*knyq
    ans = 0.*kx + 1.
    wnon0 = N.where(kx != 0.)[0]
    ans[wnon0] = (N.sin(N.pi*kx[wnon0]/knyq2)/(N.pi*kx[wnon0]/knyq2))**(2*p)

    if (N.sum(N.isnan(ans).astype(int)) > 0):
        print kx
        print N.sin(N.pi*kx/knyq2)
        print N.pi*kx/knyq2
        print ans
        unui
    return ans

def dodecahedronVertices():
    """ Returns one octant of dodecahedron vertices """
    phi = (1.+N.sqrt(5.))/2. # golden ratio
    sqrt3 = N.sqrt(3.)
    x = N.array([1.,    0.])/sqrt3
    y = N.array([1.,1./phi])/sqrt3
    z = N.array([1.,   phi])/sqrt3
    print x**2+y**2+z**2
    weights = N.array([3.,2.])/5.

    return x,y,z,weights

def pk_smallscalepowlaw(pk,k,knew, wnyqo2,logpower):
    ans = 0.*knew
    w = N.where(knew <= k[wnyqo2])[0]
    wnot = N.where(knew > k[wnyqo2])[0]
    if (len(w) > 0):
        ans[w] = utils.interpolateLinLog(pk,k,knew[w])
     
    if (len(wnot) > 0):
        ans[wnot] = pk[wnyqo2]*(knew[wnot]/k[wnyqo2])**logpower

    wsmall = N.where(knew < k[0])[0]
    ans[wsmall] *= 0.
    
    return ans

def jingle(k,pk,ngrid=256, plotfit=False, nrad = 3, logpower = 0.,nside=2):
    knyq = k[0]*ngrid/2
    wnyq = N.where(k < knyq)[0][-1]
    wnyqo2 = N.where(k < knyq/2.)[0][-1]
    if (logpower == 0.):
        logpower = N.log(pk[wnyq]/pk[wnyqo2])/N.log(k[wnyq]/k[wnyqo2])

    if (plotfit):
        #M.loglog(k,pk)
        M.loglog(k,pk[wnyqo2]*(k/k[wnyqo2])**logpower)

    nshape = (2*nrad+1,2*nrad+1,2*nrad+1)
    nx = N.fromfunction(lambda x,y,z:x-nrad, nshape).flatten()*2*knyq
    ny = N.fromfunction(lambda x,y,z:y-nrad, nshape).flatten()*2*knyq
    nz = N.fromfunction(lambda x,y,z:z-nrad, nshape).flatten()*2*knyq
    #x,y,z,weights = dodecahedronVertices()
    heallist = N.loadtxt('heallist.nside'+str(nside)+'.txt')
    x,y,z = heallist[:,0],heallist[:,1],heallist[:,2]
    

    c2 = 0.*pk
    for i in range(len(x)):
        for ki in range(len(c2)):
            totk = N.sqrt((x[i]*k[ki]+nx)**2 + (y[i]*k[ki]+ny)**2 + (z[i]*k[ki]+nz)**2)
            if (ki == -1):
                M.subplot(121)
                M.plot(x[i]*k[ki]+nx,window2(x[i]*k[ki]+nx,knyq),'.')
                print pk_smallscalepowlaw(pk,k,totk, wnyqo2,logpower)/pk[ki]

                M.subplot(122)
                M.plot(nx**2+ny**2+nz**2,pk_smallscalepowlaw(pk,k,totk, wnyqo2,logpower)/pk[ki],'.')
                M.show()
                


            c2[ki] += N.sum(window2(x[i]*k[ki]+nx,knyq)*
                            window2(y[i]*k[ki]+ny,knyq)*
                            window2(z[i]*k[ki]+nz,knyq)*
                            pk_smallscalepowlaw(pk,k,totk, wnyqo2,logpower))\
                            /pk[ki]
    

    newpk = pk/c2*len(x)
    newlogpower = N.log(newpk[wnyq]/newpk[wnyqo2])/N.log(k[wnyq]/k[wnyqo2])

    #M.loglog(k,pk)
    #M.loglog(k,pk/c2)
    #M.show()

    return newpk, newlogpower

def jingles(k,pk,ngrid=256, nrad = 3, numjingles=3):
    pkmodify = 1.*pk
    M.loglog(k,pk)

    pkmodify, newlogpower = jingle(k,pk,ngrid=ngrid, nrad = nrad)
    M.loglog(k,pkmodify)
    M.loglog(N.array([1.,1.])*k[0]*ngrid/2., [1e2,1e3],':')
    for j in range(numjingles-1):
        pkmodify, newlogpower = jingle(k,pk,ngrid=ngrid, plotfit=False, nrad = nrad,logpower=newlogpower)
        M.loglog(k,pkmodify)

def corfunk(d, boxsize=500.,binsize=1, filename='',getnuminbin=False,overwrite=True,checkmean=False,exclude0lag=True,get2d=False,pisigma3d =False):
    if (os.path.isfile(filename))*(overwrite == False):
        p = N.loadtxt(filename)
        kmean = p[:,0]
        pk = p[:,1]
        numinbin = p[:,2]

    else:
        
        if (checkmean):
            meanden = N.mean(d.flatten())
            if (meanden != 0.):
                print 'Mean = ',meanden,'.  Subtracting it off.'
                d -= meanden
        s = d.shape

        dk = N.fft.rfftn(d)/N.prod(s)
        sk = dk.shape
        dk = dk*N.conjugate(dk)
        if (len(s) == 3):
            dk[0,0,0] = 0.
        elif (len(s) == 2):
            dk[0,0] = 0.

        #note that the following are Fourier-space names, even though
        #they should be in configuration space

        dk2 = N.fft.irfftn(dk)*N.prod(s)

        if (get2d):
            xips = 1.*dk2

        sxi = dk2.shape
        kmin = boxsize/float(s[0])


        if (len(s) == 3):
            a = N.fromfunction(lambda x,y,z:x, sxi)
            a[N.where(a > s[0]/2)] -= s[0]
            b = N.fromfunction(lambda x,y,z:y, sxi)
            b[N.where(b > s[1]/2)] -= s[1]
            c = N.fromfunction(lambda x,y,z:z, sxi)
            c[N.where(c > s[2]/2)] -= s[2]
            # half-count cells on the z-axis

            if (pisigma3d):
                rp = kmin*N.sqrt(b**2+c**2)
            else:
                k = kmin*N.sqrt((a**2+b**2+c**2).flatten())

        elif (len(s) == 2):
            b = N.fromfunction(lambda y,z:y, sxi)
            b[N.where(b > s[0]/2)] -= s[0]
            c = N.fromfunction(lambda y,z:z, sxi)
            c[N.where(c > s[1]/2)] -= s[1]

            k = kmin*N.sqrt((b**2+c**2).flatten())
        elif (len(s) == 1):
            c = N.fromfunction(lambda z:z, sxi)
            c[N.where(c > s[0]/2)] -= s[0]

            k = kmin*c

        if (pisigma3d):
            c0 = 0.*c + 1.
            c0[N.where(c == 0.)] -= 0.5


            xips = N.zeros((s[0]/2,s[1]/2))
            # loop over planes perpendicular to pi direction
            for i in range(s[0]/2):
                # sort by rp
                index = N.argsort(rp[i,:,:].flatten())
                rp_sort = rp[i,:,:].flatten()[index]
                dk2_sort = dk2[i,:,:].flatten()[index]
                binedges = kmin*N.arange(0,s[0]/2+1,binsize)
                c0_indexed = c0[i,:,:].flatten()[index]

                cuts = N.searchsorted(rp_sort,binedges)
                numinbin = 0.*binedges
                nbins = len(binedges)
                for ic in N.arange(0,nbins-1):
                    if (cuts[ic+1] > cuts[ic]):
                        xips[i,ic] = N.sum(c0_indexed[cuts[ic]:cuts[ic+1]]*dk2_sort[cuts[ic]:cuts[ic+1]])/\
                            N.sum(c0_indexed[cuts[ic]:cuts[ic+1]])
                        #kmean[ic] = N.sum(c0_indexed[cuts[ic]:cuts[ic+1]]*k[cuts[ic]:cuts[ic+1]])
                        

        else:
            c0 = 0.*c.flatten() + 1.
            c0[N.where(c.flatten() == 0.)] -= 0.5
        
            index = N.argsort(k)
            k = k[index]
            dk2 = dk2.flatten()[index]
            c0 = c0[index]
        
            log2 = N.log(2.)
    
            binedges = kmin*N.arange(0,s[0]/2,binsize)
            cuts = N.searchsorted(k,binedges)
            numinbin = 0.*binedges
            pk = 0.*binedges
            kmean = 0.*binedges
            nbins = len(binedges)

            for i in N.arange(0,nbins-1):
                if (cuts[i+1] > cuts[i]):
                    numinbin[i] = N.sum(c0[cuts[i]:cuts[i+1]])
                    pk[i] = N.sum(c0[cuts[i]:cuts[i+1]]*dk2[cuts[i]:cuts[i+1]])
                    kmean[i] = N.sum(c0[cuts[i]:cuts[i+1]]*k[cuts[i]:cuts[i+1]])

            wn0 = N.where(numinbin > 0.)[0]
            pk = pk[wn0]; kmean = kmean[wn0]; numinbin = numinbin[wn0]
            pk /= numinbin
            kmean /= numinbin

        #pk *= boxsize**3/N.prod(N.array(s).astype(float))**2
            #pk /= N.prod(N.array(s).astype(float))

            if filename != '':
                N.savetxt(filename, N.transpose([kmean,pk,numinbin]))

            if (exclude0lag):
                kmean = kmean[1:]
                pk = pk[1:]
                numinbin=numinbin[1:]

    if (getnuminbin):
        return kmean,pk,numinbin
    elif (get2d):
        return xips
    else:
        return kmean,pk

def pk_nosn(d, boxsize=500.,bin2fact=1./16., filename='',getnuminbin=False,overwrite=True,checkmean=False,oldk=None):
    r,xi = corfunk(d)
    logr = N.log(r)

    if (oldk == None):
        k = 2.*N.pi/r
    else:
        k=1.*oldk

    x = N.exp(N.arange(N.log(r[0]/8.),N.log(8.*r[-1]),N.log(2.)/8.))
    plotty = utils.splineIntBig0LittleLog(xi,r,x)
    M.loglog(x,N.abs(plotty),'g:')
    M.loglog(x,plotty,'g')
    M.loglog(x,plotty,'g.')
    M.loglog(r,N.abs(xi),'.')

    hank = hankel.Hankel3D(10000)
    pk_nosn = hank.transform(lambda x:utils.splineIntBig0LittleLog(xi,r,x),k,n=10000,h=1./512.,pk2xi=False)

    return k,pk_nosn
