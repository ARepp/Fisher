"""
   reading the millennium by Mark Neyrinck

   (c) 2006 Mark Neyrinck, modified by Istvan Szapudi, HI
   Fri Feb 18 15:46:47 HST 2011

"""

import numpy as N
import array as A
import struct as S
import os
import pylab as M

def getdatacube(ngrid = 256, prefix='/home/melody/Desktop/Simulations/millennium/',file='densmesh63.f77',galtrans=False):

    F = open(prefix+file,'rb')
    print file
    f77dummy = S.unpack('i',F.read(4))[0]
    ng3 = S.unpack('i',F.read(4))[0]
    f77dummy = S.unpack('i',F.read(4))[0]
    f77dummy = S.unpack('i',F.read(4))[0]
    #print f77dummy
    den = A.array('f')
    den.fromfile(F,ng3)
    ngrid = int(round(ng3**(1./3.)))
    print ngrid
    F.close()

    den = N.array(den).reshape((ngrid,ngrid,ngrid))#.astype(N.float32)
    if(galtrans):
        den = den.transpose(1,2,0)

    return N.double(den)

def putdatacube(den, prefix='',file='test'):

    lenden = len(den.flatten())
    F = open(prefix+file,'w')
    f77dummy = N.array([4L,lenden,4L,lenden*4]).astype('int32')
    #f77dummy = N.array([4L]).astype('int32')
    print f77dummy
    f77dummy.tofile(F)
    F.flush()
    #os.system('od -i '+prefix+file)
    (((den.flatten()).astype('float32')).squeeze()).tofile(F)
    #os.system('od -f '+prefix+file+' | head')
    f77dummy = N.array([4L*lenden]).astype('int32')
    f77dummy.tofile(F)
    print f77dummy
    F.close()

def readfield(file,typestring,arraydim=0,bytes=4,dummybytes=4,arrayshape=(),dodum=True,save=True,byteswap=False):
    if dodum:
        f77dummy = S.unpack('i',file.read(dummybytes))[0]
        print f77dummy,
    if (arraydim > 0):
        ans = A.array(typestring)
        ans.fromfile(file,arraydim)
        if (byteswap):
            ans.byteswap()
        if (len(arrayshape) == 0):
            if (typestring == 'f'):
                ans = N.array(ans,dtype=N.float32)
            else:
                ans = N.array(ans)
        else:
            if (typestring == 'f'):
                ans = N.array(ans,dtype=N.float32).reshape(arrayshape)
            else:
                N.array(ans,dtype=N.float32).reshape(arrayshape)

    else:
        ans = S.unpack(typestring,file.read(bytes))[0]
    if dodum:
        f77dummy = S.unpack('i',file.read(dummybytes))[0]
        print f77dummy

    if save:
        return ans

def readpos(file='MMF200M.GAD'):

    F = open(file,'rb')

    npart = readfield(F,'i',arraydim=1,bytes=4,dodum=False)[0]
    print npart
    pos = N.zeros((npart,3))
    pos[:,0] = readfield(F,'f',arraydim=npart,dodum=False)
    pos[:,1] = readfield(F,'f',arraydim=npart,dodum=False)
    pos[:,2] = readfield(F,'f',arraydim=npart,dodum=False)
    return pos

def readgadget(prefix='/Users/neyrinck/migad/',file='MMF200M.GAD',read_vel=False,read_id=False,debug=False):

    F = open(prefix+file,'rb')

    f77dummy = S.unpack('i',F.read(4))[0]
    if (debug):
        print f77dummy
    npart = readfield(F,'i',arraydim=6,bytes=4,dodum=False)
    massarr = readfield(F,'d',arraydim=6,bytes=8,dodum=False)
    time = S.unpack('d',F.read(8))[0]
    redshift = S.unpack('d',F.read(8))[0]
    flag_sfr = S.unpack('i',F.read(4))[0]
    flag_feedback = S.unpack('i',F.read(4))[0]
    npartTotal = readfield(F,'i',arraydim=6,bytes=4,dodum=False)

    bytesleft=f77dummy - 6*4 - 6*8 - 8 - 8 - 2*4-6*4
    la = readfield(F,'i',arraydim=bytesleft/4,dodum=False,bytes=4)
    f77dummy = S.unpack('i',F.read(4))[0]

    if (debug):
        print npart
        print massarr
        print time,redshift,flag_sfr,flag_feedback
        print npartTotal

    pos = readfield(F,'f',arraydim=3*npart[1],dodum=True,arrayshape=(npart[1],3))
    if (read_vel):
        vel = readfield(F,'f',arraydim=3*npart[1],dodum=True)
    else:
        readfield(F,'f',arraydim=3*npart[1],dodum=True,save=False)

    id = readfield(F,'i',arraydim=npart[1],dodum=True)
    pos_reord = N.empty(shape=pos.shape,dtype=N.float32)
    pos_reord[id,:] = 1.*pos

    return pos_reord

def readgadgetmad(prefix='/Users/neyrinck/halomad/',file='FullBox_z0.0_0256',read_vel=False,debug=True,nfiles=1,byteswap=True):
    if nfiles==1:
        filenames=[file]
    else:
        filenames=file+'.'+N.arange(nfiles).astype(string)
    pos_reord = None
    for i in range(nfiles):
        if (nfiles == 1):
            filename = file
        else:
            filename = file+'.'+str(i)

        F = open(prefix+filename,'rb')
        f77dummy = S.unpack('>i',F.read(4))[0]
        if (debug):
            print f77dummy
        npart = readfield(F,'i',arraydim=6,bytes=4,dodum=False,byteswap=byteswap)
        massarr = readfield(F,'d',arraydim=6,bytes=8,dodum=False,byteswap=byteswap)
        time = S.unpack('>d',F.read(8))[0]
        redshift = S.unpack('>d',F.read(8))[0]
        flag_sfr = S.unpack('>i',F.read(4))[0]
        flag_feedback = S.unpack('>i',F.read(4))[0]
        npartTotal = readfield(F,'i',arraydim=6,bytes=4,dodum=False,byteswap=byteswap)

        if (debug):
            print npart
            print massarr
            print time,redshift,flag_sfr,flag_feedback
            print npartTotal
            print N.sum(npart)

        if pos_reord == None:
            pos_reord = N.empty((N.sum(npartTotal),3),dtype=N.float32)

        bytesleft=f77dummy - 6*4 - 6*8 - 8 - 8 - 2*4-6*4
        la = readfield(F,'i',arraydim=bytesleft/4,dodum=False,bytes=4,byteswap=byteswap)
        f77dummy = S.unpack('>i',F.read(4))[0]

        if (read_vel):
            readfield(F,'f',arraydim=3*N.sum(npart),dodum=True,save=False)
            pos = readfield(F,'f',arraydim=3*N.sum(npart),dodum=True,arrayshape=(N.sum(npart),3),byteswap=byteswap)
        else:
            pos = readfield(F,'f',arraydim=3*N.sum(npart),dodum=True,arrayshape=(N.sum(npart),3),byteswap=byteswap)
            readfield(F,'f',arraydim=3*N.sum(npart),dodum=True,byteswap=byteswap,save=False)

        id = readfield(F,'i',arraydim=N.sum(npart),dodum=True,byteswap=byteswap)
        maxid = N.max(id).astype(N.long)
        print min(id),maxid
        print pos_reord[id-1,:].shape, pos.shape
        pos_reord[id-1,:] = 1.*pos

    return pos_reord

def writegrid(den, prefix='/Users/neyrinck/haijun/',file='test'):

    F = open(prefix+file,'w')
    (((den.flatten()).astype('float32')).squeeze()).tofile(F)
    F.close()

def readgrid(shape=(286,286), prefix='/Users/neyrinck/haijun/dr7results/',file='test'):

    F = open(prefix+file,'r')
    den = A.array('f')
    den.fromfile(F,N.prod(shape))
    F.close()
    return(N.array(den).reshape(shape).astype(N.float32))

def vobozout(pos,filename,f77=False):
    F = open(filename,'w')

    
    np = len(pos[:,0])
    if (f77):
        (4*N.array([1])).tofile(F)
    N.array([np]).tofile(F)
    if (f77):
        (4*N.array([1,np])).tofile(F)
    (pos[:,0]).tofile(F)
    if (f77):
        (4*N.array([np,np])).tofile(F)
    (pos[:,1]).tofile(F)
    if (f77):
        (4*N.array([np,np])).tofile(F)
    (pos[:,2]).tofile(F)
    if (f77):
        (4*N.array([np])).tofile(F)

    F.close()

    #os.system('od -i '+filename+' | head -1')
    #os.system('od -f '+filename+' | head')

def vobozin(filename):
    F = open(filename,'r')

    np = S.unpack('i',F.read(4))[0]
    print np
    pos = N.empty((np,3),dtype=N.float32)
    pos[:,0] = readfield(F,'f',arraydim=N.sum(np),dodum=False)
    pos[:,1] = readfield(F,'f',arraydim=N.sum(np),dodum=False)
    pos[:,2] = readfield(F,'f',arraydim=N.sum(np),dodum=False)

    return pos

def vozisol_ascii(p,npreal=6619136,prefix='/Users/neyrinck/halomad/lg2048/',root='lg2048_dm2',randseed=12345):
    np = len(p[:,0])

    #min0 = min(p[:,0])
    #min1 = min(p[:,1])
    #min2 = min(p[:,2])
    #max0 = max(p[:,0])
    #max1 = max(p[:,1])
    #max2 = max(p[:,2])
    #maxrange=max([max0-min0,max1-min1,max2-min2])

    #print min0,min1,min2,maxrange

    rs = N.random.RandomState(randseed)

    p1=p#/1e4
    #if (randseed != None):
    #    p1 += rs.rand(np,3)/2.**20
    #p1[:,0] = (p[:,0]-min0)/maxrange
    #p1[:,1] = (p[:,1]-min1)/maxrange
    #p1[:,2] = (p[:,2]-min2)/maxrange

    F=open(prefix+root+'.pos','w')
    F.write('%d %d\n'%(np, npreal))
    for i in range(np):
        F.write('%10.8f %10.8f %10.8f\n'%(p1[i,0],p1[i,1],p1[i,2]))

    F.close()
