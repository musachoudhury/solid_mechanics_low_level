from cffi import FFI
import numpy as np

ffi = FFI()


def call_umat(stress, statev, ddsdde, dstran):
    C = ffi.dlopen(
        "/home/musa/Documents/main_projects/my_demos/solid_mechanics_low_level/ufl_BTDB/umat.so"
    )

    ffi.cdef(
        """
            void umat_(double*,double*,double*,double*,double*,double*,
        double*,double*,double*,double*,
        double*,double*,double*,double*,double*,double*,double*,double*,char[],
        int*,int*,int*,int*,double*,int*,double*,double*,double*,
        double*,double*,double*,int*,int*,int*,int*,int*,int*);
    """,
        override=True,
    )
    # stress = np.array(stress)
    # statev = np.array(statev)
    # ddsdde = np.array(ddsdde)
    # dstran = np.array(dstran)

    # stress = np.ones((6, 1), dtype=np.float64)
    # statev = np.zeros((1, 1), dtype=np.float64)
    # ddsdde = np.zeros((6, 6), dtype=np.float64)
    sse = np.array(0, dtype=np.float64)
    spd = np.array(0, dtype=np.float64)
    scd = np.array(0, dtype=np.float64)
    rpl = np.array(0, dtype=np.float64)
    ddsddt = np.zeros((6, 1), dtype=np.float64)
    drplde = np.zeros((6, 1), dtype=np.float64)
    drpldt = np.array(0, dtype=np.float64)
    stran = np.zeros((6, 1), dtype=np.float64)
    # dstran = np.zeros((6, 1), dtype=np.float64)
    time = np.array(0, dtype=np.float64)

    dtime = np.array(0, dtype=np.float64)
    temp = np.array(0, dtype=np.float64)
    dtemp = np.array(0, dtype=np.float64)
    predef = np.array(0, dtype=np.float64)
    dpred = np.array(0, dtype=np.float64)
    cmname = "name"
    ndi = np.array(3, dtype=int)
    nshr = np.array(3, dtype=int)
    ntens = np.array(6, dtype=int)
    nstatv = np.array(0, dtype=int)
    props = np.array((2, 1), dtype=np.float64)
    nprops = np.array(2, dtype=int)
    coords = np.zeros((3, 1), dtype=np.float64)
    drot = np.zeros((3, 3), dtype=np.float64)
    pnewdt = np.array(0, dtype=np.float64)
    celent = np.array(0, dtype=np.float64)
    dfgrd0 = np.zeros((3, 3), dtype=np.float64)
    dfgrd1 = np.zeros((3, 3), dtype=np.float64)
    noel = np.array(0, dtype=int)
    npt = np.array(0, dtype=int)
    layer = np.array(0, dtype=int)
    kspt = np.array(0, dtype=int)
    kstep = np.array(0, dtype=int)
    kinc = np.array(0, dtype=int)

    stress_ = ffi.cast("double*", ffi.from_buffer(stress))
    statev_ = ffi.cast("double*", ffi.from_buffer(statev))
    ddsdde_ = ffi.cast("double*", ffi.from_buffer(ddsdde))
    sse_ = ffi.cast("double*", ffi.from_buffer(sse))
    spd_ = ffi.cast("double*", ffi.from_buffer(spd))
    scd_ = ffi.cast("double*", ffi.from_buffer(scd))
    rpl_ = ffi.cast("double*", ffi.from_buffer(rpl))
    ddsddt_ = ffi.cast("double*", ffi.from_buffer(ddsddt))
    drplde_ = ffi.cast("double*", ffi.from_buffer(drplde))
    drpldt_ = ffi.cast("double*", ffi.from_buffer(drpldt))
    stran_ = ffi.cast("double*", ffi.from_buffer(stran))
    dstran_ = ffi.cast("double*", ffi.from_buffer(dstran))
    time_ = ffi.cast("double*", ffi.from_buffer(time))

    dtime_ = ffi.cast("double*", ffi.from_buffer(dtime))
    temp_ = ffi.cast("double*", ffi.from_buffer(temp))
    dtemp_ = ffi.cast("double*", ffi.from_buffer(dtemp))
    predef_ = ffi.cast("double*", ffi.from_buffer(predef))
    dpred_ = ffi.cast("double*", ffi.from_buffer(dpred))

    ndi_ = ffi.cast("int*", ffi.from_buffer(ndi))
    nshr_ = ffi.cast("int*", ffi.from_buffer(nshr))
    ntens_ = ffi.cast("int*", ffi.from_buffer(ntens))
    nstatv_ = ffi.cast("int*", ffi.from_buffer(nstatv))
    props_ = ffi.cast("double*", ffi.from_buffer(props))
    nprops_ = ffi.cast("int*", ffi.from_buffer(nprops))
    cmname_ = cmname.encode("ascii")  # ffi.cast("char*", ffi.from_buffer(cmname))
    coords_ = ffi.cast("double*", ffi.from_buffer(coords))
    drot_ = ffi.cast("double*", ffi.from_buffer(drot))
    pnewdt_ = ffi.cast("double*", ffi.from_buffer(pnewdt))
    celent_ = ffi.cast("double*", ffi.from_buffer(celent))
    dfgrd0_ = ffi.cast("double*", ffi.from_buffer(dfgrd0))
    dfgrd1_ = ffi.cast("double*", ffi.from_buffer(dfgrd1))

    noel_ = ffi.cast("int*", ffi.from_buffer(noel))
    npt_ = ffi.cast("int*", ffi.from_buffer(npt))
    layer_ = ffi.cast("int*", ffi.from_buffer(layer))
    kspt_ = ffi.cast("int*", ffi.from_buffer(kspt))
    kstep_ = ffi.cast("int*", ffi.from_buffer(kstep))
    kinc_ = ffi.cast("int*", ffi.from_buffer(kinc))

    C.umat_(
        stress_,
        statev_,
        ddsdde_,
        sse_,
        spd_,
        scd_,
        rpl_,
        ddsddt_,
        drplde_,
        drpldt_,
        stran_,
        dstran_,
        time_,
        dtime_,
        temp_,
        dtemp_,
        predef_,
        dpred_,
        cmname_,
        ndi_,
        nshr_,
        ntens_,
        nstatv_,
        props_,
        nprops_,
        coords_,
        drot_,
        pnewdt_,
        celent_,
        dfgrd0_,
        dfgrd1_,
        noel_,
        npt_,
        layer_,
        kspt_,
        kstep_,
        kinc_,
    )