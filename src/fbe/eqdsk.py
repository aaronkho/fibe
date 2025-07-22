from pathlib import Path
import numpy as np

array_types = (list, tuple, np.ndarray)


def read_eqdsk_file(fname):
    """ Read an eqdsk file """

    def _sep_eq_line(line, float_width=16, floats_per_line=5, sep=' '):
        """ Split a eqdsk-style line and inserts seperator characters """
        splitted = [
            line[num*float_width:(num+1)*float_width] for num in range(floats_per_line)
        ]
        separate = sep.join(splitted)
        return separate

    def _read_chunk(lines, length, floats_per_line=5):
        """ Read a single chunk (array/vector)

        Reads and pops for `lines` the amount of lines
        containing the to be read vector.

        Args:
            lines:  List of lines to be read. Destructive!
            length: Length of to be read vector

        Kwargs:
            floats_per_line: Amount of floats on a line [Default: 5]
        """
        num_lines = int(np.ceil(length / floats_per_line))
        vals = []
        for line in lines[:num_lines]:
            sep = _sep_eq_line(line)
            vals.append(np.fromstring(sep, sep=' '))
        del lines[:num_lines]
        return vals

    with open(fname, 'r') as ff:
        lines = ff.readlines()

    gcase = lines[0][:48].strip()
    header = lines.pop(0)[48:].split()
    eq = {}
    # Read sizes of arrays/vectors
    eq['gid'] = int(header[0])
    eq['gcase'] = str(gcase)
    eq['nr'] = int(header[1])
    eq['nz'] = int(header[2])

    # Read singles
    eq['rdim'], eq['zdim'], eq['rcentr'], eq['rleft'], eq['zmid'] = np.fromstring(_sep_eq_line(lines.pop(0)), sep=' ')
    eq['rmagx'], eq['zmagx'], eq['simagx'], eq['sibdry'], eq['bcentr'] = np.fromstring(_sep_eq_line(lines.pop(0)), sep=' ')
    eq['cpasma'], eq['simagx2'], eq['dum0'], eq['rmagx2'], eq['dum1'] = np.fromstring(_sep_eq_line(lines.pop(0)), sep=' ')
    eq['zmagx2'], eq['dum2'], eq['sibdry2'], eq['dum3'], eq['dum4'] = np.fromstring(_sep_eq_line(lines.pop(0)), sep=' ')

    # Remove dummy fields
    for ii in range(5):
        del eq[f'dum{ii}']

    # Check if duplicate fields are equal
    for base in ['simagx', 'rmagx', 'zmagx', 'sibdry']:
        assert eq.get(base) == eq.pop(base + '2'), f'Duplicate values for {base} not equal!'

    # Read 1D array blocks
    for name in ['fpol', 'pres', 'ffprime', 'pprime']:
        eq[name] = np.hstack(_read_chunk(lines, eq['nr']))

    # Read psi map
    eq['psi'] = np.hstack(_read_chunk(lines, eq['nr'] * eq['nz']))
    eq['psi'] = eq['psi'].reshape((eq['nz'], eq['nr']))

    # Read q-profile
    eq['qpsi'] = np.hstack(_read_chunk(lines, eq['nr']))

    # Read sizes of boundary vector and limiter vector
    header = lines.pop(0)
    eq['nbdry'] = int(header[:5])
    eq['nlim'] = int(header[5:])

    # Read boundary vector
    if eq['nbdry'] > 0:
        bbbs = _read_chunk(lines, eq['nbdry'] * 2)
        bbbs = np.hstack(bbbs).reshape((eq['nbdry'], 2))
        eq['rbdry'] = bbbs[:, 0]
        eq['zbdry'] = bbbs[:, 1]
        if eq['rbdry'][0] != eq['rbdry'][-1] and eq['zbdry'][0] != eq['zbdry'][-1]:
            eq['nbdry'] += 1
            eq['rbdry'] = np.concatenate([eq['rbdry'], [eq['rbdry'][0]]])
            eq['zbdry'] = np.concatenate([eq['zbdry'], [eq['zbdry'][0]]])
    else:
        eq['rbdry'] = None
        eq['zbdry'] = None

    # Read limiter vector
    if eq['nlim'] > 0:
        lim = _read_chunk(lines, eq['nlim'] * 2)
        lim = np.hstack(lim).reshape((eq['nlim'], 2))
        eq['rlim'] = lim[:, 0]
        eq['zlim'] = lim[:, 1]
        if eq['rlim'][0] != eq['rlim'][-1] and eq['zlim'][0] != eq['zlim'][-1]:
            eq['nlim'] += 1
            eq['rlim'] = np.concatenate([eq['rlim'], [eq['rlim'][0]]])
            eq['zlim'] = np.concatenate([eq['zlim'], [eq['zlim'][0]]])
    else:
        eq['rlim'] = None
        eq['zlim'] = None

    return eq


def write_eqdsk_file(
    fname,
    nr=0,
    nz=0,
    rdim=0.0,
    zdim=0.0,
    rcentr=0.0,
    rleft=0.0,
    zmid=0.0,
    rmagx=0.0,
    zmagx=0.0,
    simagx=0.0,
    sibdry=0.0,
    bcentr=0.0,
    cpasma=0.0,
    fpol=[],
    pres=[],
    ffprime=[],
    pprime=[],
    psi=[],
    qpsi=[],
    nbdry=None,
    nlim=None,
    rbdry=[],
    zbdry=[],
    rlim=[],
    zlim=[],
    gcase=None,
    gid=None,
    **kwargs,
):
    """
    Writes provided equilibrium data into the EQDSK format.

    :arg fname: str. Name of the EQDSK file to be generated.

    :arg nr: int. Number of radial (width) points in 2D grid and in 1D profiles, assumed equal to each other.

    :arg nz: int. Number of vertical (height) points in 2D grid.

    :arg rdim: float. Width of 2D grid box, in the radial (width) direction.

    :arg zdim: float. Height of 2D grid box, in the vertical (height) direction.

    :arg rcentr: float. Location of the geometric center of the machine in the radial direction, does not necessarily need to be mid-point in radial direction of 2D grid.

    :arg rleft: float. Location of the left-most point in radial direction (lowest radial value) of 2D grid, needed for grid reconstruction.

    :arg zmid: float. Location of the mid-point in vertical direction of 2D grid, needed for grid reconstruction.

    :arg rmagx: float. Location of the magnetic center of the equilibrium in the radial direction.

    :arg zmagx: float. Location of the magnetic center of the equilibrium in the vertical direction.

    :arg simagx: float. Value of the poloidal flux at the magnetic center of the equilibrium.

    :arg sibdry: float. Value of the poloidal flux at the boundary of the equilibrium, defined as the last closed flux surface and not necessarily corresponding to any edge of the 2D grid.

    :arg bcentr: float. Value of the toroidal magnetic field at the geometric center of the machine, typically provided as the value in vacuum.

    :arg cpasma: float. Value of the total current in the plasma, typically provided as the total current in the toroidal direction.

    :arg fpol: array. Absolute unnormalized poloidal flux as a function of radius.

    :arg pres: array. Total plasma pressure as a function of radius.

    :arg ffprime: array. F * derivative of F with respect to normalized poloidal flux as a function of radius.

    :arg pprime: array. Derivative of plasma pressure with respect to normalized poloidal flux as a function of radius.

    :arg psi: array. 2D poloidal flux map as a function of radial coordinate and vertical coordinate.

    :arg qpsi: array. Safety factor as a function of radius.

    :arg nbdry: int. Number of points in description of plasma boundary contour, can be zero.

    :arg nlim: int. Number of points in description of plasma limiter contour, can be zero.

    :arg rbdry: array. Ordered list of radial values corresponding to points in the plasma boundary contour description.

    :arg zbdry: array. Ordered list of vertical values corresponding to points in the plasma boundary contour description.

    :arg rlim: array. Ordered list of radial values corresponding to points in the plasma boundary contour description.

    :arg zlim: array. Ordered list of vertical values corresponding to points in the plasma boundary contour description.

    :kwarg gcase: str. String to identify file, non-essential and written into the 48 character space at the start of the file.

    :kwarg gid: int. Dummy integer value to identify file origin, non-essential and is sometimes used to identify the FORTRAN output number.

    :returns: none.
    """
    errmsg = 'g-eqdsk file write aborted.'
    assert isinstance(fname, (str, Path)), f'fname field must be a string. {errmsg}'
    assert isinstance(nr, int), f'nr field must be an integer. {errmsg}'
    assert isinstance(nz, int), f'nz field must be an integer. {errmsg}'
    assert isinstance(rdim, float), f'rdim field must be a real number. {errmsg}'
    assert isinstance(zdim, float), f'zdim field must be a real number. {errmsg}'
    assert isinstance(rcentr, float), f'rcentr field must be a real number. {errmsg}'
    assert isinstance(rleft, float), f'rleft field must be a real number. {errmsg}'
    assert isinstance(zmid, float), f'zmid field must be a real number. {errmsg}'
    assert isinstance(rmagx, float), f'rmagx field must be a real number. {errmsg}'
    assert isinstance(zmagx, float), f'zmagx field must be a real number. {errmsg}'
    assert isinstance(simagx, float), f'simagx field must be a real number. {errmsg}'
    assert isinstance(sibdry, float), f'sibdry field must be a real number. {errmsg}'
    assert isinstance(bcentr, float), f'bcentr field must be a real number. {errmsg}'
    assert isinstance(cpasma, float), f'cpasma field must be a real number. {errmsg}'
    assert isinstance(fpol, array_types), f'fpol field must be an array. {errmsg}'
    assert isinstance(pres, array_types), f'pres field must be an array. {errmsg}'
    assert isinstance(ffprime, array_types), f'ffprime field must be an array. {errmsg}'
    assert isinstance(pprime, array_types), f'pprime field must be an array. {errmsg}'
    assert isinstance(psi, array_types), f'psi field must be an array. {errmsg}'
    assert isinstance(qpsi, array_types), f'qpsi field must be an array. {errmsg}'
    if nbdry is not None:
        assert isinstance(nbdry, int), f'nbdry field must be an integer or set to None. {errmsg}'
        assert isinstance(rbdry, array_types), f'rbdry field must be an array if nbry is not None. {errmsg}'
        assert isinstance(zbdry, array_types), f'zbdry field must be an array if nbry is not None. {errmsg}'
    if nlim is not None:
        assert isinstance(nlim, int), f'nlim field must be an integer or set to None. {errmsg}'
        assert isinstance(rlim, array_types), f'rlim field must be an array if nlim is not None. {errmsg}'
        assert isinstance(zlim, array_types), f'zlim field must be an array if nlim is not None. {errmsg}'
    if gcase is not None:
        assert isinstance(gcase, str), f'gcase field must be a string or set to None. {errmsg}'
    if gid is not None:
        assert isinstance(gid, int), f'gid field must be an integer or set to None. {errmsg}'

    fpath = Path(fname)
    if fpath.exists():
        print(f'{fpath} exists, overwriting file with g-eqdsk file!')
    if nbdry is None or rbdry is None or zbdry is None:
        nbdry = 0
        rbdry = []
        zbdry = []
    if nlim is None or rlim is None or zlim is None:
        nlim = 0
        rlim = []
        zlim = []
    if gcase is None:
        gcase = ''
    if gid is None:
        gid = 0
    dummy = 0.0

    with open(fpath, 'w') as ff:
        gcase = case[:48] if len(case) > 48 else case
        ff.write(f'{gcase:-48s}{idum:4d}{nr:4d}{nz:4d}\n')
        ff.write(f'{rdim:16.9E}{zdim:16.9E}{rcentr:16.9E}{rleft:16.9E}{zmid:16.9E}\n')
        ff.write(f'{rmagx:16.9E}{zmagx:16.9E}{simagx:16.9E}{sibdry:16.9E}{bcentr:16.9E}\n')
        ff.write(f'{cpasma:16.9E}{simagx:16.9E}{dummy:16.9E}{rmagx:16.9E}{dummy:16.9E}\n')
        ff.write(f'{zmagx:16.9E}{dummy:16.9E}{sibdry:16.9E}{dummy:16.9E}{dummy:16.9E}\n')
        for ii in range(len(fpol)):
            ff.write(f'{fpol[ii]:16.9E}')
            if (ii + 1) % 5 == 0 and (ii + 1) != len(fpol):
                ff.write('\n')
        ff.write('\n')
        for ii in range(len(pres)):
            ff.write(f'{pres[ii]:16.9E}')
            if (ii + 1) % 5 == 0 and (ii + 1) != len(pres):
                ff.write('\n')
        ff.write('\n')
        for ii in range(len(ffprime)):
            ff.write(f'{ffprime[ii]:16.9E}')
            if (ii + 1) % 5 == 0 and (ii + 1) != len(ffprime):
                ff.write('\n')
        ff.write('\n')
        for ii in range(len(pprime)):
            ff.write(f'{pprime[ii]:16.9E}')
            if (ii + 1) % 5 == 0 and (ii + 1) != len(pprime):
                ff.write('\n')
        ff.write('\n')
        kk = 0
        for ii in range(nr):
            for jj in range(nz):
                ff.write(f'{psi[ii, jj]:16.9E}')
                if (kk + 1) % 5 == 0 and (kk + 1) != (nr * nz):
                    ff.write('\n')
                kk += 1
        ff.write('\n')
        for ii in range(len(qpsi)):
            ff.write(f'{qpsi[ii]:16.9E}')
            if (ii + 1) % 5 == 0 and (ii + 1) != len(q):
                ff.write('\n')
        ff.write('\n')
        ff.write(f'{nbdry:5d}{nlim:5d}\n')
        kk = 0
        for ii in range(nbdry):
            ff.write(f'{rbdry[ii]:16.9E}')
            if (kk + 1) % 5 == 0 and (ii + 1) != nbry:
                ff.write('\n')
            kk += 1
            ff.write(f'{zbdry[ii]:16.9E}')
            if (kk + 1) % 5 == 0 and (ii + 1) != nbry:
                ff.write('\n')
            kk += 1
        if kk > 0:
            ff.write('\n')
        kk = 0
        for ii in range(nlim):
            ff.write(f'{rlim[ii]:16.9E}')
            if (kk + 1) % 5 == 0 and (kk + 1) != nlim:
                ff.write('\n')
            kk += 1
            ff.write(f'{zlim[ii]:16.9E}')
            if (kk + 1) % 5 == 0 and (kk + 1) != limitr:
                ff.write('\n')
            kk += 1
    print(f'Output EQDSK file saved as {fpath}.')

