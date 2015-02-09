#include <Python.h>
#include <numpy/arrayobject.h>

/* Access to i-th value of array a:                                         */
#define INDd(a,i) *((double *)(a->data + i*a->strides[0]))
#define INDi(a,i) *((int    *)(a->data + i*a->strides[0]))
#define IND2d(a,i,j) *((double *)(a->data + i*a->strides[0] + j*a->strides[1]))
#define IND2i(a,i,j) *((int    *)(a->data + i*a->strides[0] + j*a->strides[1]))

#include "utils.h"

#define PI      (3.141592653589793)      /* PI                              */
#define LS      (2.99792458e10)          /* Light Speed (cm / s)            */
#define KB      (1.380658e-16)           /* Boltzmann constant (erg / K)    */
#define AMU     (1.66053886e-24)         /* Atomic Mass unit (g)            */
#define SQRTLN2 (0.83255461115769775635) /* sqrt(ln(2))                     */
#define H (6.6260755e-27)       /* Planck's constant (erg * s)              */
#define EC (4.8032068e-10)      /* electronic charge (statcoulomb)          */
#define ME (9.1093897e-28)      /* Electron mass (g)                        */

#define SIGCTE  (PI*EC*EC/LS/LS/ME/AMU)
#define EXPCTE  (H*LS/KB)

#define FDOP (3.581175136e-7)  /* sqrt(2*KB/AMU) * sqrt(ln2) / c  */
#define FLOR (1.461466451e17)  /* sqrt(2*KB/pi/AMU) / (AMU*c)     */

PyDoc_STRVAR(extinction__doc__,
"Calculate the extinction-coefficient for each molecule, at a given\n\
pressure and temperature, over a wavenumber range.                 \n\
                                                                           \n\
Parameters:                                                                \n\
-----------                                                                \n\
ext: 2D float ndarray                    \n\
profile: 1D float ndarray                    \n\
psize: 2D integer ndarray                    \n\
pindex: 2D integer ndarray                    \n\
lorentz: 1D Float ndarray                    \n\
doppler: 1D Float ndarray                    \n\
wn: 1D Float ndarray                    \n\
own: 1D Float ndarray                    \n\
divisors: 1D integer ndarray                    \n\
moldensity: 1D Float ndarray                    \n\
molq: 1D Float ndarray                    \n\
molrad: 1D Float ndarray                    \n\
molmass: 1D Float ndarray                    \n\
isoimol: 1D Float ndarray                    \n\
isomass: 1D Float ndarray                    \n\
isoratio: 1D Float ndarray                    \n\
isoz: 1D Float ndarray                    \n\
isoiext: 1D Float ndarray                    \n\
lwn: 1D Float ndarray                    \n\
elow: 1D Float ndarray                    \n\
gf: 1D Float ndarray                    \n\
lID: 1D integer ndarray                    \n\
ethresh: Float                    \n\
pressure: Float                    \n\
temp: Float                    \n\
sum: Integer                    \n\
                    \n\
Developers:                                                                \n\
-----------                                                                \n\
Patricio Rojo      U Cornell  pato@astro.cornell.edu (pato@oan.cl)         \n\
Patricio Cubillos  UCF        pcubillos@fulbrightmail.org                  \n\
                                                                           \n\
Modification History:                                                      \n\
---------------------                                                      \n\
2006        p. rojo      Writen as a component of the transit package.     \n\
2014-02-08  p. cubillos  Modified for use with the pyrat project.");

static PyObject *extinction(PyObject *self, PyObject *args){
  PyArrayObject *ext,                             /* Extinction coefficient */
           *profile, *psize, *pindex,
           *lorentz, *doppler,                    /* Voigt data             */
           *wn, *own, *divisors,                  /* Wavenumber data        */
           *moldensity, *molq, *molrad, *molmass, /* Molecular data         */
           *isoimol, *isomass, *isoratio,
           *isoz, *isoiext,                       /* Isotopic data          */
           *lwn, *lID, *elow, *gf;                /* Line transition data   */

  int next, nLor, nDop,
      nmol, niso,
      nlines,
      dnwn, onwn, ndivs, iown, idwn, offset, subw,
      imol, ofactor, iprof,
      nadd=0, nskip=0, neval=0, minj, maxj,
      sum=0;
  int i, j, m, ln; /* Auxilliary for-loop indices    */
  double pressure, temp, csdiameter, density, minwidth=1e5, vwidth, ethresh,
         florentz, fdoppler, wnstep, ownstep, dwnstep, wavn, next_wn, k;
  double *alphal, *alphad, *kmax, **ktmp;
  int *idop, *ilor;

  /* Load inputs:                                                           */
  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOOOOOOddd|i",
                              &ext,
                              &profile, &psize, &pindex, &lorentz, &doppler,
                              &wn, &own, &divisors,
                              &moldensity, &molq, &molrad, &molmass,
                              &isoimol, &isomass, &isoratio, &isoz, &isoiext,
                              &lwn, &elow, &gf, &lID,
                              &ethresh, &pressure, &temp, &sum))
    return NULL;

  nLor   = lorentz->dimensions[0];  /* Number of Lorentz widths             */
  nDop   = doppler->dimensions[0];  /* Number of Doppler widths             */
  nmol   = molmass->dimensions[0];  /* Number of species                    */
  niso   = isomass->dimensions[0];  /* Number of isotopes                   */
  ndivs  = divisors->dimensions[0]; /* Number of divisors of osamp          */
  onwn   = own->dimensions[0];      /* Number of fine-wavenumber samples    */
  nlines = lwn->dimensions[0];      /* Number of line transitions           */
  next   = ext->dimensions[0];      /* Number of extinction-coef. species   */

  if (sum)
    next = 1;

  /* Constant factors for line widths:                                      */
  fdoppler = sqrt(2*KB*temp/AMU) * SQRTLN2 / LS;
  florentz = sqrt(2*KB*temp/PI/AMU) / (AMU*LS);
  //printf("Florentz: %.6e, Fdoppler: %.6e\n", florentz, fdoppler);

  /* Allocate alpha Lorentz and Doppler arrays:                             */
  alphal = (double *)calloc(niso, sizeof(double));
  alphad = (double *)calloc(niso, sizeof(double));

  /* Array to hold maximum line strength:                                   */
  kmax = (double *)calloc(next, sizeof(double));

  /* Allocate width indices array:                                          */
  idop = (int *)calloc(niso, sizeof(int));
  ilor = (int *)calloc(niso, sizeof(int));

  ktmp    = (double **)calloc(next,      sizeof(double *));
  ktmp[0] = (double  *)calloc(next*onwn, sizeof(double  ));
  for (i=1; i<next; i++)
    ktmp[i] = ktmp[0] + onwn;

  /* Calculate the isotopes' widths:                                        */
  for (i=0; i<niso; i++){
    imol = INDi(isoimol, i);
    /* Lorentz profile width:                                               */
    alphal[i] = 0.0;
    for(j=0; j<nmol; j++){
      /* Isotope's collision diameter with j-th species:                    */
      csdiameter = INDd(molrad, imol) + INDd(molrad, j);
      /* Number density of molecule colliding with current isotope:         */
      density = AMU * INDd(molq,j) * pressure / (KB*temp);
      /* Line width:                                                        */
      alphal[i] += density * csdiameter * csdiameter *
                   sqrt(1/INDd(isomass,i) + 1/INDd(molmass,j));
      if (i==-1)
        printf("j:%d,  %.6e,  %.6e,  %.6e\n", j,
                     density, csdiameter, INDd(molmass,j));
    }
    alphal[i] *= florentz;

    /* Doppler profile width (divided by central wavenumber):               */
    alphad[i] = fdoppler / sqrt(INDd(isomass,i));
    /* Print Lorentz and Doppler broadening widths:                         */
    if(i <= 0)
      //printf("Imass: %.4f, imol: %d\n", INDd(isomass,i), imol);
      printf("Lorentz: %.6e cm-1, Doppler: %.6e cm-1 (T=%.1f, "
             "p=%.2e).\n", alphal[i], alphad[i]*INDd(wn,0), temp, pressure);

    /* Estimate the Voigt width:                                            */
    vwidth = 0.5346*alphal[i] + sqrt(pow(alphal[i], 2)*0.2166    +
                                     pow(alphad[i]*INDd(wn,0),2) );
    minwidth = fmin(minwidth, vwidth);

    /* Search for aDop and aLor indices for alphal[i] and alphad[i]:        */
    idop[i] = binsearchapprox(doppler, alphad[i]*INDd(wn,0), 0, nDop);
    ilor[i] = binsearchapprox(lorentz, alphal[i],            0, nLor);
  }

  wnstep  = INDd(wn, 1) - INDd(wn, 0);
  ownstep = INDd(own,1) - INDd(own,0);
  /* Set the wavenumber sampling resolution:                                */
  for (i=1; i<ndivs; i++)
    if (INDi(divisors,i)*ownstep >= 0.5 * minwidth){
      break;
    }
  ofactor = INDi(divisors,(i-1));   /* Dynamic-sampling oversampling factor */
  dwnstep = ownstep * ofactor;      /* Dynamic-sampling grid stepsize       */
  dnwn    = 1 + (onwn-1) / ofactor; /* Number of dynamic-sampling values    */
  printf("Dynamic-sampling grid interval: %.4e "
         "(factor:%i, minwidth:%.3e)\n", dwnstep, ofactor, minwidth);
  printf("Number of dynamic-sampling values: %d\n", dnwn);

  /* Find the maximum line-strength per molecule:                           */
  for (ln=0; ln<nlines; ln++){
    wavn = INDd(lwn, ln);    /* Wavenumber of line transition               */
    i    = INDi(lID, ln);    /* Isotope index of line transition:           */
    m    = INDi(isoiext, i); /* Extinction-coefficient table index          */
    if (sum)
      m = 0;  /* Collapse everything into the first index                   */

    /* If this line falls beyond limits, skip to next line transition:      */
    if ((wavn < INDd(own,0)) || (wavn > INDd(own,(onwn-1))))
      continue;

    /* Calculate the line strength divided by the molecular abundance:      */
    k = INDd(isoratio,i)                   *  /* Density                    */
        SIGCTE * INDd(gf,ln)               *  /* Constant * gf              */
        exp(-EXPCTE*INDd(elow,ln) / temp)  *  /* Level population           */
        (1-exp(-EXPCTE*wavn/temp))         /  /* Induced emission           */
        INDd(isomass,i)                    /  /* Isotope mass               */
        INDd(isoz,i);                         /* Partition function         */
    /* Check if this is the maximum k:                                      */
    kmax[m] = fmax(kmax[m], k);
  }
  //for (i=0; i<next; i++)
  //  printf("Kmax[%d] is: %.3e\n", i, kmax[i]);

  /* Compute the extinction-coefficient for each species:                   */ 
  for(ln=0; ln<nlines; ln++){
    wavn = INDd(lwn, ln);    /* Wavenumber                                  */
    i    = INDi(lID, ln);    /* Isotope index                               */
    m    = INDi(isoiext, i); /* extinction-coefficient table index          */
    if (sum)
      m = 0;  /* Collapse everything into the first index                   */

    if ((wavn < INDd(own,0)) || (wavn > INDd(own,(onwn-1))))
      continue;

    /* Line strength:                                                       */
    k = INDd(isoratio,i) * SIGCTE * INDd(gf,ln)  *
        exp(-EXPCTE * INDd(elow,ln) / temp)      *
        (1 - exp(-EXPCTE * wavn / temp))         /
        (INDd(isomass,i) * INDd(isoz,i));

    /* Index of closest oversampled wavenumber:                             */
    iown = (wavn - INDd(own,0))/ownstep;
    if (fabs(wavn - INDd(own,(iown+1))) < fabs(wavn - INDd(own,iown)))
      iown++;

    /* Check if the next line falls on the same sampling index:             */
    while (ln+1 != nlines && INDi(lID, (ln+1)) == i){
      next_wn = INDd(lwn, (ln+1));
      if (fabs(next_wn - INDd(own,iown)) < ownstep){
        nadd++;
        ln++;
        /* Add the contribution from this line into the opacity:            */
        k += INDd(isoratio,i) * SIGCTE * INDd(gf,ln) *
             exp(-EXPCTE * INDd(elow,ln) / temp)     *
             (1 - exp(-EXPCTE * next_wn / temp))     /
             (INDd(isomass,i) * INDd(isoz,i));
      }
      else
        break;
    }
    /* Multiply by the species' density:                                    */
    if (sum)
      k *= INDd(moldensity, (INDi(isoimol,i)));

    /* Skip weakly contributing lines:                                      */
    if (k < ethresh * kmax[m]){
      nskip++;
      continue;
    }

    /* Index of closest (but not larger than) dynamic-sampling wavenumber:  */
    idwn = (wavn - INDd(own,0))/dwnstep;

    /* FINDME: de-hard code this threshold                                  */
    if (alphad[i]*wavn/alphal[i] >= 1e-1){
      /* Recalculate index for Doppler width:                               */
      idop[i] = binsearchapprox(doppler, alphad[i]*wavn, 0, nDop);
    }

    /* Sub-sampling offset between center of line and dyn-sampled wn:       */
    subw = iown - idwn*ofactor;
    /* Offset between the profile and the wavenumber-array indices:         */
    offset = ofactor*idwn - IND2i(psize, idop[i], ilor[i]) + subw;
    /* Range that contributes to the opacity:                               */
    /* Set the lower and upper indices of the profile to be used:           */
    minj = idwn - (IND2i(psize, idop[i], ilor[i]) - subw) / ofactor;
    maxj = idwn + (IND2i(psize, idop[i], ilor[i]) + subw) / ofactor;
    if (minj < 0)
      minj = 0;
    if (maxj > dnwn)
      maxj = dnwn;

    /* Add the contribution from this line to the opacity spectrum:         */
    iprof = IND2i(pindex, idop[i], ilor[i]);
    //printf("minj: %d, maxj: %d, prof0: %d, offset: %d, ofactor: %d\n",
    //       minj, maxj, iprof, offset, ofactor);
    for(j=minj; j<maxj; j++){
      //transitprint(1, 2, "%i  ", j-offset);
      //transitprint(1, 2, "j=%d, p[%i]=%.2g   ", j, j-offset,
      //                    profile[idop[i]][ilor[i]][subw][j-offset]);
      //printf("  iprof: %d\n", iprof + ofactor*j - offset);
      //printf("  Val: %.3e\n", INDd(profile, (iprof + ofactor*j - offset)));
      ktmp[m][j] += k * INDd(profile, (iprof + ofactor*j - offset));
    }
    neval++;
  }
  //printf("Downsample now: (%f, %d)\n", wnstep/ownstep, ofactor);

  /* Downsample ktmp to the final sampling size:                            */
  for (m=0; m<next; m++){
    //printf("IN [0,N]: {%.3e, %.3e}\n", ktmp[m][0], ktmp[m][dnwn-1]);
    downsample(ktmp, ext, dnwn, wnstep/ownstep/ofactor, m);
  }

  /* Free the no-longer used memory:                                        */
  free(alphal);
  free(alphad);
  free(kmax);
  free(idop);
  free(ilor);
  free(ktmp[0]);
  free(ktmp);

  return Py_BuildValue("i", 1);
}


/* The module doc string    */
PyDoc_STRVAR(extcoeff__doc__, "Python wrapper for the extinction-coefficient "
                              "calculation.");

/* A list of all the methods defined by this module.                        */
static PyMethodDef extcoeff_methods[] = {
    {"extinction", extinction, METH_VARARGS, extinction__doc__},
    {NULL,         NULL,       0,            NULL}              /* sentinel */
};


/* When Python imports a C module named 'X' it loads the module */
/* then looks for a method named "init"+X and calls it.         */
void initextcoeff(void){
  Py_InitModule3("extcoeff", extcoeff_methods, extcoeff__doc__);
  import_array();
}