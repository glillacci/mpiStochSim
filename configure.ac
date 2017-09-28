#	Autoconf script for
#
#	mpiStochSim
#
# 
#	Created by Gabriele Lillacci in December 2011.
#	Latest revision: December 2011.
#
#
#	This free software is available under the Creative Commons Attribution Non-Commercial Share Alike License.
#	You are permitted to use, share, copy, redistribute and adapt this software as long as appropriate credit
#	is given to the original author, all derivative works are distributed under the same license or a compatible one,
#	and this software and its derivatives are not used for commercial purposes.
#	For more information, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or contact
#	Creative Commons, 171 2nd Street, Suite 300, San Francisco, California, 94105, USA.


# Require a minimum version of autoconf
AC_PREREQ(2.59)
# Initialize autoconf
AC_INIT(mpiStochSim, 1.0)

AC_CANONICAL_SYSTEM

# Initialize automake
AM_INIT_AUTOMAKE()

# Check for the mpicc compiler
AC_PROG_CC(mpicc)
AC_PROG_CC_C99

# Check for required libraries
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])
AC_CHECK_LIB([parestlib],[rxn_sample_path_alloc])
AC_CHECK_LIB([stochmod],[syncirc_mod_setup])
AC_CHECK_LIB([xml2],[xmlReadFile])

# Declare the makefiles
AC_CONFIG_FILES(Makefile mpistochsim/Makefile mpidcmsim/Makefile mpibatchsim/Makefile)
# Write the output files
AC_OUTPUT

