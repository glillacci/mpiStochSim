/*
 *  mpibatchsim.c
 *
 *  Batch parallel stochatic simulator with parameter resampling from
 *  posterior densities.
 *
 *
 *  This file is part of mpiStochSim.
 *  Copyright 2011-2017 Gabriele Lillacci.
 *
 *  mpiStochSim is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  mpiStochSim is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with mpiStochSim.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Libxml2 includes
#include <libxml2/libxml/xmlversion.h>
#include <libxml2/libxml/xmlstring.h>
#include <libxml2/libxml/tree.h>
#include <libxml2/libxml/parser.h>
#include <libxml2/libxml/xpath.h>

#include <stochmod.h>
#include <parestlib.h>

// Tags for message passing
#define WORKTAG 1
#define DIETAG 0

// Global variable declarations
stochmod * model;
gsl_vector * times;
gsl_matrix * data;
gsl_matrix * parameters;
gsl_vector * weights;
gsl_vector * params;
gsl_matrix * out;
gsl_matrix * bg;
size_t M;					// Number of experimental samples to use for identification
size_t P;					// Number of measured species
size_t K;					// Number of time points
size_t R;					// Number of parameters
size_t nsam;				// Number of accepted particles in the posterior
double mufp;				// Mean fluorescence level
double sigfp;				// Sd of fluorescence level

// Functions for parallel simulation
static void master (int argc, char * argv[]);
static void slave ();

// Problem definition processing
static int processBackground (xmlDocPtr doc, gsl_matrix * bg);
static double sample_fl (const gsl_rng * r, double z, double mufp, double sigfp, gsl_matrix * bg);

int main (int argc, char * argv[])
{
	// ================================== USER-PROVIDED INFORMATION ==========================================

	// Choose the stochasitc model to estimate
	model = (stochmod *) malloc (sizeof (stochmod));
	if (model == NULL)
	{
		fprintf (stderr, "error in memory allocation\n");
		return EXIT_FAILURE;
	}
	lacgfp7_mod_setup (model);

	// Path of posterior file
	char * pfilename = "./mpibatchsim/res_lacgfp7_syn_1000";
	char * wfilename = "./mpibatchsim/res_lacgfp7_syn_1000_weights";

	// Set the number of particles in the posterior
	nsam = 5000;

	// ==================================== INITIALIZE LIBRARIES =============================================

	// Initialize MPI
	MPI_Init (&argc, &argv);

	// Initialize libxml2
	LIBXML_TEST_VERSION

	// Set up times vector
	size_t ntimes = 5;
	double timesArr[] = {0, 2, 3, 4, 5};
	gsl_vector_view timesv = gsl_vector_view_array (timesArr, ntimes);
	times = gsl_vector_alloc (ntimes);
	gsl_vector_memcpy (times, &timesv.vector);

	// Allocate the parameter matrix and the weight vector
	parameters = gsl_matrix_alloc (nsam, 2*model->nout + model->nparams);
	weights = gsl_vector_alloc (nsam);

	// Load posterior of the parameters
	FILE * pfile = fopen (pfilename, "r");
	if (!pfile || (gsl_matrix_fscanf (pfile, parameters) != GSL_SUCCESS))
	{
		fprintf (stderr, "error: could not load parameter posterior distribution.\n");
		return EXIT_FAILURE;
	}
	fclose (pfile);

	// Load the weights
	FILE * wfile = fopen (wfilename, "r");
	if (!wfile || (gsl_vector_fscanf (wfile, weights) != GSL_SUCCESS))
	{
		fprintf (stderr, "error: could not load weights.\n");
		return EXIT_FAILURE;
	}
	fclose (wfile);

	// Renormalize the weights
	double sow = gsl_blas_dasum (weights);
	gsl_vector_scale (weights, 1/sow);

	// Set up a vector that contains the resampled parameter set
	params = gsl_vector_alloc (model->nparams + model->nin);


	// ================================= PROCESS COMMAND-LINE ARGUMENTS =========================================

	// Find out my identity in the default communicator
	int myrank;
	MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

	// Check correctness of command line arguments
	if (argc != 5)
	{
		if (myrank == 0)
			printf ("Usage: %s problem-file data-file nsim ntimes\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	// Put command line arguments into correct variables
	char * problemfile = argv[1];
	char * datafile = argv[2];
	size_t nsim = atoi (argv[3]);
	size_t nbatch = atoi (argv[4]);

	// Inform the user that we are ready to start
	if (myrank == 0)
	{
		printf ("\nThis is mpibatchsim v1.0\nI am the master process\n\n");
		printf ("Problem definition file:\n%s\n", problemfile);
		printf ("Data file:\n%s\n", datafile);
		printf ("Simulating %s for %d simulations, %d times.\n\n", model->name, (int) nsim, (int) nbatch);
	}

	// =============================== EXTRACT INFO FROM PROBLEM FILE ========================================
	// Parse the problem file
	xmlDocPtr problem = xmlReadFile (problemfile, NULL, 0);
	if (problem == NULL)
	{
		fprintf (stderr, "error reading problem file\n");
		return EXIT_FAILURE;
	}

	// Initialize XPath evaluation context
	xmlXPathContextPtr ctxt = xmlXPathNewContext (problem);

	// Initialize a xmlXPathObjectPtr object
	xmlXPathObjectPtr xobj;

	// Use XPath to find the number of samples to use for identification
	xobj = xmlXPathEvalExpression ((xmlChar *) "/problem/samples", ctxt);
	M = (size_t) xmlXPathCastToNumber (xobj);
	xmlXPathFreeObject (xobj);

	// Use XPath to find the number of measured species
	xobj = xmlXPathEvalExpression ((xmlChar *) "/problem/outputs", ctxt);
	P = (size_t) xmlXPathCastToNumber (xobj);
	xmlXPathFreeObject (xobj);

	// Use XPath to find the number of time points
	xobj = xmlXPathEvalExpression ((xmlChar *) "/problem/timepoints", ctxt);
	K = (size_t) xmlXPathCastToNumber (xobj);
	xmlXPathFreeObject (xobj);

	// Use XPath to find the number of parameters
	xobj = xmlXPathEvalExpression ((xmlChar *) "/problem/parameters", ctxt);
	R = (size_t) xmlXPathCastToNumber (xobj);
	xmlXPathFreeObject (xobj);

	// Set up background matrix
	bg = gsl_matrix_alloc (M, P);
	if (processBackground (problem, bg) != GSL_SUCCESS)
	{
		fprintf (stderr, "error: could not process data set information\n");
		return EXIT_FAILURE;
	}
	// gsl_matrix_set_zero (bg);

	// Set up output matrix
	out = gsl_matrix_calloc (model->nout, model->nspecies);
	model->output (out);


	// ============================ RUN THE PARALLEL COMPUTATION ==================================
	// Start a new clock to keep track of elapsed time
	clock_t tic = clock ();

	// Decide what I have to do
	if (myrank == 0)
	{
		// I am in charge
		master (argc, argv);
	}
	else
	{
		// I have to do what I am told
		slave ();
	}

	// Check how much time has passed
	clock_t toc = clock ();
	if (myrank == 0)
		printf ("\n\nSimulations completed.\nElapsed time: %g sec.\n\n", (((double) (toc - tic)) / CLOCKS_PER_SEC));

	// Shut down MPI
	MPI_Finalize ();

	// Clean up
	free (model);
	gsl_matrix_free (out);
	gsl_matrix_free (data);
	gsl_matrix_free (bg);
	gsl_vector_free (times);
	gsl_vector_free (params);

	return EXIT_SUCCESS;
}


static void master (int argc, char * argv[])
{
	int ntasks, rank, count = 0;
	double work;
	MPI_Status status;
	size_t res_no = (model->nspecies)*(times->size);
	double results[res_no];
	size_t nsim = atoi (argv[3]);
	size_t nbatch = atoi (argv[4]);
	double thetac[model->nparams + model->nin];

	// Set up the random number generator
	gsl_rng_env_setup ();
	gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
	gsl_rng_set (r, (unsigned long int) time (NULL));

	// Find out how many processes there are in the default communicator
	MPI_Comm_size (MPI_COMM_WORLD, &ntasks);

	// Allocate rxn_ensemble object to store the results
	rxn_ensemble * res = rxn_ensemble_alloc (nsim, model->nspecies, times->size);

	// Allocate counts matrix
	gsl_matrix * counts = gsl_matrix_calloc (res->nreplic, model->nout*res->ntimes);


	// Loop for the batch size
	for (size_t i = 0; i < nbatch; i++)
	{
		// Reset count
		count = 0;

		// Resample a parameter set from the posterior (with weights)

		// Generate a uniform random number
		double u = gsl_rng_uniform (r);

		// Accumulate the weights until they are larger than u
		double mu = 0.0;
		size_t ix = 0;
		while ((mu < u) && (ix < (nsam-1)))
		{
			mu += gsl_vector_get (weights, ix);
			ix++;
		}

		// Set mufp and sigfp
		mufp = gsl_matrix_get (parameters, ix, 0);
		sigfp = gsl_matrix_get (parameters, ix, 1);

		// Take the particle corresponding to ix
		for (size_t j = 0; j < model->nparams; j++)
		{
			thetac[j] = gsl_matrix_get (parameters, ix, j + 2*model->nout);
		}

		// Set the value of the input
		thetac[R] = 10;


		// Seed the slaves; send one unit of work to each slave.
		for (rank = 1; rank < ntasks; rank++)
		{
			MPI_Send (thetac,             					  // message buffer
					model->nparams + model->nin,                 // # of data items
					MPI_DOUBLE,       					 	// data items are of type double
					rank,            						  // destination process rank
					WORKTAG + count,           				// user chosen message tag
					MPI_COMM_WORLD);   						// default communicator

			count++;
		}

		// Loop over getting new work requests until there is no more work to be done
		while (count < nsim)
		{
			// Receive results from a slave
			MPI_Recv (results,      	   // message buffer
					res_no,              // # of data items
					MPI_DOUBLE,          // of type double real
					MPI_ANY_SOURCE,      // receive from any sender
					MPI_ANY_TAG,     	   // any type of message
					MPI_COMM_WORLD,      // default communicator
					&status);            // info about the received message

			// Process the results
			for (size_t k = 0; k < res_no; k++)
			{
				res->data[status.MPI_TAG]->counts->data[k] = results[k];
			}

			// Send the slave a new work unit
			MPI_Send (thetac,             					  // message buffer
					model->nparams + model->nin,                 // # of data items
					MPI_DOUBLE,       					 	// data items are of type double
					status.MPI_SOURCE,            			// slave we just received from
					WORKTAG + count,           				// user chosen message tag
					MPI_COMM_WORLD);   						// default communicator

			// Get the next unit of work to be done
			count++;
		}

		// There's no more work to be done, so receive all the outstanding results from the slaves
		for (rank = 1; rank < ntasks; rank++)
		{
			MPI_Recv (results, res_no, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			// Process the results
			for (size_t k = 0; k < res_no; k++)
			{
				res->data[status.MPI_TAG]->counts->data[k] = results[k];
			}
		}


		// Extract the output
		rxn_ensemble_counts (res, counts, out);

		// Compute the flourescence levels
		for (size_t l = 0; l < counts->size1; l++)
		{
			for (size_t m = 0; m < counts->size2; m++)
			{
				// Current count
				double z = gsl_matrix_get (counts, l, m);

				// Generate a fluorescence level
				gsl_matrix_set (counts, l, m, sample_fl (r, z, mufp, sigfp, bg));
			}
		}


		// Write simulation data to file
		char datafile2[500];
		sprintf (datafile2, "%s_%d", argv[2], (int) i+1);

		FILE * datafile = fopen (datafile2, "w");
		if (datafile == NULL)
		{
			printf (">> error opening file for writing.. program exiting\n");
			return;
		}
		if (gsl_matrix_fprintf (datafile, counts, "%f")) {
			printf (">> error writing simulation data.. program exiting\n");
			return;
		}
		fclose (datafile);

		printf("Written data to file:\n%s\n", datafile2);
	}

	// Tell all the slaves to exit by sending an empty message with the DIETAG
	for (rank = 1; rank < ntasks; rank++)
	{
		MPI_Send (0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
	}

	// Free manually allocated resources
	rxn_ensemble_free (res);
	gsl_matrix_free (counts);
	gsl_rng_free (r);

	return;
}


static void slave ()
{
	// Declare some variables
	int myrank;
	size_t res_no = (model->nspecies)*(times->size);
	double pars[model->nparams + model->nin], results[res_no];
	MPI_Status status;

	// Find my rank
	MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

	// Set up the random number generator
	gsl_rng_env_setup ();
	gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
	gsl_rng_set (r, (unsigned long int) time (NULL) + myrank);

	// Allocate necessary objects
	gsl_vector * X0 = gsl_vector_alloc (model->nspecies);
	rxn_sample_path * rsp = rxn_sample_path_alloc (model->nspecies, times->size);
	gsl_vector_view params;

	while (1)
	{
		// Receive a message from the master
		MPI_Recv (pars, model->nparams + model->nin, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// Check the tag of the received message
		if (status.MPI_TAG == DIETAG)
		{
			printf ("This is process %d, I am exiting.\n", myrank);
			break;
		}

		// Create the vector view with the received parameters
		params = gsl_vector_view_array (pars, model->nparams + model->nin);

		// Sample a new initial state
		model->initial (X0, r);

		// Do the work
		ssa_direct_trajectory (model, &params.vector, times, X0, rsp, r);

		// Communicate that the simulation has been done
		// printf ("This is slave process %d, completed simulation %d.\n", myrank, status.MPI_TAG);

		// Send the result back
		MPI_Send (rsp->counts->data, res_no, MPI_DOUBLE, 0, status.MPI_TAG - WORKTAG, MPI_COMM_WORLD);
	}

	// Free manually allocated resources
	gsl_rng_free (r);
	gsl_vector_free (X0);
	rxn_sample_path_free (rsp);

	return;
}


// Sample a fluorescence level given the background and the count
static double sample_fl (const gsl_rng * r, double z, double mufp, double sigfp, gsl_matrix * bg)
{
	double F = 0.0;

	while (F <= 0)
	{
		double sambg = 1.0;
		if (bg)
		{
			sambg = gsl_matrix_get (bg, gsl_rng_uniform_int (r, M), 0);
		}

		double samfl = mufp*z + gsl_ran_gaussian (r, sqrt (sigfp*sigfp*z));
		F = sambg + samfl;
	}

	return F;
}

/**
 This function processes the <background> node of the XML problem definition file and reads
 the specified FCS file.
 */
int processBackground (xmlDocPtr doc, gsl_matrix * bg)
{
	// Initialize a new XPath evaluation context
	xmlXPathContextPtr ctxt = xmlXPathNewContext (doc);

	// Initialize a xmlXPathObjectPtr object
	xmlXPathObjectPtr result;

	// Check if the user specified a <background> node
	result = xmlXPathEvalExpression ((xmlChar *) "/problem/data/background/fcsfile", ctxt);

	// If the resulting node set is non-empty (i.e. there is a bg distribution)
	if (result->nodesetval)
	{
		// Extract the path of the background distribution
		xmlChar * path = xmlXPathCastNodeToString (result->nodesetval->nodeTab[0]);

		// Open the FCS file
		FILE * fcs = fopen (path, "rb");
		if (fcs == NULL)
		{
			fprintf (stderr, "error in processDataSet: could not open background FCS file\n");
			return GSL_FAILURE;
		}
		// Read the FCS file header
		float ver;
		long int dos;
		fscanf (fcs, "FCS%f    %*d%*d%ld%*d%*d%*d", &ver, &dos);

		// Initialize a new xmlXPathObjectPtr object
		xmlXPathObjectPtr result2;
		// Apply XPath expression to find the background measurement specification
		result2 = xmlXPathEvalExpression ((xmlChar *) "/problem/data/background/measuredspecies/measurement", ctxt);
		// Find out how many nodes were found (if any)
		size_t size2 = (result2->nodesetval) ? result2->nodesetval->nodeNr : 0;
		// Check that we have a non-empty nodeset
		if (size2 != P)
		{
			fprintf (stderr, "error in processDataSet: size mismatch in background data\n");
			return GSL_FAILURE;
		}
		// Find out how many total events are stored in the FCS file
		int tot = 0;
		if (ver < 3.0)
			tot = fcs2_read_int_kw (fcs, "$TOT");
		else
			tot = fcs3_read_int_kw (fcs, "$TOT");
		// Check that there are enough events
		if (tot < M)
		{
			fprintf (stderr, "error in processDataSet: the background FCS file does not contain enough events\n");
			return GSL_FAILURE;
		}

		// Find out how many parameters are there in each event
		int par = 0;
		if (ver < 3.0)
			par = fcs2_read_int_kw (fcs, "$PAR");
		else
			par = fcs3_read_int_kw (fcs, "$PAR");

		// Initialize read buffer
		float fbuf[par];

		// Iterate through the resulting node set and read the events from the FCS file (keeping only the ones > 1.0)
		for (size_t k = 0; k < size2 ; k++)
		{
			// Get first element child of current node -- this is the node <index>
			xmlNode * cur2 = xmlFirstElementChild (result2->nodesetval->nodeTab[k]);
			// Extract the index of current measured species
			size_t l = (size_t) xmlXPathCastNodeToNumber (cur2);

			// Advance to the next element sibling of the current node -- this is <description> -- currently unused
			cur2 = xmlNextElementSibling (cur2);

			// Advance to the next element sibling of the current node -- this is <fcsparam>
			cur2 = xmlNextElementSibling (cur2);
			// Extract the parameter of current measured species
			size_t p = (size_t) xmlXPathCastNodeToNumber (cur2);

			// Position the file cursor at the beginning of the data segment
			fseek (fcs, dos, SEEK_SET);

			// Read M events
			for (size_t m = 0; m < M; m++)
			{
				// Keep reading until the desired parameter is > 1.1
				do
				{
					fread_floats_swap (fbuf, par, fcs);
				}
				while (fbuf[p-1]<=1.1);

				// Place the data in the proper spot
				gsl_matrix_set (bg, m, l-1, fbuf[p-1]);
			}
		}

		// Close the FCS file
		fclose (fcs);
	}
	else
	{
		gsl_matrix_set_zero (bg);
	}

	// Return
	return GSL_SUCCESS;
}
