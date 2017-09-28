/*
 *  mpiStochSim.c
 *  mpiStochSim
 *
 *  Parallel stochatic simulator
 *
 *  Created by Gabriele Lillacci in December 2011.
 *	Latest revision: April 2013.
 *
 *
 *	This free software is available under the Creative Commons Attribution Share Alike License.
 *	You are permitted to use, redistribute and adapt this software as long as appropriate credit
 *	is given to the original author, and all derivative works are distributed under the same
 *	license or a compatible one.
 *	For more information, visit http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to
 *	Creative Commons, 171 2nd Street, Suite 300, San Francisco, California, 94105, USA.
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
#define DIETAG 2

// Global variable declarations
stochmod * model = NULL;
gsl_vector * times = NULL;
gsl_matrix * data = NULL;
gsl_vector * params = NULL;
gsl_matrix * out = NULL;
gsl_matrix * bg = NULL;
size_t M;					// Number of experimental samples to use for identification
size_t P;					// Number of measured species
size_t K;					// Number of time points
size_t R;					// Number of parameters
double mufp;				// Mean fluorescence level
double sigfp;				// Sd of fluorescence level

// Functions for parallel simulation
static void master (int argc, char * argv[]);
static void slave ();

// Problem definition processing
int processBackground (xmlDocPtr doc, gsl_matrix * bg);
double sample_fl (const gsl_rng * r, double z, double mufp, double sigfp, gsl_matrix * bg);


int main (int argc, char * argv[])
{
	// ================================== USER-PROVIDED INFORMATION ==========================================

	// Choose the stochastic model to estimate
	model = (stochmod *) malloc (sizeof (stochmod));
	if (model == NULL)
	{
		fprintf (stderr, "error in memory allocation\n");
		return EXIT_FAILURE;
	}
	//birthdeath_mod_setup (model);
	// lacgfp7_mod_setup (model);
	// lacgfp8_mod_setup (model);
	// lacgfp10_mod_setup (model);
	//iff_mod_setup (model);
	//fbk_mod_setup (model);
	synpi1_mod_setup (model);

	// ==================================== INITIALIZE LIBRARIES =============================================

	// Initialize MPI
	MPI_Init (&argc, &argv);

	// Initialize libxml2
	LIBXML_TEST_VERSION

	// Set up times vector
	size_t ntimes = 7;
	double timesArr[] = {0, 1, 2, 3, 4, 5, 6};
	//double timesArr[] = {0, 0.5, 1, 2, 10};
	//double timesArr[] = {0, 5, 10, 20, 40, 100};
	//double timesArr[] = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100};
	gsl_vector_view timesv = gsl_vector_view_array (timesArr, ntimes);
	times = gsl_vector_alloc (ntimes);
	gsl_vector_memcpy (times, &timesv.vector);

	// Set up output matrix
	out = gsl_matrix_calloc (model->nout, model->nspecies);
	model->output (out);

	// Set up the parameters

	// SYNPI1
	double thetaArr[] = {
			34.298269,
			26.886552,
			0.002897,
			15.683755,
			1.468133,
			3.981358,
			25.305498,
			2.88266,
			1.470988,
			1.851022,
			1.03305,
			0.026716,
			2.886733,
			20,
	};
	mufp = 43.826539;
	sigfp = 18.399774;


	/*
	// LACGFP8 del
	double thetaArr[] = {
			47.310318,
			0.726813,
			7.220694,
			3.507423,
			50.262086,
			1913.532142,
			9.644028,
			1.413203,
			18.529878,
			16.54605,
			0.390285,
			0.842991,
			2.28938,
			0.552476,
			0.362483,
			10,
	};
	mufp = 21.19559;
	sigfp = 21.435175;
	*/

	/*
	// LACGFP8 wt
	double thetaArr[] = {
			15.148251,
			10.140071,
			10.0335,
			0.167728,
			3.566034,
			764.826404,
			2.322734,
			0.71456,
			1.481745,
			1.368398,
			0.178245,
			0.140088,
			3.597409,
			2.6404,
			1.478131,
			10,
	};
	mufp = 7.714692;
	sigfp = 96.97072;
	*/

	/*
	// LACGFP10
	double thetaArr[] = {
			5.081817,
			22.169527,
			3.948393,
			0.655126,
			0.36713,
			10,
	};
	mufp = 23.982056;
	sigfp = 185.522169;
	*/

/*
	// LACGFP7
	double thetaArr[] = {
			1.967605,
			2.488408,
			6.990915,
			7.405571,
			3.409623,
			2944.698612,
			7.35014,
			0.225702,
			0.996885,
			198.32985,
			0.540904,
			134.717476,
			0.240542,
			0.008364,
			4.649781,
			27.54942,
			1.515594,
			3.334262,
			10,
	};
	mufp = 28.260522;
	sigfp = 7.333291;
	*/

	/*
	// LACGFP7-syn
	double thetaArr[] = {
			21.110953,
			19.946339,
			40.708508,
			14.565776,
			21.382297,
			114.806751,
			5.982911,
			1.003165,
			0.542236,
			1054.174046,
			5.286989,
			136.857762,
			0.114027,
			0.005684,
			4.128266,
			19.482701,
			1.372921,
			5.70492,
			10,
	};
	mufp = 28.610798;
	sigfp = 17.597984;
	*/


	/*
	// IFF initial parameters -- ATT! k7 = 0.1 - Output = P
	double thetaArr[] = {
			2.0000,
			0.1000,
			1.0000,
			0.1000,
			1.0000,
			0.0100
	};

	mufp = 100.0;
	sigfp = 20.0;
	*/

/*
	// IFF new parameters -- ATT! k7 = 1 - Output = M
	double thetaArr[] = {
			120,
			0.1,
			1,
			0.1,
			1,
			0.05,
	};

	mufp = 100.0;
	sigfp = 20.0;
*/

/*
	// FBK new parameters -- ATT! k7 = 1 - Output = M
	double thetaArr[] = {
			120,
			0.1,
			1,
			0.1,
			1,
			1,
	};

	mufp = 100.0;
	sigfp = 20.0;
*/

/*
	// IFF estimated
	double thetaArr[] = {
			0.746071,
			6.109691,
			0.439069,
			7.798176,
			3.976363,
			0.634621,

	};

	mufp = 93.874867;
	sigfp = 26.149121;
*/

/*
	// IFF estimated
	double thetaArr[] = {
			1.370015,
			1.044293,
			0.076421,
			0.098566,
			3.077971,
			0.905376,
	};

	mufp = 144.260491;
	sigfp = 48.520402;
*/
/*
	// FBK old parameters -- ATT!! k7 = 0.1, Output = P
	double thetaArr[] = {
			10.0000,
			0.1,
			1.0000,
			0.1000,
			1.0000,
			0.1,
	};

	mufp = 100.0;
	sigfp = 20.0;
*/

/*
	// FBK estimated
	double thetaArr[] = {
			1.611916,
			0.156929,
			0.047154,
			0.041275,
			0.873675,
			9.247406,
	};

	mufp = 185.817405;
	sigfp = 48.074888;
*/

	gsl_vector_view pars = gsl_vector_view_array (thetaArr, model->nparams + model->nin);
	params = gsl_vector_alloc (model->nparams + model->nin);
	gsl_vector_memcpy (params, &pars.vector);


	// ================================= PROCESS COMMAND-LINE ARGUMENTS =========================================

	// Find out my identity in the default communicator
	int myrank;
	MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

	// Check correctness of command line arguments
	if (argc != 4)
	{
		if (myrank == 0)
			printf ("Usage: %s problem-file data-file nsim\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	// Put command line arguments into correct variables
	char * problemfile = argv[1];
	char * datafile = argv[2];
	size_t nsim = atoi (argv[3]);

	// Inform the user that we are ready to start
	if (myrank == 0)
	{
		printf ("\nThis is mpiStochSim v1.0\nI am the master process\n\n");
		printf ("Problem definition file:\n%s\n", problemfile);
		printf ("Data file:\n%s\n", datafile);
		printf ("Simulating %s for %d simulations.\n\n", model->name, (int) nsim);
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
	if (gsl_matrix_isnull (bg))
	{
		gsl_matrix_free (bg);
		bg = NULL;
	}

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

	// Set up the random number generator
	gsl_rng_env_setup ();
	gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
	gsl_rng_set (r, (unsigned long int) time (NULL));

	// Allocate rxn_ensemble object to store the results
	rxn_ensemble * res = rxn_ensemble_alloc (nsim, model->nspecies, times->size);

	// Find out how many processes there are in the default communicator
	MPI_Comm_size (MPI_COMM_WORLD, &ntasks);

	// Seed the slaves; send one unit of work to each slave.
	for (rank = 1; rank < ntasks; rank++)
	{
		MPI_Send (&count,             // message buffer
				  1,                 // # of data items
				  MPI_INT,        // data items are of type double
				  rank,              // destination process rank
				  WORKTAG,           // user chosen message tag
				  MPI_COMM_WORLD);   // default communicator

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
		size_t i;
		for (i = 0; i < res_no; i++)
		{
			res->data[status.MPI_TAG]->counts->data[i] = results[i];
		}

		// Send the slave a new work unit
		MPI_Send (&count,              // message buffer
				  1,                   // # of data items
				  MPI_INT,             // data items are of type int
				  status.MPI_SOURCE,   // destination process rank (the slave we just received from)
				  WORKTAG,             // user chosen message tag
				  MPI_COMM_WORLD);     // default communicator

		// Get the next unit of work to be done
		count++;
	}

	// There's no more work to be done, so receive all the outstanding results from the slaves
	for (rank = 1; rank < ntasks; rank++)
	{
		MPI_Recv (results, res_no, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// Process the results
		size_t i;
		for (i = 0; i < res_no; i++)
		{
			res->data[status.MPI_TAG]->counts->data[i] = results[i];
		}
	}

	// Tell all the slaves to exit by sending an empty message with the DIETAG
	for (rank = 1; rank < ntasks; rank++)
	{
		MPI_Send (0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
	}

	// Allocate counts matrix
	gsl_matrix * counts = gsl_matrix_calloc (res->nreplic, model->nout*res->ntimes);

	// Extract the output
	rxn_ensemble_counts (res, counts, out);

	// Compute the flourescence levels
	for (size_t i = 0; i < counts->size1; i++)
	{
		for (size_t j = 0; j < counts->size2; j++)
		{
			// Current count
			double z = gsl_matrix_get (counts, i, j);

			// Generate a fluorescence level
			gsl_matrix_set (counts, i, j, sample_fl (r, z, mufp, sigfp, bg));
		}
	}

	// Write simulation data to file
	FILE * datafile = fopen (argv[2], "w");
	if (datafile == NULL)
	{
		printf (">> error opening file for writing.. program exiting\n");
		return;
	}
	if (gsl_matrix_fprintf (datafile, counts, "%e")) {
		printf (">> error writing simulation data.. program exiting\n");
		return;
	}
	fclose (datafile);

	printf ("Written data to file:\n%s\n", argv[2]);

	// Free manually allocated resources
	rxn_ensemble_free (res);
	gsl_matrix_free (counts);
	gsl_rng_free (r);

	return;
}


static void slave ()
{
	int myrank;
	int work;
	size_t res_no = (model->nspecies)*(times->size);
	double results[res_no];
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

	while (1)
	{
		// Receive a message from the master
		MPI_Recv (&work, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// Check the tag of the received message
		if (status.MPI_TAG == DIETAG)
		{
			break;
		}

		// Sample a new initial state
		model->initial (X0, r);

		// Do the work
		ssa_direct_trajectory (model, params, times, X0, rsp, r);

		// Communicate that the simulation has been done
		printf ("This is slave process %d, completed simulation %d.\n", myrank, work);

		// Send the result back
		MPI_Send (rsp->counts->data, res_no, MPI_DOUBLE, 0, work, MPI_COMM_WORLD);
	}

	// Free manually allocated resources
	gsl_rng_free (r);
	gsl_vector_free (X0);
	rxn_sample_path_free (rsp);

	return;
}


// Sample a fluorescence level given the background and the count
double sample_fl (const gsl_rng * r, double z, double mufp, double sigfp, gsl_matrix * bg)
{
	double sambg = 0.0, samfl = 0.0;

	// Sample a background value if a bg is present
	if (bg)
	{
		sambg = gsl_matrix_get (bg, gsl_rng_uniform_int (r, M), 0);
	}

	// Sample a fluorescence level if the count is not zero
	if (z > 0)
	{
		samfl = mufp*z + gsl_ran_gaussian (r, sqrt (sigfp*sigfp*z));
	}

	// Return
	return sambg + samfl;
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
	xmlXPathObjectPtr result = NULL;

	// Check if the user specified a <background> node
	result = xmlXPathEval ((xmlChar *) "/problem/data/background/fcsfile", ctxt);

	// If the resulting node set is non-empty (i.e. there is a bg distribution)
	if (result->nodesetval->nodeNr)
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
