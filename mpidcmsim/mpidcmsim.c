/*
 *  mpidcmsim.c
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

#include <stochmod.h>
#include <parestlib.h>

// Tags for message passing
#define WORKTAG 1
#define DIETAG 2

// Functions for parallel simulation
static void master (int argc, char * argv[], stochmod * model, gsl_vector * times, gsl_matrix * params);
static void slave (stochmod * model, gsl_matrix * params, gsl_vector * times);


int main (int argc, char * argv[])
{
	// ================================== USER-PROVIDED INFORMATION ==========================================
	// Choose the stochasitc model to simulate
	stochmod * model = malloc (sizeof (stochmod));
	stochrep_mod_setup (model);

	// Set up times vector
	size_t ntimes = 13;
	double timesArr[] = {0*60, 10*60, 20*60, 30*60, 40*60, 50*60, 60*60, 70*60, 80*60, 90*60, 100*60, 110*60, 120*60};
	gsl_vector_view times = gsl_vector_view_array (timesArr, ntimes);


	// ======================================= INITIALIZATION ================================================
	// Initialize MPI
	MPI_Init (&argc, &argv);

	// Find out my identity in the default communicator
	int myrank;
	MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

	// Check correctness of command line arguments
	if (argc != 4)
	{
		if (myrank == 0)
			printf ("Usage: %s parfile datadir nsim/cand\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	// Put command line arguments into correct variables
	char * parpath = argv[1];
	char * datadir = argv[2];
	size_t nsim = atoi (argv[3]);

	// Inform the user that we are ready to start
	if (myrank == 0)
	{
		printf ("\nThis is mpiDcmSim v1.0\nI am the master process\n\n");
		printf ("Parameter file:\n%s\n", parpath);
		printf ("Data directory:\n%s\n", datadir);
		printf ("Simulating %s for %d simulations per candidate\n", model->name, (int) nsim);
	}


	// ============================ SET UP COMMON OBJECTS FOR ALL PROCESSES ==================================
	// Read the parameter matrix from file
	FILE * parfile = fopen (parpath, "r");
	gsl_matrix * params = gsl_matrix_alloc (64, 48);

	if (parfile == NULL)
	{
		printf (">> error opening parameters file .. program exiting\n");
		return EXIT_FAILURE;
	}
	if (gsl_matrix_fscanf (parfile, params)) {
		printf (">> error reading parameters data.. program exiting\n");
		return EXIT_FAILURE;
	}
	fclose (parfile);

	// Inform the user about the # of candidates
	if (myrank == 0)
	{
		printf ("Found %d candidates in the parameter file.\n\n", (int) params->size1);
	}


	// ============================ RUN THE PARALLEL COMPUTATION ==================================
	// Start a new clock to keep track of elapsed time
	clock_t tic = clock ();

	// Decide what I have to do
	if (myrank == 0)
	{
		// I am in charge
		master (argc, argv, model, &times.vector, params);
	}
	else
	{
		// I have to do what I am told
		slave (model, params, &times.vector);
	}

	// Check how much time has passed
	clock_t toc = clock ();
	if (myrank == 0)
		printf ("\n\nSimulations completed.\nElapsed time: %g sec.\n\n", (((double) (toc - tic)) / CLOCKS_PER_SEC));

	// Shut down MPI
	MPI_Finalize ();

	// Clean up
	free (model);
	gsl_matrix_free (params);

	return EXIT_SUCCESS;
}


static void master (int argc, char * argv[], stochmod * model, gsl_vector * times, gsl_matrix * params)
{
	int ntasks, rank, count = 0;
	int work[2];
	MPI_Status status;
	size_t res_no = (model->nspecies)*(times->size);
	double results[res_no];
	size_t nsim = atoi (argv[3]);

	// Allocate rxn_ensemble object to store the results
	rxn_ensemble * res = rxn_ensemble_alloc (nsim, model->nspecies, times->size);

	// Find out how many processes there are in the default communicator
	MPI_Comm_size (MPI_COMM_WORLD, &ntasks);

	// Allocate output matrix
	gsl_matrix * out = gsl_matrix_calloc (model->nout, model->nspecies);
	gsl_matrix_set (out, 0, 6, 1);
	gsl_matrix_set (out, 1, 13, 1);
	gsl_matrix_set (out, 2, 20, 1);

	// Allocate counts matrix
	gsl_matrix * counts = gsl_matrix_calloc (res->nreplic, model->nout*res->ntimes);

	// Loop for the candidates
	size_t j;
	for (j = 0; j < params->size1; j++)
	{
		// Prepare message for the slaves
		work[0] = j;

		// Do the simulations in parallel
		// Seed the slaves; send one unit of work to each slave.
		for (rank = 1; rank < ntasks; rank++)
		{
			work[1] = count;

			MPI_Send (work,	 				// message buffer
					  2,  					// # of data items
					  MPI_INT,        		// data items are of type int
					  rank,					// destination process rank
				      WORKTAG,				// user chosen message tag
					  MPI_COMM_WORLD);		// default communicator

			count++;
		}

		// Loop over getting new work requests until there is no more work to be done
		while (count < nsim)
		{
			// Receive results from a slave
			MPI_Recv (results,				// message buffer
				      res_no,				// # of data items
				      MPI_DOUBLE,			// of type double real
				      MPI_ANY_SOURCE,		// receive from any sender
				      MPI_ANY_TAG,			// any type of message
				      MPI_COMM_WORLD,		// default communicator
				      &status);				// info about the received message

			// Process the results
			size_t i;
			for (i = 0; i < res_no; i++)
			{
				res->data[status.MPI_TAG]->counts->data[i] = results[i];
			}

			// Send the slave a new work unit
			work[1] = count;

			MPI_Send (work,					// message buffer
					  2,					// # of data items
					  MPI_INT,				// data items are of type int
					  status.MPI_SOURCE,	// destination process rank (the slave we just received from)
					  WORKTAG,				// user chosen message tag
					  MPI_COMM_WORLD);		// default communicator

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

		// Extract the output
		rxn_ensemble_counts (res, counts, out);

		// Write to file
		char filepath[80];
		sprintf (filepath, "%s/SimData_%d_%d.txt", argv[2], (int) res->nreplic, (int) j+1);

		FILE * datafile = fopen (filepath, "w");
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

		printf("Written data to file:\n%s\n\n", filepath);

		// Reset the counter
		count = 0;
	}

	// Tell all the slaves to exit by sending an empty message with the DIETAG
	for (rank = 1; rank < ntasks; rank++)
	{
		MPI_Send (0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
	}

	// Free manually allocated resources
	rxn_ensemble_free (res);
	gsl_matrix_free (out);
	gsl_matrix_free (counts);

	return;
}


static void slave (stochmod * model, gsl_matrix * params, gsl_vector * times)
{
	int myrank;
	int work[2];
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
		MPI_Recv (work, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// Check the tag of the received message
		if (status.MPI_TAG == DIETAG)
		{
			break;
		}

		// Sample a new initial state
		gsl_vector_set (X0, 0, gsl_rng_uniform_int (r, 3));
		gsl_vector_set (X0, 1, 2 - gsl_vector_get (X0, 0));
		gsl_vector_set (X0, 2, 0);
		gsl_vector_set (X0, 3, 0);
		gsl_vector_set (X0, 4, 0);
		gsl_vector_set (X0, 5, gsl_rng_uniform_int (r, 13));
		gsl_vector_set (X0, 6, 4*gsl_rng_uniform_int (r, 11));

		gsl_vector_set (X0, 7, gsl_rng_uniform_int (r, 3));
		gsl_vector_set (X0, 8, 2 - gsl_vector_get (X0, 7));
		gsl_vector_set (X0, 9, 0);
		gsl_vector_set (X0, 10, 0);
		gsl_vector_set (X0, 11, 0);
		gsl_vector_set (X0, 12, gsl_rng_uniform_int (r, 13));
		gsl_vector_set (X0, 13, 1*gsl_rng_uniform_int (r, 11));

		gsl_vector_set (X0, 14, gsl_rng_uniform_int (r, 3));
		gsl_vector_set (X0, 15, 2 - gsl_vector_get (X0, 14));
		gsl_vector_set (X0, 16, 0);
		gsl_vector_set (X0, 17, 0);
		gsl_vector_set (X0, 18, 0);
		gsl_vector_set (X0, 19, gsl_rng_uniform_int (r, 13));
		gsl_vector_set (X0, 20, 12*gsl_rng_uniform_int (r, 11));

		// Extract appropriate candidate parameters
		gsl_vector_view pars = gsl_matrix_row (params, work[0]);

		// Do the work
		ssa_direct_trajectory (model, &pars.vector, times, X0, rsp, r); ///RESTART FROM HERE need to pass 2 things!!

		// Communicate that the simulation has been done
		printf ("This is slave process %d, completed simulation %d of candidate %d.\n", myrank, work[1]+1, work[0]+1);

		// Send the result back
		MPI_Send (rsp->counts->data, res_no, MPI_DOUBLE, 0, work[1], MPI_COMM_WORLD);
	}

	// Free manually allocated resources
	gsl_rng_free (r);
	gsl_vector_free (X0);
	rxn_sample_path_free (rsp);

	return;
}
