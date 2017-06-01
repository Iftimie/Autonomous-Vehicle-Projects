#include <sys/types.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
using namespace std;

/* bzero is not always available (e.g., in Win32) */
#ifndef bzero
#define bzero(b,len) (memset((b), '\0', (len)), (void) 0)
#endif

/* are we maximizing or minimizing? */
#define HUNGARIAN_MIN 0
#define HUNGARIAN_MAX 1

/*
* the two main internal routines return this type, telling us what to do next
*/
typedef enum
{
	HUNGARIAN_ERROR,
	HUNGARIAN_ROUTINE_ONE,
	HUNGARIAN_ROUTINE_TWO,
	HUNGARIAN_DONE
} hungarian_code_t;

// the Q matrix is filled with instances of this type
typedef enum
{
	HUNGARIAN_ZERO,
	HUNGARIAN_ONE,
	HUNGARIAN_STAR
} hungarian_q_t;

/*
* a simple linked list
*/
class HungarianSequenceT{
public:
	int* i;
	int* j;
	int k;
};

/*
* we'll use objects of this type to keep track of the state of the thislem
* and its solution
*/
class HungarianT{
public:
	int dummy_columns;
	size_t m, n;  // thislem dimensions
	int** r;     // the rating (utility) matrix
	int** q;     // the Q matrix
	int* u;      // the U vector
	int* v;      // the V vector
	int* ess_rows; // list of essential rows
	int* ess_cols; // list of essential columns
	HungarianSequenceT seq; // sequence of i's and j's
	int row_total, col_total; // row and column totals
	int* a;  // assignment vector
	int maxutil;  // maximum utility
	int mode; // are we maximizing or minimizing?

	/*
	* initialize the given object as an mXn thislem.  allocates storage, which
	* should be freed with hungarian_fini().
	*/
	HungarianT(int* r, size_t m, size_t n, int dummy_cols,int mode){
		this->dummy_columns = dummy_cols;
		int i, j;
		// did we get a good object pointer?
		assert(this);

		// we can't work with matrices that have more rows than columns.
		//
		// TODO: automatically transpose such matrices
		assert(m <= n);

		// init the struct
		this->m = m;
		this->n = n;
		assert(this->r = (int**)calloc(m, sizeof(int*)));
		assert(this->q = (int**)calloc(m, sizeof(int*)));
		this->mode = mode;
		this->maxutil = 0;
		for (i = 0; i<m; i++)
		{
			assert(this->r[i] = (int*)calloc(n, sizeof(int)));
			assert(this->q[i] = (int*)calloc(n, sizeof(int)));
			for (j = 0; j<n; j++)
			{
				this->r[i][j] = r[i*n + j];
				if (this->r[i][j] > this->maxutil)
					this->maxutil = this->r[i][j];
			}
		}
		// if we're going to minimize, rather than maximize, we need to subtract
		// each utility from the maximum.  this operation will be reversed before
		// computing the benefit.
		if (mode == HUNGARIAN_MIN)
		{
			for (i = 0; i<m; i++)
			{
				for (j = 0; j<n; j++)
					this->r[i][j] = this->maxutil - this->r[i][j];
			}
		}

		assert(this->a = (int*)calloc(m, sizeof(int)));
		assert(this->u = (int*)calloc(m, sizeof(int)));
		assert(this->v = (int*)calloc(n, sizeof(int)));
		assert(this->seq.i = (int*)calloc(m*n, sizeof(int)));
		assert(this->seq.j = (int*)calloc(m*n, sizeof(int)));
		assert(this->ess_rows = (int*)calloc(m, sizeof(int)));
		assert(this->ess_cols = (int*)calloc(n, sizeof(int)));
	}

	/*
	* frees storage associated with the given thislem object.  you must have
	* called hungarian_init() first.
	*/
	~HungarianT(){
		int i;
		assert(this);
		for (i = 0; i<this->m; i++)
		{
			free(this->q[i]);
			free(this->r[i]);
		}
		free(this->q);
		free(this->r);
		free(this->a);
		free(this->u);
		free(this->v);
		free(this->seq.i);
		free(this->seq.j);
		free(this->ess_rows);
		free(this->ess_cols);
	}

	/*
	* make an initial cover
	*/
	void hungarian_make_cover(){
		int i, j;
		int* row_max;
		int* col_max;

		assert(this);
		assert(row_max = (int*)calloc(this->m, sizeof(int)));
		assert(col_max = (int*)calloc(this->n, sizeof(int)));

		this->row_total = this->col_total = 0;

		// find the max in each row (col) and sum them
		for (i = 0; i<this->m; i++)
		{
			for (j = 0; j<this->n; j++)
			{
				if (this->r[i][j] > row_max[i])
					row_max[i] = this->r[i][j];
			}
			this->row_total += row_max[i];
		}

		for (j = 0; j<this->n; j++)
		{
			for (i = 0; i<this->m; i++)
			{
				if (this->r[i][j] > col_max[j])
					col_max[j] = this->r[i][j];
			}
			this->col_total += col_max[j];
		}

		// make u and v into an initial cover, based on row and col totals
		if (this->row_total <= this->col_total)
			memcpy(this->u, row_max, sizeof(int)*this->m);
		else
			memcpy(this->v, col_max, sizeof(int)*this->n);

		free(row_max);
		free(col_max);
	}

	void hungarian_build_q(){
		int i, j;
		assert(this);

		for (i = 0; i<this->m; i++)
		{
			for (j = 0; j<this->n; j++)
			{
				if ((this->u[i] + this->v[j]) == this->r[i][j])
				{
					if (this->q[i][j] == HUNGARIAN_ZERO)
						this->q[i][j] = HUNGARIAN_ONE;
				}
				else
					this->q[i][j] = HUNGARIAN_ZERO;
			}
		}
	}

	void hungarian_add_stars(){
		int i, j, k;

		assert(this);

		if (this->row_total <= this->col_total)
		{
			for (i = 0; i<this->m; i++)
			{
				for (j = 0; j<this->n; j++)
				{
					if (this->q[i][j] == HUNGARIAN_ONE)
					{
						for (k = 0; k<this->m; k++)
						{
							if ((k != i) && (this->q[k][j] == HUNGARIAN_STAR))
								break;
						}
						if (k == this->m)
						{
							this->q[i][j] = HUNGARIAN_STAR;
							break;
						}
					}
				}
			}
		}
		else
		{
			for (j = 0; j<this->n; j++)
			{
				for (i = 0; i<this->m; i++)
				{
					if (this->q[i][j] == HUNGARIAN_ONE)
					{
						for (k = 0; k<this->n; k++)
						{
							if ((k != j) && (this->q[i][k] == HUNGARIAN_STAR))
								break;
						}
						if (k == this->n)
						{
							this->q[i][j] = HUNGARIAN_STAR;
							break;
						}
					}
				}
			}
		}
	}

	/*
	* Kuhn's Routine I
	*/
	hungarian_code_t hungarian_routine_one()
	{
		int i, j, k;
		char jumpcase1, jumpprelim;

		assert(this);
		this->seq.k = 0;
		for (i = 0; i<this->m; i++)
			this->ess_rows[i] = 0;

		j = 0;
		// look for a 1* in each column
		while (j<this->n)
		{
			i = 0;
			while (i<this->m)
			{
				if (this->q[i][j] == HUNGARIAN_STAR)
				{
					// found a 1*; next column
					break;
				}
				i++;
			}
			if (i<this->m)
			{
				// found a 1*; next column
				j++;
			}
			else
			{
				// didn't find a 1*; column j is eligible; search it for a 1
				i = 0;
				while (i<this->m)
				{
					if (this->q[i][j] == HUNGARIAN_ONE)
					{
						// found a 1 (i,j); start recording
						this->seq.i[this->seq.k] = i;
						this->seq.j[this->seq.k] = j;
						this->seq.k++;
						// CASE 1
						jumpprelim = 0;
						while (!jumpprelim)
						{
							// look in row ik for a 1*
							j = 0;
							jumpcase1 = 0;
							while (j<this->n && !jumpcase1)
							{
								if (this->q[i][j] == HUNGARIAN_STAR)
								{
									// CASE 2
									i = 0;
									while (!jumpcase1)
									{
										// found a 1* in (ik,jk); search col jk for a 1
										while (i<this->m)
										{
											if (this->q[i][j] == HUNGARIAN_ONE)
											{
												// test i for distinctness
												for (k = 0; k<this->seq.k; k++)
												{
													if (this->seq.i[k] == i)
														break;
												}
												if (k<this->seq.k)
												{
													// i is not distinct
													i++;
													continue;
												}
												else
												{
													// i is distinct; record and back to Case 1
													this->seq.i[this->seq.k] = this->seq.i[this->seq.k - 1];
													this->seq.j[this->seq.k] = j;
													this->seq.k++;
													this->seq.i[this->seq.k] = i;
													this->seq.j[this->seq.k] = j;
													this->seq.k++;

													jumpcase1 = 1;
													break;
												}
											}
											else
												i++;
										}
										if (i >= this->m)
										{
											// didn't find a 1 in col jk; row ik is essential
											this->ess_rows[this->seq.i[this->seq.k - 1]] = 1;

											// delete last two elts of sequence
											j = this->seq.j[this->seq.k - 1];
											i = this->seq.i[this->seq.k - 1] + 1;

											if (this->seq.k > 1)
											{
												// back to Case 2
												this->seq.k -= 2;
												continue;
											}
											else
											{
												// k==1; back to prelim search for 1 in (i1+1,j0)
												this->seq.k--;
												jumpcase1 = jumpprelim = 1;
												break;
											}
										}
									}
								}
								else
									j++;
							}
							if (j >= this->n)
							{
								// didn't find a 1* in row ik; toggle and Alternative Ia
								for (; this->seq.k > 0; this->seq.k--)
								{
									if (this->q[this->seq.i[this->seq.k - 1]][this->seq.j[this->seq.k - 1]]
										== HUNGARIAN_ONE)
									{
										this->q[this->seq.i[this->seq.k - 1]][this->seq.j[this->seq.k - 1]]
											= HUNGARIAN_STAR;
									}
									else
									{
										this->q[this->seq.i[this->seq.k - 1]][this->seq.j[this->seq.k - 1]]
											= HUNGARIAN_ONE;
									}
								}
								// Alternative Ia
								return(HUNGARIAN_ROUTINE_ONE);
							}
						}
					}
					else
						i++;
				}
				// didn't find a 1 in col j; next col
				j++;
			}
		}
		// out of cols; Alternative Ib
		// determine ess cols
		for (j = 0; j<this->n; j++)
		{
			this->ess_cols[j] = 0;
			for (i = 0; i<this->m; i++)
			{
				if (this->q[i][j] == HUNGARIAN_STAR && !this->ess_rows[i])
				{
					this->ess_cols[j] = 1;
					break;
				}
			}
		}
		return(HUNGARIAN_ROUTINE_TWO);
	}

	/*
	* Kuhn's Routine II
	*/
	hungarian_code_t hungarian_routine_two(){
		int i, j, d, m, dtmp;
		int oldsum, newsum;

		assert(this);

		oldsum = 0;
		for (i = 0; i<this->m; i++)
			oldsum += this->u[i];
		for (j = 0; j<this->n; j++)
			oldsum += this->v[j];

		d = 0;
		for (i = 0; i<this->m; i++)
		{
			if (this->ess_rows[i])
				continue;
			for (j = 0; j<this->n; j++)
			{
				if (this->ess_cols[j])
					continue;
				dtmp = this->u[i] + this->v[j] - this->r[i][j];
				if (dtmp<0)
				{
					printf("SUPERMOO: %d + %d < %d\n",
						this->u[i], this->v[j], this->r[i][j]);
				}
				if (!d || ((dtmp>0) && dtmp < d))
					d = dtmp;
			}
		}

		if (d<0)
			printf("MOO: %d < 0\n", d);

		//if(d<=0)
		if (!d)
			return(HUNGARIAN_DONE);
		else
		{
			// is there some u[i]==0?
			for (i = 0; i<this->m; i++)
			{
				if (!this->u[i])
					break;
			}
			if (i < this->m)
			{
				// CASE 2; some u[i] == 0; compute m as the min of d and inessential v[j]
				m = d;
				for (j = 0; j<this->n; j++)
				{
					if ((!this->ess_cols[j]) && (this->v[j] < m))
						m = this->v[j];
				}

				// adjust the cover
				for (i = 0; i<this->m; i++)
				{
					if (this->ess_rows[i])
						this->u[i] += m;
				}
				for (j = 0; j<this->n; j++)
				{
					if (!this->ess_cols[j])
						this->v[j] -= m;
				}
			}
			else
			{
				// CASE 1; all u[i] > 0; compute m as the min of d and inessential u[j]
				m = d;
				for (i = 0; i<this->m; i++)
				{
					if ((!this->ess_rows[i]) && (this->u[i] < m))
						m = this->u[i];
				}

				// adjust the cover
				for (i = 0; i<this->m; i++)
				{
					if (!this->ess_rows[i])
						this->u[i] -= m;
				}
				for (j = 0; j<this->n; j++)
				{
					if (this->ess_cols[j])
						this->v[j] += m;
				}
			}
		}

		newsum = 0;

		for (i = 0; i<this->m; i++)
		{
			for (j = 0; j<this->n; j++)
			{
				if (this->u[i] + this->v[j]<this->r[i][j])
				{
					printf("SUPERMOO (%d,%d): %d + %d < %d\n", i, j,
						this->u[i], this->v[j], this->r[i][j]);
				}
			}
		}

		// Alternative IIa; build new q and return to routine I
		hungarian_build_q();
		return(HUNGARIAN_ROUTINE_ONE);
	}

	/*
	* solve the given thislem.  runs the Hungarian Method on the rating matrix
	* to produce optimal assignment, which is stored in the vector this->a.
	* you must have called hungarian_init() first.
	*/
	void hungarian_solve(){
		hungarian_code_t state = HUNGARIAN_ROUTINE_ONE;
		int i, j;

		assert(this);

		bzero(this->u, sizeof(int)*this->m);
		bzero(this->v, sizeof(int)*this->n);
		this->seq.k = 0;
		bzero(this->ess_rows, sizeof(int)*this->m);
		bzero(this->ess_cols, sizeof(int)*this->n);
		for (i = 0; i<this->m; i++)
			bzero(this->q[i], sizeof(int)*this->n);

		hungarian_make_cover();
		hungarian_build_q();
		hungarian_add_stars();

		while (state != HUNGARIAN_DONE)
		{
			if (state == HUNGARIAN_ROUTINE_ONE)
			{
				state = hungarian_routine_one();
			}
			else
			{
				state = hungarian_routine_two();
			}
		}

		// fill in the assignment vector
		for (i = 0; i<this->m; i++)
		{
			for (j = 0; j<this->n; j++)
			{
				if (this->q[i][j] == HUNGARIAN_STAR)
				{
					this->a[i] = j;
					break;
				}
			}
		}
	}

	/*
	* prints out the current state of the Q matrix.  also computes and prints
	* out the benefit from the current assignment.  mostly useful for
	* debugging.
	*/
	void hungarian_print_stars()
	{
		int i, j;
		int benefit = 0;
		assert(this);
		puts("\nQ: ");
		for (i = 0; i<this->m; i++)
		{
			printf("  [ ");
			for (j = 0; j<this->n; j++)
			{
				if (this->q[i][j] == HUNGARIAN_ZERO)
					printf("%4d ", 0);
				else if (this->q[i][j] == HUNGARIAN_ONE)
					printf("%4d ", 1);
				else
				{
					printf("%3d%s ", 1, "*");
					if (this->mode == HUNGARIAN_MIN)
						benefit += this->maxutil - this->r[i][j];
					else
						benefit += this->r[i][j];
				}
			}
			puts(" ]");
		}
		printf("\nBenefit: %d\n\n", benefit);
	}

	/*
	* prints out the resultant assignment in a 0-1 matrix form.  also computes
	* and prints out the benefit from the assignment.  you must have called
	* hungarian_solve() first.
	*/
	void hungarian_print_assignment(vector<pair<int, int>> &assignements, vector<int> &trackingsWithoutDetections, vector<int> &ignoredDetections){

		int **mat = (int**)malloc(sizeof(int*)*this->m);
		for (int i = 0; i < this->m; i++)
			mat[i] = (int*)malloc(sizeof(int)*this->n);


		int i, j;
		assert(this);
        //puts("\nA:");
		for (i = 0; i<this->m; i++)
		{
            //printf("  [ ");
			for (j = 0; j<this->n; j++)
			{
                //printf("%4d ", (j == this->a[i]) ? 1 : 0);
				mat[i][j] = (j == this->a[i]) ? 1 : 0;
				/*
				if(j == this->a[i])
				printf("1 ");
				else
				printf("0 ");
				*/
			}
            //printf(" ]\n");
		}

		for (int i = 0; i < this->m; i++){
			for (int j = 0; j < this->n; j++){
				//printf("%d ", mat[i][j]);
			}
			//printf("\n");
		}

		//printf("Assignements:\n");
		for (int i = 0; i < this->m; i++){
			for (int j = 0; j < (this->n - this->dummy_columns); j++){
				if (mat[i][j] == 1){
					//printf("Assignement for %d = %d\n", i, j);
					assignements.push_back(pair<int,int>(i, j));
				}

			}
		}

		//printf("Trackings without detections:\n");
		for (int i = 0; i < this->m; i++){
			int zeros = 0;
			for (int j = 0; j < (this->n - this->dummy_columns); j++){
				if (mat[i][j] == 0)
					zeros++;
			}
			if (zeros == this->n - this->dummy_columns){
				//printf("%d did not get a detection\n", i);
				trackingsWithoutDetections.push_back(i);
			}
		}

		//printf("Ignored detections:\n");
		for (int j = 0; j < this->n; j++){
			int zeros = 0;
			for (int i = 0; i< (this->m); i++){
				if (mat[i][j] == 0)
					zeros++;
			}
			if (zeros == this->m){
				//printf("%d detection is ignored\n", j);
				ignoredDetections.push_back(j);
			}
		}

		for (int i = 0; i < this->m; i++)
			free(mat[i]);
		free(mat);

	}

	/*
	* prints out the rating matrix for the given thislem.  you must have called
	* hungarian_solve() first.
	*/
	void hungarian_print_rating(){
		int i, j;

		puts("\nR: ");
		for (i = 0; i<this->m; i++)
		{
			printf("  [ ");
			for (j = 0; j<this->n; j++)
			{
				printf("%4d ", this->r[i][j]);
			}
			puts(" ]");
		}
	}

	/*
	* check whether an assigment is feasible.  returns 1 if the assigment is
	* feasible, 0 otherwise.  you must have called hungarian_solve() first.
	*/
	int hungarian_check_feasibility()
	{
		int i, j;
		char assigned;

		// check for over/under assigned rows
		for (i = 0; i<this->m; i++)
		{
			assigned = 0;
			for (j = 0; j<this->n; j++)
			{
				if (this->q[i][j] == HUNGARIAN_STAR)
				{
					if (assigned)
						return(0);
					else
						assigned = 1;
				}
			}
			if ((this->m <= this->n) && !assigned)
				return(0);
		}
		// check for over/under assigned cols
		for (j = 0; j<this->n; j++)
		{
			assigned = 0;
			for (i = 0; i<this->m; i++)
			{
				if (this->q[i][j] == HUNGARIAN_STAR)
				{
					if (assigned)
						return(0);
					else
						assigned = 1;
				}
			}
			if ((this->n <= this->m) && !assigned)
				return(0);
		}
		return(1);
	}

	/*
	* computes and returns the benefit from the assignment.  you must have
	* called hungarian_solve() first.
	*/
	int hungarian_benefit()
	{
		int i;
		int benefit = 0;
		assert(this);
		for (i = 0; i<this->m; i++)
		{
			if (this->mode == HUNGARIAN_MIN)
				benefit += this->maxutil - this->r[i][this->a[i]];
			else
				benefit += this->r[i][this->a[i]];
		}

		return(benefit);
	}

};
