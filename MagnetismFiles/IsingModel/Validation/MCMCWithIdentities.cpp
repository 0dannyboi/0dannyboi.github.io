#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <random>
#include <string>
using namespace std;


const double J = 1.0; //coupling
const int N = 3; // spins per side of square
const int spinCount = N * N; // total spins
/* 
   Total number of microstates
   w.r.t. symmetry condition.

*/
constexpr int nStates = 1 << (spinCount - 1);
vector<int> state(spinCount);
int NSweeps = 300000;
/*
   Wait to start measurements in order to
   minimize influence of nonequilibirum dynamics.
*/
int startCollect = 10000; 
int NAvg = 100;
vector<double> betaValues(100);


// initialize random distribution
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dis(0.0, 1.0);


// assign a random value to the global state variable
void initState(){
	for (int i = 0; i < spinCount; i++){
		state[i] = 2 * (rand() % 2) - 1;
	}
}

/* 
    Energy only computed using whole configuration
    when state is initialized. 
*/
float getEnergy(){
	double myEnergy = 0.0;
	int spin2;
	int spin1;
	/*
       	   To prevent double counting on a rectangular
       	   lattice, the neighbors to the right of
           and/or below a site are accounted for
           at each site.
    	*/
	for (int i = 0; i < spinCount; i++){
		spin1 = state[i];
		int col = i % N;
		int row = (i - col) / N;
		if (col < N - 1) {
			spin2 = state[i + 1];
			myEnergy += spin1 * spin2;
		}
		if (row < N - 1) {
			int iBelow = N * row + N + col;
			spin2 = state[iBelow];
			myEnergy += spin1 * spin2; 
		}
	}
	return -1.0 * myEnergy * J;
}

// Metropolis-Hastings MCMC Algorithm
void update (double& beta, double& M,double& e){
	double deltaE;
	int spin = rand() % spinCount; // index of spin to try flipping
	int neighborSpins = 0;
	int currentSpin = state[spin];
	int col = spin % N;
	int row = (spin - col) / N;
	double BoltzmannFactor;
	if (col < N - 1){
		neighborSpins += state[spin + 1];
	}
	if (col > 0){
		neighborSpins += state[spin - 1];
	}
	if (row > 0) {
		neighborSpins += state[N * row - N + col];
	}
	if (row < N - 1) {
		neighborSpins += state[N * row + N + col];
	}
	deltaE = 2.0 * J * neighborSpins * currentSpin;
	// Accept 100% of moves that lower the energy.
	if (deltaE < 0){
		e += deltaE / spinCount;
		M -= (2.0 * currentSpin / static_cast<double>(spinCount));
		state[spin] *= -1;
	} else {
		/*
		    Accept configuration if random number is less
		    than acceptance ratio.
		*/
		double randomValue = dis(gen);
		BoltzmannFactor = (-beta * deltaE);
		if (BoltzmannFactor > log(randomValue)){
			e += deltaE / spinCount;
			M -= (2.0 * currentSpin / static_cast<double>(spinCount));
			state[spin] *= -1;
		}
	}
}


/*
    Given a state, retrieve a base-10 representation
    of the binary identity of the state, where spins
    {-1, 1} are mapped to digits {0, 1}. From symmetry,
    I am considering states with identities, i where
    0 <= i < 2 ^ (spinCount - 1). For N odd, these
    states necessarily have a negative value of M.
    Any state with M > 0, can be mapped to the
    the state where each spin is flipped by
    considering its one's complement.
*/
int getID(double& m){
	int identity = 0;
	int multiplier = 1;
	if (m > 0) {
		multiplier = -1;
	}
	for (int power = 0; power < spinCount; power++){
		int index = spinCount - 1 - power;
		int factor = (1 + multiplier * state[index]) / 2;
		identity += static_cast<int>(pow(2, power) * factor);
	}
	return identity;
}


int main() {
	double myStep = 0.1;
	for (int i = 0; i < 100; i++) {
		betaValues[i] = i * myStep;
	}
	ofstream identityFile("Ising3x3MCMCIdentitiesTWO.csv");
	ofstream expectationFile("Ising3x3MCMCExpectationsTWO.csv");
	if (expectationFile.is_open() && identityFile.is_open()) {
		expectationFile << "beta,E,M" << "\n";
		identityFile << "beta,";
		for (int id = 0; id < nStates - 1; id++){
			identityFile << id << ",";
		}
		identityFile << nStates - 1 << "\n";
		for (int bInd = 0; bInd < 100; bInd ++){
			double beta = betaValues[bInd];
			double AvgM = 0;
			double AvgE = 0;
			cout << "Running Beta = " << beta << "\n";
			vector<int> stateSpace(nStates);
			for (int trial = 0; trial < NAvg; trial++){
				initState();
				double e = getEnergy() / spinCount;
				double M = static_cast<double>(accumulate(state.begin(), 
							state.end(), 0)) / spinCount;
				for (int sweep = 0; sweep < NSweeps; sweep++){
					for (int counter = 0; counter < spinCount; counter++){
						update(beta, M, e);
					}
					if (sweep > startCollect){
						int myID = getID(M);
						stateSpace[myID] += 1;
						AvgM += abs(M) / (NAvg * (NSweeps - startCollect));
						AvgE += e / (NAvg * (NSweeps - startCollect));
					}
				}
			}
		expectationFile << beta << ", ";
                expectationFile << AvgE << ", ";
                expectationFile << AvgM << "\n";
		for (int id = 0; id < nStates - 1; id++){
			double p = static_cast<double>(stateSpace[id]) / (NAvg * (NSweeps - startCollect));
			identityFile << p << ", ";
		}
		double p = static_cast<double>(stateSpace[nStates - 1]) / (NAvg * (NSweeps - startCollect));
		identityFile << p << "\n";
		}
	}	
	return 0;
}
