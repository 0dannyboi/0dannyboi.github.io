#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <tuple>
#include<bitset>
#include <numeric>
#include <string>

using namespace std;


const double J = 1.0; //coupling


//number of spins per side of square
const int N = 3;


//total number of spins
constexpr int spinCount = N * N;


//total number of unique states (using symmetry)
constexpr int nStates = 1 << (spinCount - 1);


vector < double > betaValues(100);


string binaryString(int & n) {
  return bitset < spinCount > (n).to_string();
}


/* 
   Funtion returns energy and magnetization for
   a state indexed by a base-10 number.
*/
tuple < double, int > getEnergyMag(int & id) {
  double myEnergy = 0.0;
  int myMag = 0;
  string state = binaryString(id);
  int spin2; // neighbor spin
  int spin1; // test spin
  for (int i = 0; i < spinCount; i++) {
    /*
       Convert string to int by subtracting
       ASCII value of char '0' from it.
       Then map values {0, 1} to {-1, 1}.
    */
    spin1 = 2 * (state[i] - '0') - 1;
    myMag += spin1;
    int col = i % N;
    int row = (i - col) / N;
    /*
       To prevent double counting on a rectangular
       lattice, the neighbors to the right of
       and/or below a site are accounted for
       at each site.
    */
    if (col < N - 1) {
      spin2 = 2 * (state[i + 1] - '0') - 1;
      myEnergy += spin1 * spin2;
    }
    if (row < N - 1) {
      int iBelow = N * row + N + col;
      spin2 = 2 * (state[iBelow] - '0') - 1;
      myEnergy += spin1 * spin2;
    }
  }
  tuple < double, int > res = make_tuple(-1.0 * myEnergy * J, myMag);
  return res;
}


/*
    Boltzmann Factor to provide a
    statistical weight for each state
    based on its energy.
*/
double evalBoltzmann(double & energy, const double & beta) {
  return exp(-1 * beta * energy);
};



int main() {
  double myStep = 0.1;
  for (int i = 0; i < 100; i++) {
    betaValues[i] = i * myStep;
  }
  ofstream identityFile("Ising3x3ExactIdentities.csv");
  ofstream expectationFile("Ising3x3ExactExpectations.csv");
  expectationFile << "beta,E,M" << "\n";
  identityFile << "beta,";
  if (expectationFile.is_open() && identityFile.is_open()) {
    vector < double > BoltzmannFactors(nStates);
    vector < double > probs(nStates);
    vector < double > energies(nStates);
    vector < int > mags(nStates);
    double partitionFunction;
    int mag;
    double energy;
    tuple < double, int > res;
    double muEnergy;
    double muMag;
    for (int id = 0; id < nStates; id++) {
      cout << 100.0 * id / nStates << "%" << "\n";
      res = getEnergyMag(id);
      energy = get < 0 > (res);
      mag = get < 1 > (res);
      energies[id] = energy;
      mags[id] = -1 * mag;
      if (id < nStates - 1) {
        identityFile << id << ",";
      } else {
        identityFile << id << "\n";
      }
    }
    for (int bInd = 0; bInd < 100; bInd++) {
      double beta = betaValues[bInd];
      transform(energies.begin(), energies.end(), BoltzmannFactors.begin(),
        [beta](double x) {
          return evalBoltzmann(x, beta);
        });
      partitionFunction = accumulate(BoltzmannFactors.begin(),
        BoltzmannFactors.end(), 0.0);
      transform(BoltzmannFactors.begin(), BoltzmannFactors.end(),
        probs.begin(), [partitionFunction](double x) {
          return (x / partitionFunction);
        });
      /*
	The mean (or expectation value) for a parameter
	is the total of the value of the parameter for 
	any state multiplied by each respective state's
	probability. For simplicity, a dot product is used.
      */
      muEnergy = inner_product(probs.begin(), probs.end(), energies.begin(), 0.0);
      muMag = inner_product(probs.begin(), probs.end(), mags.begin(), 0.0);
      expectationFile << beta << "," << muEnergy << "," << muMag << "\n";
      identityFile << beta << ",";
      for (int id = 0; id < nStates - 1; id++) {
        identityFile << probs[id] << ",";
      }
      identityFile << probs[nStates - 1] << "\n";
    }
  }
  return 0;
}
