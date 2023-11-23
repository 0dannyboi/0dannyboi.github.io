#ifndef ISING_PACKAGE_H
#define ISING_PACKAGE_H

// Implementation of Welford's Algorithm courtesy of John D. Cook, 
// see https://www.johndcook.com/blog/standard_deviation/
#include "running_stat.h" 

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <execution>
#include <omp.h>
#include <time.h>

using namespace std;

class Ising {
private:
	int N, NRenorm;
	double J;
	vector<vector<int>> neighborInds;
	random_device rd;
	double energy;
	double M;
	double energyR;
	double mR;
	mt19937 gen;
	mt19937 myShuffler;
	uniform_real_distribution<double> dis;
    uniform_int_distribution<int> spinSelector;
	int nRs;

public:
	int NSteps = 2500;
	int startCollect = 1000;
	int Navg = 1;
	int spinCount, NRblockCount;
    vector<int> state;
    vector<vector<int>> renormBlocks;
	// Constructor
	Ising(int N, double J): N(N), J(J){
		spinCount = static_cast<int>(pow(N, 2));
		state.resize(spinCount);
		dis = uniform_real_distribution<double>(0.0, 1.0);
        spinSelector = uniform_int_distribution<int>(0, spinCount - 1);
		mt19937 myShuffler(rd());
		mt19937 gen(rd());
	}

	void configRenormBlocks(int nR){
		NRenorm = N / nR;
		nRs = static_cast<int>(pow(nR, 2));
		NRblockCount = static_cast<int>(pow(NRenorm, 2));
		for (int i = 0; i < NRblockCount; i++){
			vector<int> blockMembers;
			int blockCol = i % NRenorm;
			int blockRow = (i - blockCol) / NRenorm;
			int firstRow = blockRow * nR;
			int firstCol = blockCol * nR;
			for (int r = 0; r < nR; r++){
				int row = firstRow + r;
				for (int c = 0; c < nR; c++){
					int col = firstCol + c;
					int ind = N * row + col;
					blockMembers.push_back(ind);
				}
			}
			renormBlocks.push_back(blockMembers);
		}
	}

	
	void evalRenorm(const double externalField = 0.0){
		int myEnergyR = 0;
		mR = 0.0;
		for (int j = 0; j < NRblockCount; j++){
			vector<int> members = renormBlocks[j];
			int spinTotal1R = 0;
			int spinTotal2R = 0;
			for (int i = 0; i < nRs; i++){
				spinTotal1R += state[members[i]];
			}
			int spin1R = 2 * (spinTotal1R > 0) - 1;
            myEnergyR -= spin1R * externalField;
			mR += static_cast<double>(spin1R) / NRblockCount;
			int colR = j % NRenorm;
			int rowR = (j - colR) / NRenorm;
			if (colR < NRenorm - 1) {
				vector<int> leftMembers = renormBlocks[j + 1];
				for (int i = 0; i < nRs; i++){
					spinTotal2R += state[leftMembers[i]];
				}
				int spin2R = 2 * (spinTotal2R > 0) - 1;
				myEnergyR += spin1R * spin2R;
			}
			if (rowR < NRenorm - 1) {
				vector<int> belowMembers = renormBlocks[j + NRenorm];
				int spinTotal2R = 0;
				for (int i = 0; i < nRs; i++){
					spinTotal2R += state[belowMembers[i]];
				}
				int spin2R = 2 * (spinTotal2R > 0) - 1;
				myEnergyR += spin1R * spin2R;
			}
		}
		energyR = -1.0 * myEnergyR * J;
	}

    vector<int> randomState(double spinProportion = 0.0) {
        vector<int> myState(spinCount);
        if (spinProportion == 0.0) {
            uniform_int_distribution<int> upDown(0, 1);
            for (int i = 0; i < spinCount; i++){
			    myState[i] =  2 * upDown(gen) - 1;
		    }
        } else {
            int spinUp;
            if (spinProportion > 1.0){
                spinUp = static_cast<int>(spinProportion);
            } else {
                spinUp = static_cast<int>(spinProportion * spinCount);
            }
            for (int i = 0; i < spinUp; i++){
                myState[i] = 1;
            }
            for (int i = spinUp; i < spinCount; i++){
                myState[i] = -1;
            }
            shuffle(myState.begin(), myState.end(), myShuffler);
        }
        return myState;
    }
	
	void setState(vector<int>& inputState){
		for (int i = 0; i < spinCount; i++){
			state[i] = inputState[i];
		}
	}

    void setRandomState(){
		vector<int> myRandomState = randomState();
        setState(myRandomState);
	}

	void configure(const double externalField = 0.0){
		double myEnergy = 0.0;
		M = 0.0;
		int spin1;
		int spin2;
		/*
       	   	   To prevent double counting on a rectangular
       	   	   lattice, the neighbors to the right of
           	   and/or below a site are accounted for
           	   at each site.
    	*/
		for (int i = 0; i < spinCount; i++){
			vector<int> subInds;
			spin1 = state[i];
            myEnergy -= - spin1 * externalField;
			M += spin1 / static_cast<double>(spinCount);
			int col = i % N;
			int row = (i - col) / N;
			if (col < N - 1) {
				spin2 = state[i + 1];
				myEnergy += spin1 * spin2;
				subInds.push_back(i + 1);
			}
			if (row < N - 1) {
				spin2 = state[i + N];
				myEnergy += spin1 * spin2;
				subInds.push_back(i + N);
			}
			if (col > 0) {
				subInds.push_back(i - 1);
			}
			if (row > 0) {
				subInds.push_back(i - N);
			}
			neighborInds.push_back(subInds);
		}
		energy = -1.0 * myEnergy * J;
	}


	void configureCustom(vector<vector<int>> customNeighbors,const double externalField = 0.0){
		double myEnergy = 0.0;
		M = 0.0;
		int spin1;
		int spin2;
		neighborInds = customNeighbors;
		for (int i = 0; i < spinCount; i++){
			vector<int> subNeighbors = neighborInds[i];
			spin1 = state[i];
            myEnergy += spin1 * externalField;
			M += spin1 / static_cast<double>(spinCount);
			for (int j = 0; j < subNeighbors.size(); j++){
				spin2 = subNeighbors[j];
				myEnergy += 0.5 * spin1 * spin2;
			}
		}
		energy = -1.0 * myEnergy * J;
	}

	double getCustomEnergy(const double externalField = 0.0){
		double myEnergy = 0.0;
		int spin1;
		int spin2;
		for (int i = 0; i < spinCount; i++){
			vector<int> subNeighbors = neighborInds[i];
			spin1 = state[i];
            myEnergy += spin1 * externalField;
			for (int j = 0; j < subNeighbors.size(); j++){
				spin2 = subNeighbors[j];
				myEnergy += 0.5 * spin1 * spin2;
			}
		}
		myEnergy *= (-1.0 * J);
		return myEnergy;
	}

	double getCorrelation(){
		double myCorr = pow(M, 2);
		double myWeight = 0.5 / pow(spinCount, 2);
		int spin1, spin2;
		for (int i = 0; i < spinCount; i++){
			spin1 = state[i];
			vector<int> subNeighbors = neighborInds[i];
			for (int j = 0; j < subNeighbors.size(); j++){
				spin2 = state[j];
				myCorr -= myWeight * spin1 * spin2;
			}
		}
		return myCorr;
	}

	double showEnergy(){
		return energy;
	}

	double getEnergy(const double externalField = 0.0){
		double myEnergy = 0.0;
		for (int i = 0; i < spinCount; i++){
			int spin1 = state[i];
			myEnergy -= spin1 * externalField;
			int col = i % N;
			int row = (i - col) / N;
			if (col < N - 1) {
				int spin2 = state[i + 1];
				myEnergy += 1.0 * spin1 * spin2;
			}
			if (row < N - 1) {
				int spin2 = state[i + N];
				myEnergy += 1.0 * spin1 * spin2;
			}
		}
		myEnergy *= (-1.0 * J);
		return myEnergy;
	}

	void updateMetro(const double beta, const double externalField = 0.0) {
		double deltaE, BoltzmannFactor, randomValue;
		int spin = spinSelector(gen);//rand() % spinCount; // index of spin to try flipping
        int currentSpin = state[spin];
		int neighborSpins = 0;
        for (const auto& neighbor: neighborInds[spin]){
            neighborSpins += state[neighbor];
        }
		deltaE = 2.0 * (J * neighborSpins + externalField) * currentSpin;
		// Accept 100% of moves that lower the energy.
		if (deltaE < 0){
			energy += deltaE;
			M -= (2.0 * currentSpin / static_cast<double>(spinCount));
			state[spin] *= -1;
		} else {
			/*
		    	    Accept configuration if random number is less
		            than acceptance ratio.
			*/
			randomValue = dis(gen);
			BoltzmannFactor = (-beta * deltaE);
			if (BoltzmannFactor > log(randomValue)){
				energy += deltaE;
				M -= (2.0 * currentSpin / static_cast<double>(spinCount));
				state[spin] *= -1;
			}
		}
	}

	void updateWolff(double beta) {
		vector<int> cluster;
		int seed = spinSelector(gen); // index of cluster seed
		int seedValue = state[seed];
		vector<int> spinNeighbors;
		double bondProb = 1.0 - exp(- 2.0 * beta * J);
		cluster.push_back(seed);
		state[seed] *= -1;
		M -= (2.0 * seedValue / static_cast<double>(spinCount));
		int i = 0;
		while (i < cluster.size()) {
			int currentSpin = cluster[i];
			spinNeighbors = neighborInds[currentSpin];
			shuffle(spinNeighbors.begin(), spinNeighbors.end(), myShuffler);
			for (int j = 0; j < spinNeighbors.size(); j++){
				int myNeighbor = spinNeighbors[j];
				if (state[myNeighbor] == seedValue){
					if (myNeighbor == seed){
						continue;
					}
					bool inValid = false;
					for (int h = 0; h < i; h++) {
					if (cluster[h] == myNeighbor){
						inValid = true;
						break;
						}
					}
					if (inValid){
						continue;
					}
					for (int h = 1 + i; h < cluster.size(); h++) {
						if (cluster[h] == myNeighbor){
							inValid = true;
							break;
						}
					}
					if (inValid){
						continue;
					}
					if (dis(gen) < bondProb) {
						cluster.push_back(myNeighbor);
						M -= (2.0 * seedValue / static_cast<double>(spinCount));
						state[myNeighbor] *= -1;
					}
				}
			}
			i++;
		}
	}


	double showM(bool recompute = false) {
        if (recompute){
            int spinSum = accumulate(state.begin(), state.end(), 0);
            M = spinSum / spinCount;
        }
		return M;
	}

	double showEnergyR() {
		return energyR;
	}

	double showMR() {
		return abs(mR);
	}
	
	void setNSteps(int n){
		NSteps = n;
	}

	void setStartCollect(int n){
		startCollect = n;
	}
	
	void setNavg(int n){
		Navg = n;
	}
	
	void runWolff(string fileName, vector<double> betaRange, vector<int> initialState = {}){
        if (initialState.empty()){
		    initialState = randomState();
        }
        setState(initialState);
		double averaging = static_cast<double>(Navg * (NSteps - startCollect));
		ofstream outputFile(fileName);
		if (outputFile.is_open()){
			outputFile << "Beta,M,M_STD,E,E_STD,E_Squared,E_SquaredSTD,M_Squared,M_SquaredSTD";
			outputFile << "\n";
			int len = betaRange.size();
			for(int bInd = 0; bInd < len; bInd++) {
				double beta = betaRange[bInd];
				cout << (100.0 * bInd) / len << "%" << "\n";
				cout << beta << "\n";
				RunningStat runningE;
				RunningStat runningM;
				RunningStat runningES;
                RunningStat runningMS;
				for (int iter = 0; iter < Navg; iter++){
					setRandomState();
					configure();
					for (int step = 0; step < startCollect; step ++){
						updateWolff(beta);
					}
					for (int step = startCollect; step < NSteps; step ++){
						updateWolff(beta);
						double tempE = getEnergy();
						double tempM = abs(showM());
						runningE.Push(tempE);
						runningM.Push(tempM);
						runningES.Push(pow(tempE, 2));
                        runningMS.Push(pow(tempM, 2));
					}
				}
				double MAvg = runningM.Mean();
				double EAvg = runningE.Mean();
				double ESAvg = runningES.Mean();
                double MSAvg = runningMS.Mean();
				double Mstd = runningM.StandardDeviation();
				double Estd = runningE.StandardDeviation();
				double ESstd = runningES.StandardDeviation();
                double MSstd = runningMS.StandardDeviation();
				outputFile << beta << "," << MAvg << "," << Mstd << "," << EAvg << "," << Estd;
				outputFile << "," << ESAvg << "," << ESstd << "," << MSAvg << "," << MSstd << "\n";
			};
		}
	}

    void runMetro(string fileName, vector<double> betaRange, vector<double> fieldRange = {0.0}, vector<int> initialState = {}){
        if (initialState.empty()){
		    initialState = randomState();
        }
        double averaging = static_cast<double>(Navg * (NSteps - startCollect) * spinCount);
        ofstream outputFile(fileName);
        if (outputFile.is_open()){
			outputFile << "Beta,Field,M,M_STD,E,E_STD,E_Squared,E_SquaredSTD,M_Squared,M_SquaredSTD";
			outputFile << "\n";
			int betaLen = betaRange.size();
            int fieldLen = fieldRange.size();
            #pragma omp parallel for
            for(int bInd = 0; bInd < betaLen; bInd++) {
                double beta = betaRange[bInd];
                for (int fInd = 0; fInd < fieldLen; fInd++){
                    double externalField = fieldRange[fInd];
                    RunningStat runningE;
				    RunningStat runningM;
				    RunningStat runningES;
                    RunningStat runningMS;
                    for (int n = 0; n < Navg; n++){
                        setRandomState();
                        configure(externalField);
                        for (int step = 0; step < startCollect; step ++){
                            for (int k = 0; k < spinCount; k++){
                                updateMetro(beta, externalField);
                            }
                        }
                        for (int step = startCollect; step < NSteps; step ++){
                            for (int k = 0; k < spinCount; k++){
                                updateMetro(beta, externalField);
                            }
                        }
                    }
                    double MAvg = runningM.Mean();
				    double EAvg = runningE.Mean();
				    double ESAvg = runningES.Mean();
                    double MSAvg = runningMS.Mean();
				    double Mstd = runningM.StandardDeviation();
				    double Estd = runningE.StandardDeviation();
				    double ESstd = runningES.StandardDeviation();
                    double MSstd = runningMS.StandardDeviation();
				    outputFile << beta << "," << externalField << "," << MAvg << "," << Mstd << "," << EAvg << "," << Estd;
				    outputFile << "," << ESAvg << "," << ESstd << "," << MSAvg << "," << MSstd << "\n";
                }
            }
        }

    }

	
	
};


#endif
