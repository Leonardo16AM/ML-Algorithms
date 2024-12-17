// g++ -std=c++11 -O2 -o nb_iris nb_iris.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <random>
#include <chrono>

using namespace std;
#define M_PI 3.14159265358979323846


struct IrisSample{
    vector<double> features;
    string label;
};

vector<IrisSample> loadIrisData(const string& filename) {
    vector<IrisSample> dataset;
    ifstream file(filename);
    if(!file.is_open()) {
        cerr<<"Cannot open file: "<<filename<<endl;
        return dataset;
    }
    string line;
    while(getline(file,line)){
        if(line.empty())continue;
        stringstream ss(line);
        string val;
        vector<double>features;
        string label;

        for(int i=0;i<4;i++){
            if(!getline(ss,val,','))break;
            features.push_back(stod(val));
        }

        if(getline(ss,val,',')){
            label=val;
        }else{
            continue;
        }

        IrisSample sample;
        sample.features = features;
        sample.label = label;
        dataset.push_back(sample);
    }
    return dataset;
}

class NaiveBayes{
public:
    void fit(const vector<IrisSample>& data) {
        map<string, vector<vector<double>>> class_data;
        for(auto& sample : data) {
            class_data[sample.label].push_back(sample.features);
        }

        int feature_count = (int)data[0].features.size();

        for(auto& kv: class_data) {
            const string& c = kv.first;
            const auto& samples = kv.second;
            int n=(int)samples.size();

            vector<double> means(feature_count,0.0);
            vector<double> variances(feature_count,0.0);

            for(auto& feat_vec: samples){
                for(int i=0;i<feature_count;i++){
                    means[i]+= feat_vec[i];
                }
            }
            for(int i=0;i<feature_count;i++){
                means[i]/=n;
            }

            for(auto& feat_vec: samples) {
                for(int i=0;i<feature_count;i++){
                    double diff = feat_vec[i]-means[i];
                    variances[i]+= diff*diff;
                }
            }
            for(int i=0;i<feature_count;i++){
                variances[i]= variances[i]/(n-1);
                if(variances[i]<1e-9){
                    variances[i] = 1e-9;
                }
            }

            class_stats[c].means = means;
            class_stats[c].variances = variances;
            class_counts[c] = n;
        }

        int total_samples = (int)data.size();
        for(auto& kv: class_counts){
            class_priors[kv.first] = (double)kv.second/total_samples;
        }
    }

    string predict(const IrisSample& sample)const{
        string best_class;
        double best_log_prob = -INFINITY;

        for(auto& kv: class_stats) {
            const string& c = kv.first;
            const auto& stats = kv.second;

            // log(P(Class))
            double log_prob = log(class_priors.at(c));

            for(size_t i=0;i<stats.means.size();i++){
                double mean = stats.means[i];
                double var = stats.variances[i];
                double x = sample.features[i];
                // PDF Gaussiana: (1 / sqrt(2*pi*var)) * exp(-(x-mean)^2/(2*var))
                // log(P(x|c)) = -0.5*log(2*pi*var) - (x-mean)^2/(2*var) (para evitar el underflow numerico)
                double diff = x-mean;
                double part = -0.5*log(2*M_PI*var)-(diff*diff)/(2*var);
                log_prob+= part;
            }

            if(log_prob>best_log_prob) {
                best_log_prob=log_prob;
                best_class=c;
            }
        }

        return best_class;
    }

private:
    struct ClassStats {
        vector<double> means;
        vector<double> variances;
    };
    map<string,ClassStats> class_stats;
    map<string,int> class_counts;
    map<string,double> class_priors;
};

double kFoldCrossValidation(const vector<IrisSample>& data, int folds) {
    vector<IrisSample> shuffled = data;
    unsigned seed = (unsigned)chrono::system_clock::now().time_since_epoch().count();
    shuffle(shuffled.begin(),shuffled.end(),default_random_engine(seed));

    int fold_size = (int)shuffled.size()/folds;
    double total_accuracy = 0.0;

    for(int i=0;i<folds;i++) {
        int start = i*fold_size;
        int end = (i==folds-1) ? (int)shuffled.size() : (i+1)*fold_size;

        vector<IrisSample>train_data;
        vector<IrisSample>test_data;

        for(int idx=0;idx<(int)shuffled.size();idx++){
            if(idx>=start && idx<end){
                test_data.push_back(shuffled[idx]);
            }else{
                train_data.push_back(shuffled[idx]);
            }
        }

        NaiveBayes nb;
        nb.fit(train_data);

        int correct=0;
        for(auto& sample: test_data){
            string pred = nb.predict(sample);
            if(pred == sample.label)correct++;
        }
        double accuracy = (double)correct/test_data.size();
        total_accuracy+= accuracy;
    }

    return total_accuracy/folds;
}

int main() {
    string filename = "iris/iris.data";
    auto data = loadIrisData(filename);
    if(data.empty()) {
        cerr<<"No data loaded. Exiting."<<endl;
        return 1;
    }

    int folds = 5;
    double avg_accuracy = kFoldCrossValidation(data, folds);

    cout<<"Average accuracy over "<<folds<<"-fold cross-validation (Naive Bayes) is: "<<(avg_accuracy*100.0)<<"%\n";

    return 0;
}
