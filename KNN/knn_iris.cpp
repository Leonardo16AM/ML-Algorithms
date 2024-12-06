// g++ -std=c++11 -O2 -o knn_iris knn_iris.cpp
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

struct IrisSample {
    vector<double> features;
    string label;
};

class KNNClassifier {
public:
    KNNClassifier(int k):K(k){}

    void fit(const vector<IrisSample>& training_data) {
        data = training_data;
    }

    string predict(const IrisSample& sample) const {
        vector<pair<double,string>> distances;
        distances.reserve(data.size());

        for (const auto& train_pt: data) {
            double dist=euclideanDistance(sample.features,train_pt.features);
            distances.push_back({dist, train_pt.label});
        }

        sort(distances.begin(),distances.end(),
                [](const pair<double,string>& a,
                    const pair<double,string>& b) {
                    return a.first<b.first;
                });

        map<string, int> countMap;
        for (int i=0;i<K;++i) {
            countMap[distances[i].second]++;
        }

        int max_count=-1;
        string best_label;
        for (auto& kv:countMap) {
            if (kv.second>max_count) {
                max_count=kv.second;
                best_label=kv.first;
            }
        }
        return best_label;
    }

private:
    int K;
    vector<IrisSample> data;

    double euclideanDistance(const vector<double>& a,const vector<double>& b)const{
        double sum=0.0;
        for (size_t i=0;i<a.size();++i) {
            double diff=a[i]-b[i];
            sum+=diff*diff;
        }
        return sqrt(sum);
    }
};


vector<IrisSample> loadIrisData(const string& filename) {
    vector<IrisSample> dataset;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr<<"Cannot open file: "<<filename<<endl;
        return dataset;
    }
    string line;
    while (getline(file, line)) {
        if (line.empty())continue;
        stringstream ss(line);
        string val;
        vector<double>features;
        string label;

        for (int i=0;i<4;i++) {
            if(!getline(ss,val,','))break;
            features.push_back(stod(val));
        }

        if (getline(ss, val, ',')) {
            label = val;
        } else {
            continue;
        }

        IrisSample sample;
        sample.features=features;
        sample.label=label;
        dataset.push_back(sample);
    }
    return dataset;
}

double kFoldCrossValidation(const vector<IrisSample>& data, int K, int folds) {
    vector<IrisSample> shuffled=data;
    unsigned seed=(unsigned)chrono::system_clock::now().time_since_epoch().count();
    shuffle(shuffled.begin(),shuffled.end(),default_random_engine(seed));

    int fold_size=(int)shuffled.size()/folds;
    double total_accuracy=0.0;

    for (int i=0; i<folds;i++) {
        int start=i*fold_size;
        int end=(i==folds-1)?(int)shuffled.size():(i+1)*fold_size;

        vector<IrisSample> train_data;
        vector<IrisSample> test_data;

        for (int idx=0; idx<(int)shuffled.size();idx++) {
            if (idx>=start && idx<end) {
                test_data.push_back(shuffled[idx]);
            }else{
                train_data.push_back(shuffled[idx]);
            }
        }

        KNNClassifier knn(K);
        knn.fit(train_data);

        int correct=0;
        for (auto& sample:test_data) {
            string pred=knn.predict(sample);
            if (pred==sample.label) correct++;
        }
        double accuracy=(double)correct/test_data.size();
        total_accuracy+=accuracy;
    }

    return total_accuracy/folds;
}



int main() {
    string filename = "iris/iris.data";
    auto data = loadIrisData(filename);
    if (data.empty()) {
        cerr<<"No data loaded. Exiting."<<endl;
        return 1;
    }

    int K=5;
    int folds=5;
    double avg_accuracy=kFoldCrossValidation(data,K,folds);

    cout<<"Average accuracy over "<<folds<<"-fold cross-validation with K="<<K<<" is: "<<(avg_accuracy*100.0)<<"%\n";

    return 0;
}
