// g++ -std=c++11 -O2 -o dt_iris dt_iris.cpp
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

vector<IrisSample> loadIrisData(const string& filename) {
    vector<IrisSample>dataset;
    ifstream file(filename);
    if(!file.is_open()){
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
            if (!getline(ss,val,',')) break;
            features.push_back(stod(val));
        }

        if (getline(ss,val,',')) {
            label=val;
        }else{
            continue;
        }

        IrisSample sample;
        sample.features=features;
        sample.label=label;
        dataset.push_back(sample);
    }
    return dataset;
}

double entropy(const vector<IrisSample>& data) {
    map<string,int>counts;
    for (auto& d:data) {
        counts[d.label]++;
    }

    double total=(double)data.size();
    double ent=0.0;
    for (auto& kv:counts) {
        double p=kv.second/total;
        ent-=p*log2(p);
    }
    return ent;
}

struct Node {
    bool is_leaf;
    string label; 
    int feature_index; 
    double threshold;
    Node* left;
    Node* right;
    Node():is_leaf(false),feature_index(-1),threshold(0.0),left(NULL),right(NULL){}
};

void splitData(const vector<IrisSample>& data, int feature_index, double threshold,
               vector<IrisSample>& left_data, vector<IrisSample>& right_data) {
    for(auto& sample: data){
        if(sample.features[feature_index]<=threshold) {
            left_data.push_back(sample);
        }else{
            right_data.push_back(sample);
        }
    }
}


struct SplitResult {
    int feature_index;
    double threshold;
    double info_gain;
};

SplitResult findBestSplit(const vector<IrisSample>& data) {
    double base_entropy = entropy(data);
    int num_features=(int)data[0].features.size();

    SplitResult best;
    best.feature_index=-1;
    best.threshold=0.0;
    best.info_gain=0.0;

    for (int f=0;f<num_features;f++) {
        vector<double>vals;
        for (auto& d: data) {
            vals.push_back(d.features[f]);
        }
        sort(vals.begin(),vals.end());
        vals.erase(unique(vals.begin(), vals.end()), vals.end());

        if (vals.size()==1)continue;

        for (size_t i=0;i<vals.size()-1;i++) {
            double threshold=(vals[i]+vals[i+1])/2.0;

            vector<IrisSample> left_data,right_data;
            splitData(data,f,threshold,left_data,right_data);

            if(left_data.empty()||right_data.empty())continue;

            double p_left = (double)left_data.size()/data.size();
            double p_right = 1.0-p_left;
            double new_entropy = p_left*entropy(left_data)+p_right*entropy(right_data);

            double info_gain = base_entropy-new_entropy;
            if(info_gain>best.info_gain){
                best.info_gain=info_gain;
                best.feature_index=f;
                best.threshold=threshold;
            }
        }
    }

    return best;
}

bool allSameLabel(const vector<IrisSample>& data) {
    if(data.empty()) return true;
    string first_label=data[0].label;
    for (auto& d: data){
        if(d.label!=first_label)return false;
    }
    return true;
}

string majorityLabel(const vector<IrisSample>& data) {
    map<string,int> counts;
    for (auto& d: data) {
        counts[d.label]++;
    }
    int max_count = -1;
    string best_label;
    for (auto& kv: counts) {
        if(kv.second>max_count){
            max_count = kv.second;
            best_label = kv.first;
        }
    }
    return best_label;
}

class DecisionTree {
public:
    DecisionTree():root(NULL){}

    void fit(const vector<IrisSample>& data) {
        root = buildTree(data);
    }

    string predict(const IrisSample& sample) const {
        return predictSample(root, sample);
    }

    ~DecisionTree(){
        freeNode(root);
    }

private:
    Node* root;

    Node* buildTree(const vector<IrisSample>& data) {
        if(data.empty())return NULL;
        if(allSameLabel(data)) {
            Node* leaf = new Node();
            leaf->is_leaf = true;
            leaf->label = data[0].label;
            return leaf;
        }

        SplitResult split = findBestSplit(data);
        if(split.feature_index==-1){
            Node* leaf = new Node();
            leaf->is_leaf = true;
            leaf->label = majorityLabel(data);
            return leaf;
        }

        vector<IrisSample> left_data, right_data;
        splitData(data,split.feature_index,split.threshold,left_data,right_data);

        if(left_data.empty()||right_data.empty()){
            Node* leaf = new Node();
            leaf->is_leaf = true;
            leaf->label = majorityLabel(data);
            return leaf;
        }

        Node* node = new Node();
        node->feature_index = split.feature_index;
        node->threshold = split.threshold;
        node->left = buildTree(left_data);
        node->right = buildTree(right_data);

        if(!node->left || !node->right){
            Node* leaf = new Node();
            leaf->is_leaf = true;
            leaf->label = majorityLabel(data);
            freeNode(node->left);
            freeNode(node->right);
            delete node;
            return leaf;
        }

        return node;
    }

    string predictSample(Node* node,const IrisSample& sample)const{
        if (!node) return ""; 
        if (node->is_leaf) return node->label;

        if(sample.features[node->feature_index] <= node->threshold){
            return predictSample(node->left, sample);
        } else {
            return predictSample(node->right, sample);
        }
    }

    void freeNode(Node* node) {
        if (!node) return;
        freeNode(node->left);
        freeNode(node->right);
        delete node;
    }
};

double kFoldCrossValidation(const vector<IrisSample>& data,int folds) {
    vector<IrisSample> shuffled = data;
    unsigned seed = (unsigned)chrono::system_clock::now().time_since_epoch().count();
    shuffle(shuffled.begin(),shuffled.end(),default_random_engine(seed));

    int fold_size = (int)shuffled.size()/folds;
    double total_accuracy = 0.0;

    for(int i=0;i<folds;i++){
        int start = i*fold_size;
        int end = (i==folds-1) ? (int)shuffled.size() : (i+1)*fold_size;

        vector<IrisSample> train_data;
        vector<IrisSample> test_data;

        for(int idx=0;idx<(int)shuffled.size(); idx++){
            if(idx>=start && idx<end){
                test_data.push_back(shuffled[idx]);
            }else{
                train_data.push_back(shuffled[idx]);
            }
        }

        DecisionTree dt;
        dt.fit(train_data);

        int correct=0;
        for(auto& sample: test_data){
            string pred = dt.predict(sample);
            if(pred == sample.label)correct++;
        }
        double accuracy = (double)correct/test_data.size();
        total_accuracy += accuracy;
    }

    return total_accuracy/folds;
}

int main() {
    string filename = "iris/iris.data";
    auto data = loadIrisData(filename);
    if(data.empty()){
        cerr<<"No data loaded. Exiting."<<endl;
        return 1;
    }

    int folds = 5;
    double avg_accuracy = kFoldCrossValidation(data,folds);

    cout<<"Average accuracy over "<<folds<<"-fold cross-validation (ID3 Decision Tree) is: "<<(avg_accuracy*100.0)<<"%\n";

    return 0;
}
